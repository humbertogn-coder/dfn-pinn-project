"""Rate-parametric Li-SPAN PINN (Phase 3a): one network for a whole range of C-rates.

Every field takes the C-rate r as an extra input, so after one training the voltage and the internal fields at any
rate in [rate_min, rate_max] cost one forward pass (no retraining per rate).  This is the building block of the
two-curve calibration tool (docs/PHASE2_LITERATURE_DATA.md, proposal task 4); here the cell parameters are fixed at
their nominal values and only the rate varies (Phase 3b adds chemistry parameters as inputs).

Coordinates.  Time is normalized per rate, T = t / t_end(r), so that all rates use T in [0, 1] and the discharge
phases sit at similar T.  The window t_end(r) = window_fraction * exp(poly(ln r)) / r stops just before the cut-off
collapse (the single-rate PINN does the same with 0.98 x the reference end time); poly is a low-order fit of
ln(r t_cut) in ln r to finite-volume cut-off times at a few anchor rates (window_rates).  These cut-off times are the
only information the network takes from the finite-volume solver: the fields are trained by the physics alone.

Physics and hard structure are those of LiSPANPINN (pinn.py, docstring there).  Quantities that scaled with the
protocol current in the single-rate model (residual normalizations, output scales of c_e and phi_e, current ramp,
short-time features, initial-condition factor) are per-point functions of r here, so the residuals of every rate are
O(1) and no rate dominates the loss.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn

from ..v2.model import MLP, gauss_legendre
from ..v2.train import AdaptiveWeights
from .params import F, R as RGAS, LiSPANParams
from .pinn import GROUPS, LiSPANPINN, LiSPANTrainConfig, _grad, huber

ONE_C = 10.0            # A/m2 per C-rate (LiSPANProtocol.from_crate: 1 C = 1 mA/cm2)


@dataclass
class RateTrainConfig(LiSPANTrainConfig):
    rate_min: float = 0.1
    rate_max: float = 1.0
    ramp_s: float = 30.0                  # current switch-on I = I0 tanh(t / ramp_s), as in the single-rate configs
    window_rates: tuple = (0.1, 0.2, 0.5, 1.0)                    # FV cut-off times that define t_end(r)
    window_degree: int = 2
    window_fraction: float = 0.98
    eval_rates: tuple = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)      # FV references for the evaluation only
    history_every: int = 10               # keep every n-th step in history.json (all logged steps are kept too)
    device: str = "cpu"                   # "cpu", "cuda" or "auto"
    stop_step: int = 0                    # stop (save latest.pt) after this step; 0 = run to adam_steps (replays, splits)
    # --- training stabilization (wave 2; off by default so that wave-1 runs resume unchanged, see GuardedWeights)
    weight_guard: bool = False            # a loss group that is getting worse may not lose adaptive weight
    weight_guard_rise: float = 3.0        #   "getting worse" = EMA of its loss > rise x the best EMA so far
    weight_guard_boost: float = 1.25      #   such a group's weight is multiplied by boost at each update instead
                                          #   (1.0 = hold: the weight may not decrease while the group is rising)
    weight_guard_ref: str = "best"        #   reference level: "best" = best EMA ever (wave 2; fires on ordinary noise and
                                          #   ratchets every weight to the cap) or "recent" = lowest 500-step median of the
                                          #   last weight_guard_horizon steps (as the rollback monitor)
    weight_guard_horizon: int = 5000
    adaptive_stop: int = 0                # > 0: from this step on the adaptive weights are frozen at the median of their
                                          #   values over the previous adaptive_freeze_window steps (wave 3: every late
                                          #   failure was a weight runaway set off by the charge inventory)
    adaptive_freeze_window: int = 5000
    rollback_start: int = 0               # rollbacks only from this step on (the monitor observes from the start); wave 3:
                                          #   the bursts before ~30 000 steps recover by themselves, rolling them back
                                          #   looped and used up the budget
    charge_weight_floor: float = 0.0      # > 0: after every adaptive update the charge-group weight is at least this
                                          #   fraction of the ode-group weight (wave 3: every excursion started with the
                                          #   charge weight falling from ~100-800 to ~0.2 while ode stayed at 300-1000)
    loss_ema_decay: float = 0.98          #   per-step EMA of the unweighted group losses (~50 steps)
    rollback: bool = False                # restore the last healthy in-memory snapshot when the kinetics blow up
    rollback_factor: float = 100.0        #   trigger: EMA(ode + charge losses) > factor x its running median ...
    rollback_floor: float = 1e-2          #   ... and above this absolute level (wave-1 histories: factor 100 fires
                                          #   at every lasting excursion and only 5x in the two runs that stayed good)
    snapshot_every: int = 500             #   healthy snapshot interval (steps)
    rollback_lr_factor: float = 0.5       #   learning rate multiplied by this after a rollback (again for every further
    rollback_cooldown: int = 2000         #   rollback within the cooldown) for this many steps, then back to the schedule
    max_rollbacks: int = 10
    rollback_budget_window: int = 0       # > 0: max_rollbacks counts only the rollbacks of the last this-many steps, so early
                                          #   rollback loops cannot use up the protection for the rest of the run (wave 3)
    rollback_horizon: int = 0             # wave 3: > 0 -> the rollback baseline is the lowest 500-step median of the last
                                          #   rollback_horizon steps (HealthMonitor), so a slow ramp cannot drag it along;
                                          #   0 = the wave-2 monitor (running median, follows slow ramps and never fires)
    keep_best: bool = True                # save best.pt / best_metrics.json at the evaluation with the lowest mean V error
                                          #   over the ANCHOR rates (window_rates); the other rates stay unseen by the choice

    def to_dict(self):
        d = super().to_dict()
        d["window_rates"] = list(self.window_rates)
        d["eval_rates"] = list(self.eval_rates)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RateTrainConfig":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


def fit_window(rates, t_cut, degree=2):
    """Coefficients (c0, c1, c2) of ln(r t_cut) ~ c0 + c1 ln r + c2 (ln r)^2 and the largest relative error of the
    fitted cut-off time at the anchors.  r t_cut is the cut-off charge per unit 1 C current (smooth in ln r)."""
    r = np.asarray(rates, float)
    x, y = np.log(r), np.log(r * np.asarray(t_cut, float))
    deg = int(min(degree, len(x) - 1, 2))
    coef = np.polynomial.polynomial.polyfit(x, y, deg)
    coef = np.pad(coef, (0, 3 - len(coef)))
    err = float(np.max(np.abs(np.exp(np.polynomial.polynomial.polyval(x, coef) - y) - 1.0)))
    return [float(c) for c in coef], err


def window_t_end(coef, rate, fraction=0.98):
    """t_end(r) [s] of the PINN window (numpy/float version of LiSPANRatePINN.t_end)."""
    x = math.log(rate)
    return fraction * math.exp(coef[0] + coef[1] * x + coef[2] * x * x) / rate


class LiSPANRatePINN(LiSPANPINN):
    """LiSPANPINN with the C-rate r (tensor, same shape as T) as an extra argument of every field."""

    def __init__(self, params: LiSPANParams, window_coef, rate_min=0.1, rate_max=1.0, ramp_s=30.0, window_fraction=0.98,
                 width=96, depth=4, act="tanh", fourier_t=8, fourier_period=1.0, short_t=(60.0, 600.0), ic_tau_s=60.0,
                 quad_order=12, salt_sep_natural_scale=False):
        nn.Module.__init__(self)                       # own networks (one more input); physics helpers are inherited
        self.p = p = params
        self.RT = RGAS * p.T
        self.f2 = F / (2.0 * self.RT)
        self.rate_min, self.rate_max = float(rate_min), float(rate_max)
        self.ramp_s = max(float(ramp_s), 1e-6)
        self.window_fraction = float(window_fraction)
        self.fourier_t, self.fourier_period = int(fourier_t), float(fourier_period)
        self.short_t = tuple(float(s) for s in short_t)
        self.ic_tau_s = float(ic_tau_s)
        self.register_buffer("window_coef", torch.as_tensor(list(window_coef)[:3], dtype=torch.float64))
        n_t = 2 + 2 * self.fourier_t + len(self.short_t) + 1          # time features + the rate feature
        self.net_R = MLP(1 + n_t, width, depth, n_out=1, act=act)
        self.net_xi = MLP(1 + n_t, width, depth, n_out=3, act=act)
        self.net_ce = MLP(3 + n_t, width, depth, n_out=1, act=act)
        self.net_phie = MLP(3 + n_t, width, depth, n_out=1, act=act)
        with torch.no_grad():                       # extents: P1 ~ 3 (phase 1 in a third of the window), P2, P3 lagging
            self.net_xi.net[-1].bias.copy_(torch.tensor([math.log(3.0), math.log(0.3), math.log(0.1)]))
        q, w = gauss_legendre(quad_order)
        self.register_buffer("q_nodes", q)
        self.register_buffer("q_weights", w)
        self.Yi = p.L_cat / p.L_tot
        self.kap_e0 = p.kappa0 * p.eps_e_sep ** p.beta_sep
        self.D_ion = {"Li": p.D_salt / (2 * (1 - p.t_plus)), "PF6": p.D_salt / (2 * p.t_plus)}
        self.salt_total_scale = 1.0
        self.charge_total_scale = 0.0
        self.salt_sep_natural = bool(salt_sep_natural_scale)
        self.log_mult = nn.ParameterDict({n: nn.Parameter(torch.zeros(()), requires_grad=False) for n in self.PARAMETERS})

    # ------------------------------------------------------------------ rate-dependent scalars (per point)
    def I0(self, r):
        return ONE_C * r

    def t_end(self, r):
        x = torch.log(r)
        c = self.window_coef.to(r.dtype)
        return self.window_fraction * torch.exp(c[0] + c[1] * x + c[2] * x * x) / r

    def rate_feature(self, r):
        lo, hi = math.log(self.rate_min), math.log(self.rate_max)
        return 2.0 * (torch.log(r) - lo) / max(hi - lo, 1e-12) - 1.0

    def ce_scale(self, r):
        return torch.clamp(0.03 * torch.clamp(self.I0(r), min=0.3), max=0.5)     # as LiSPANPINN: ~3 % per A/m2

    def phie_scale(self, r):
        return 2.0 * self.I0(r) * self.p.L_tot / self.kap_e0

    def charge_s_scale(self, r):
        return self.I0(r) / self.p.kappa_s_eff

    def charge_e_scale(self, r):
        return self.I0(r) / self.kap_e0

    def salt_scale(self, r):
        return (1.0 - self.p.t_plus) * self.I0(r) / (F * self.p.L_cat)

    def salt_scale_sep(self, r):
        s = self.salt_scale(r)
        return s * self.p.L_cat / self.p.L_sep if self.salt_sep_natural else s

    # ------------------------------------------------------------------ time features
    def g(self, T, r):
        return torch.tanh(T * self.t_end(r) / self.ramp_s)

    def I(self, T, r):
        return self.I0(r) * self.g(T, r)

    def charge_passed(self, T, r):
        """Q(t) = int_0^t I dt = I0 tau ln cosh(t/tau) [C/m2], overflow-free, t = T t_end(r)."""
        te = self.t_end(r)
        tau = self.ramp_s / te
        return self.I0(r) * te * (T + tau * (torch.log1p(torch.exp(-2.0 * T / tau)) - math.log(2.0)))

    def ic_factor(self, T, r):
        return 1 - torch.exp(-T * self.t_end(r) / self.ic_tau_s)

    def tf(self, T, r):
        te = self.t_end(r)
        feats = [2 * T - 1, 2 * self.g(T, r) - 1]
        for k in range(1, self.fourier_t + 1):
            w = 2 * math.pi * k / self.fourier_period
            feats += [torch.sin(w * T), torch.cos(w * T)]
        for tau in self.short_t:
            feats.append(2 * torch.exp(-T * te / tau) - 1)
        feats.append(self.rate_feature(r))
        return torch.cat(feats, dim=1)

    # ------------------------------------------------------------------ cathode fields
    def _remaining(self, Yc, T, r):
        P = torch.exp(torch.clamp(self.net_xi(torch.cat([2 * Yc - 1, self.tf(T, r)], dim=1)), -16.0, 6.0))
        return tuple(torch.exp(-torch.clamp(T * P[:, m:m + 1], max=60.0)) for m in range(3))

    def extents(self, Yc, T, r):
        e1, e2, e3 = self._remaining(Yc, T, r)
        xi1 = 1 - e1
        xi2 = xi1 * (1 - e2)
        xi3 = xi2 * (1 - e3)
        return xi1, xi2, xi3

    def species(self, Yc, T, r):
        p = self.p
        c40 = p.c_init[0]
        e1, e2, e3 = self._remaining(Yc, T, r)
        xi1 = 1 - e1
        xi2 = xi1 * (1 - e2)
        xi3 = xi2 * (1 - e3)
        c_S4 = c40 * e1
        c_S3 = c40 * xi1 * e2
        c_S2 = c40 * xi2 * e3
        c_S1 = c40 * (xi1 + xi3)
        eps_L = p.eps_L0 + p.V_m_L * c40 * (xi2 + xi3)
        eps_e = 1 - p.eps_SPAN - p.eps_CB - eps_L
        a = p.a_SPAN0 * (eps_e / p.eps_e0) ** p.xi
        return {"xi": (xi1, xi2, xi3), "c_S4": c_S4, "c_S3": c_S3, "c_S2": c_S2, "c_S1": c_S1, "eps_L": eps_L, "eps_e": eps_e, "a": a}

    def rho(self, Yc, T, r):
        return torch.exp(torch.clamp(self.net_R(torch.cat([2 * Yc - 1, self.tf(T, r)], dim=1)), -12.0, 12.0))

    def _a_rho_nodes(self, Yc_upper, T, r):
        n, q = T.shape[0], self.q_nodes.shape[0]
        nodes = self.q_nodes.to(T.dtype).view(1, q)
        Yq = (Yc_upper.view(n, 1) * nodes).reshape(-1, 1)
        Tq = T.view(n, 1).expand(n, q).reshape(-1, 1)
        rq = r.view(n, 1).expand(n, q).reshape(-1, 1)
        val = (self.species(Yq, Tq, rq)["a"] * self.rho(Yq, Tq, rq)).view(n, q)
        return Yc_upper.view(n, 1) * (val * self.q_weights.to(T.dtype).view(1, q)).sum(1, keepdim=True)

    def current_fields(self, Yc, T, r, sp=None):
        sp = sp or self.species(Yc, T, r)
        denom = self._a_rho_nodes(torch.ones_like(Yc), T, r)
        cum = self._a_rho_nodes(Yc, T, r)
        I = self.I(T, r)
        Rr = I / (F * self.p.L_cat) * self.rho(Yc, T, r) / denom
        return Rr, -I * cum / denom

    # ------------------------------------------------------------------ electrolyte fields
    def ce(self, Y, T, r):
        dev = torch.tanh(self.net_ce(torch.cat([self.efeat(Y), self.tf(T, r)], dim=1)))
        return self.p.c_Li0 * (1 + self.ce_scale(r) * self.ic_factor(T, r) * dev)

    def phie_anode(self, T, r):
        Y1 = torch.ones_like(T)
        a_Li = torch.clamp(self.ce(Y1, T, r) / self.p.c_Li0, 1e-6)
        eta = (2 * self.RT / F) * torch.asinh(self.I(T, r) / (2 * F * self.p.k0_Li * torch.sqrt(a_Li)))
        return -(self.RT / F) * torch.log(a_Li) - eta

    def phie(self, Y, T, r):
        return self.phie_anode(T, r) + (1 - Y) * self.phie_scale(r) * self.net_phie(torch.cat([self.efeat(Y), self.tf(T, r)], dim=1))

    # ------------------------------------------------------------------ voltage
    def voltage(self, T, r):
        Y0 = torch.zeros_like(T)
        sp = self.species(Y0, T, r)
        a_Li = torch.clamp(self.ce(Y0, T, r) / self.p.c_Li0, 1e-6)
        Rr, _ = self.current_fields(Y0, T, r, sp)
        d = self.dphi(Rr, sp, a_Li)
        return d + self.phie(Y0, T, r) - self.p.Z_CC * self.mult("Z_CC") * self.I(T, r)

    def terminal_voltage(self, T, r):
        return self.voltage(T, r)

    def discharge_curve(self, rate: float, n: int = 400, t=None) -> dict:
        """Voltage curve at one C-rate inside the trained range: {'t' [s], 'V' [V], 'capacity_mAh_gS', 't_end'}.
        Rates outside [rate_min, rate_max] are refused (the network was not trained there)."""
        if not (self.rate_min * (1 - 1e-9) <= rate <= self.rate_max * (1 + 1e-9)):
            raise ValueError(f"C-rate {rate:g} outside the trained range [{self.rate_min:g}, {self.rate_max:g}]")
        ref = next(self.parameters())
        te = window_t_end(self.window_coef.tolist(), rate, self.window_fraction)
        t = np.linspace(0.0, te, n) if t is None else np.asarray(t, float)
        T = torch.as_tensor(t / te, dtype=ref.dtype, device=ref.device).view(-1, 1)
        r = torch.full_like(T, float(rate))
        with torch.no_grad():
            V = self.voltage(T, r).view(-1).cpu().numpy()
            Q = self.charge_passed(T, r).view(-1).cpu().numpy()          # C/m2
        cap = Q / 3.6 / (self.p.m_S_model * 1e3)                          # mAh/g_S (1 mAh = 3.6 C)
        return {"t": t, "V": V, "capacity_mAh_gS": cap, "t_end": te, "rate": float(rate)}


# ====================================================================== residuals
def rate_residuals(model: LiSPANRatePINN, batch):
    """Residuals of pinn.residuals with per-point rates (batch: 'c' = (Yc, T, r), 's' = (Ys, T, r), 'b' = (T, r))."""
    p = model.p
    out = {}
    # ---------------- cathode
    Yc, T, r = batch["c"]
    Yc = Yc.clone().requires_grad_(True); T = T.clone().requires_grad_(True)
    te = model.t_end(r)
    Yg = Yc * p.L_cat / p.L_tot
    sp = model.species(Yc, T, r)
    c = model.ce(Yg, T, r)
    a_Li = torch.clamp(c / p.c_Li0, 1e-6)
    Rr, i_e = model.current_fields(Yc, T, r, sp)
    d = model.dphi(Rr, sp, a_Li)
    f = model.rates(d, sp, a_Li)
    c40 = p.c_init[0]
    for m, xi in enumerate(sp["xi"]):
        out[f"ode{m + 1}"] = _grad(xi, T) - te * sp["a"] * f[m] / (2 * c40)
    dc_dy = _grad(c, Yc) / p.L_cat
    B = F * (model.D_ion["Li"] - model.D_ion["PF6"]) * sp["eps_e"] ** p.beta_cat * dc_dy * model.mult("D_salt")
    kap = model.kappa_e(c, sp["eps_e"], p.beta_cat)
    phie = model.phie(Yg, T, r)
    dphie_dy = _grad(phie, Yc) / p.L_cat
    ddphi_dy = _grad(d, Yc) / p.L_cat
    I = model.I(T, r)
    out["charge_s"] = (ddphi_dy - ((I + i_e) / p.kappa_s_eff + (i_e + B) / kap)) / model.charge_s_scale(r)
    out["charge_e"] = (dphie_dy + (i_e + B) / kap) / model.charge_e_scale(r)
    D = model.D_eff(sp["eps_e"], p.beta_cat)
    dflux_dy = _grad(D * dc_dy, Yc) / p.L_cat
    dc_dt = _grad(c, T) / te
    deps_dt = _grad(sp["eps_e"], T) / te
    out["salt_c"] = (sp["eps_e"] * dc_dt + c * deps_dt - dflux_dy + (1 - p.t_plus) * sp["a"] * Rr) / model.salt_scale(r)
    # ---------------- separator (i_e = -I)
    Ys, Ts, rs = batch["s"]
    Ys = Ys.clone().requires_grad_(True); Ts = Ts.clone().requires_grad_(True)
    Yg_s = (p.L_cat + Ys * p.L_sep) / p.L_tot
    cs = model.ce(Yg_s, Ts, rs)
    dcs_dy = _grad(cs, Ys) / p.L_sep
    eps_s = torch.full_like(cs, p.eps_e_sep)
    Bs = F * (model.D_ion["Li"] - model.D_ion["PF6"]) * p.eps_e_sep ** p.beta_sep * dcs_dy * model.mult("D_salt")
    kap_s = model.kappa_e(cs, eps_s, p.beta_sep)
    phie_s = model.phie(Yg_s, Ts, rs)
    Is = model.I(Ts, rs)
    out["charge_e"] = torch.cat([out["charge_e"], (_grad(phie_s, Ys) / p.L_sep + (-Is + Bs) / kap_s) / model.charge_e_scale(rs)])
    Ds = model.D_eff(eps_s, p.beta_sep)
    dflux_s = _grad(Ds * dcs_dy, Ys) / p.L_sep
    out["salt_s"] = (p.eps_e_sep * _grad(cs, Ts) / model.t_end(rs) - dflux_s) / model.salt_scale_sep(rs)
    # ---------------- boundary / interface conditions on the salt
    Tb, rb = batch["b"]
    ss_b = model.salt_scale(rb)
    Y0 = torch.zeros_like(Tb).requires_grad_(True)
    out["bc_c0"] = _grad(model.ce(Y0, Tb, rb), Y0) / p.L_tot / (ss_b * p.L_cat / p.D_salt)
    Y1 = torch.ones_like(Tb).requires_grad_(True)
    c1 = model.ce(Y1, Tb, rb)
    D1 = model.D_eff(torch.full_like(c1, p.eps_e_sep), p.beta_sep)
    out["bc_an"] = (D1 * _grad(c1, Y1) / p.L_tot - (1 - p.t_plus) * model.I(Tb, rb) / F) / (ss_b * p.L_cat)
    eps_if = 1e-4
    Ym = torch.full_like(Tb, model.Yi - eps_if).requires_grad_(True)
    Yp = torch.full_like(Tb, model.Yi + eps_if).requires_grad_(True)
    cm, cp = model.ce(Ym, Tb, rb), model.ce(Yp, Tb, rb)
    sp_if = model.species(torch.ones_like(Tb), Tb, rb)
    Dm = model.D_eff(sp_if["eps_e"], p.beta_cat); Dp = model.D_eff(torch.full_like(cp, p.eps_e_sep), p.beta_sep)
    out["bc_if"] = (Dm * _grad(cm, Ym) - Dp * _grad(cp, Yp)) / p.L_tot / (ss_b * p.L_cat)
    # ---------------- global inventories (quadrature over the electrode at the boundary sample times)
    if model.salt_total_scale > 0 or model.charge_total_scale > 0:
        q = model.q_nodes.shape[0]; nb = Tb.shape[0]
        nodes = model.q_nodes.to(Tb.dtype).view(1, q); w = model.q_weights.to(Tb.dtype).view(1, q)
        Tq = Tb.view(nb, 1).expand(nb, q).reshape(-1, 1)
        rq = rb.view(nb, 1).expand(nb, q).reshape(-1, 1)
        Yc_q = nodes.expand(nb, q).reshape(-1, 1)
    if model.salt_total_scale > 0:
        spq = model.species(Yc_q, Tq, rq)
        inv_c = ((spq["eps_e"] * model.ce(Yc_q * p.L_cat / p.L_tot, Tq, rq)).view(nb, q) * w).sum(1, keepdim=True) * p.L_cat
        Ys_q = ((p.L_cat + nodes * p.L_sep) / p.L_tot).expand(nb, q).reshape(-1, 1)
        inv_s = (model.ce(Ys_q, Tq, rq).view(nb, q) * w).sum(1, keepdim=True) * p.eps_e_sep * p.L_sep
        inv0 = p.eps_e0 * p.c_Li0 * p.L_cat + p.eps_e_sep * p.c_Li0 * p.L_sep
        out["salt_total"] = (inv_c + inv_s - inv0) / (model.salt_total_scale * p.L_tot)
    if model.charge_total_scale > 0:
        xi1, xi2, xi3 = model.extents(Yc_q, Tq, rq)
        inv = p.c_init[0] * ((xi1 + xi2 + xi3).view(nb, q) * w).sum(1, keepdim=True)
        out["charge_total"] = (inv - model.charge_passed(Tb, rb) / (2.0 * F * p.L_cat)) / model.charge_total_scale
    return out


# ====================================================================== sampling / evaluation
class RateSampler:
    """Uniform in Y, time as the single-rate sampler (a fraction of the points in the early window), C-rate log-uniform
    in [rate_min, rate_max].  Drawn on the CPU (reproducible generator) and moved to the device."""

    def __init__(self, cfg: RateTrainConfig, dtype, gen, device):
        self.cfg, self.dtype, self.gen, self.device = cfg, dtype, gen, device
        self.lo, self.hi = math.log(cfg.rate_min), math.log(cfg.rate_max)

    def u(self, n):
        return torch.rand(n, 1, generator=self.gen, dtype=self.dtype)

    def times(self, n):
        t = self.u(n)
        m = int(round(self.cfg.early_fraction * n))
        t[:m] *= self.cfg.early_window
        return t

    def rates(self, n):
        return torch.exp(self.lo + (self.hi - self.lo) * self.u(n))

    def draw(self):
        c = self.cfg
        dev = self.device
        return {"c": (self.u(c.n_cathode).to(dev), self.times(c.n_cathode).to(dev), self.rates(c.n_cathode).to(dev)),
                "s": (self.u(c.n_separator).to(dev), self.times(c.n_separator).to(dev), self.rates(c.n_separator).to(dev)),
                "b": (self.times(c.n_boundary).to(dev), self.rates(c.n_boundary).to(dev))}


def evaluate_rate(model: LiSPANRatePINN, rate: float, ref: dict) -> dict:
    """Errors against a finite-volume reference at one C-rate (same metrics as pinn.evaluate)."""
    p = model.p
    prm = next(model.parameters())
    dtype, dev = prm.dtype, prm.device
    te = window_t_end(model.window_coef.tolist(), rate, model.window_fraction)
    t = np.asarray(ref["t"]); keep = t <= te
    t = t[keep]
    T = torch.as_tensor(t / te, dtype=dtype, device=dev).view(-1, 1)
    r = torch.full_like(T, float(rate))
    out = {"t_end_s": te, "t_cut_ref_s": float(np.asarray(ref["t"])[-1])}
    with torch.no_grad():
        V = model.voltage(T, r).view(-1).cpu().numpy()
        e = V - np.asarray(ref["V"])[keep]
        m = t > 100.0
        out.update({"V_rmse_mV": 1e3 * float(np.sqrt(np.mean(e[m] ** 2))), "V_max_mV": 1e3 * float(np.abs(e[m]).max())})
        yc = np.asarray(ref["y_c"]); nt, ny = len(t), len(yc)
        Yq = torch.as_tensor(yc / p.L_cat, dtype=dtype, device=dev).view(1, ny, 1).expand(nt, ny, 1).reshape(-1, 1)
        Tq = T.view(nt, 1, 1).expand(nt, ny, 1).reshape(-1, 1)
        rq = torch.full_like(Tq, float(rate))
        sp = model.species(Yq, Tq, rq)
        for key in ("c_S4", "c_S3", "c_S2", "c_S1"):
            pred = sp[key].view(nt, ny).cpu().numpy().T
            out[key + "_rmse"] = float(np.sqrt(np.mean((pred - np.asarray(ref[key])[:, keep]) ** 2)))
        y = np.asarray(ref["y"]); ng = len(y)
        Yg = torch.as_tensor(y / p.L_tot, dtype=dtype, device=dev).view(1, ng, 1).expand(nt, ng, 1).reshape(-1, 1)
        Tg = T.view(nt, 1, 1).expand(nt, ng, 1).reshape(-1, 1)
        rg = torch.full_like(Tg, float(rate))
        ce = model.ce(Yg, Tg, rg).view(nt, ng).cpu().numpy().T
        out["c_e_rmse"] = float(np.sqrt(np.mean((ce - np.asarray(ref["c_Li"])[:, keep]) ** 2)))
        out["c_e_max"] = float(np.abs(ce - np.asarray(ref["c_Li"])[:, keep]).max())
        phie = model.phie(Yg, Tg, rg).view(nt, ng).cpu().numpy().T
        out["phi_e_rmse_mV"] = 1e3 * float(np.sqrt(np.mean((phie - np.asarray(ref["phi_e"])[:, keep]) ** 2)))
    return out


def evaluate_rates(model: LiSPANRatePINN, refs: dict) -> dict:
    """{rate: reference} -> per-rate metrics ('<metric>@<rate>') plus the mean and worst voltage errors over the rates."""
    out, rms, mx = {}, [], []
    for rate in sorted(refs):
        ev = evaluate_rate(model, rate, refs[rate])
        out.update({f"{k}@{rate:g}": v for k, v in ev.items()})
        rms.append(ev["V_rmse_mV"]); mx.append(ev["V_max_mV"])
    if rms:
        out.update({"V_rmse_mean_mV": float(np.mean(rms)), "V_rmse_worst_mV": float(np.max(rms)), "V_max_worst_mV": float(np.max(mx))})
    return out


# ====================================================================== stabilization
class GuardedWeights(AdaptiveWeights):
    """Gradient-norm loss balancing (AdaptiveWeights) without its runaway.

    The plain rule sets w_g ~ mean_g' |grad L_g'| / |grad L_g|: a group whose residual starts to grow has a growing
    gradient and therefore gets a SMALLER weight, so nothing pulls it back (wave 1: charge weight 54 -> 10 while its
    residual went 1.5e-3 -> 66; charge-inventory weight ~0.2 while its residual went 1 -> 1e4).  Here a group whose
    EMA loss exceeds `rise` x its best EMA may not lose weight and is pushed up by `boost` per update (capped at wmax),
    which turns the positive feedback into a restoring one; otherwise the plain rule applies."""

    def __init__(self, groups, alpha=0.9, rise=3.0, boost=1.25, decay=0.98, wmax=1e3, ref="best", horizon=5000):
        super().__init__(groups, alpha)
        self.rise, self.boost, self.decay, self.wmax = float(rise), float(boost), float(decay), float(wmax)
        self.ref, self.horizon, self.n = ref, int(horizon), 0
        self.ema, self.best = {}, {}
        self.mon = {g: HealthMonitor(decay, factor=math.inf, floor=0.0, horizon=self.horizon) for g in groups} \
            if ref == "recent" else {}

    def observe(self, losses):
        self.n += 1
        for g, names in self.groups.items():
            present = [float(losses[n].detach()) if torch.is_tensor(losses[n]) else float(losses[n]) for n in names if n in losses]
            if not present:
                continue
            v = sum(present)
            if not math.isfinite(v):
                continue
            e = self.ema.get(g)
            e = v if e is None else self.decay * e + (1.0 - self.decay) * v
            self.ema[g] = e
            self.best[g] = min(self.best.get(g, e), e)
            if g in self.mon:
                self.mon[g].observe(self.n, v)

    def rising(self, g):
        if self.ref == "recent":
            m = self.mon.get(g)
            base = m.baseline() if m is not None else None
            return base is not None and m.ema is not None and m.ema > self.rise * base
        return g in self.ema and self.ema[g] > self.rise * self.best[g]

    def update(self, model, losses):
        old = dict(self.w)
        norms = super().update(model, losses)
        for g in norms:
            if self.rising(g):
                self.w[g] = min(self.wmax, max(self.w[g], old[g]) * self.boost)
        return norms

    def state(self):
        return {"w": dict(self.w), "ema": dict(self.ema), "best": dict(self.best), "n": self.n,
                "mon": {g: {"ema": m.ema, "samples": list(m.samples), "medians": list(m.medians)} for g, m in self.mon.items()}}

    def seed(self, history, start_step):
        """Seed the "recent" monitors from stored history records (resume / replay of a run without guard state)."""
        for g, names in self.groups.items():
            m = self.mon.get(g)
            if m is None or len(m.samples):
                continue
            m.seed([sum(r[n] for n in names if n in r) for r in history
                    if r["step"] > start_step - self.horizon and any(n in r for n in names)])

    def restore_weights(self, st):
        """After a rollback: the weights of the snapshot; the monitors keep their history (they observe, they are not
        part of the trained state), with the EMA put back to the baseline as in the rollback monitor."""
        self.w.update(st.get("w", {}))
        for m in self.mon.values():
            if m.baseline() is not None:
                m.reset()

    def load(self, st):
        self.w.update(st.get("w", {})); self.ema.update(st.get("ema", {})); self.best.update(st.get("best", {}))
        self.n = int(st.get("n", self.n))
        for g, d in st.get("mon", {}).items():
            if g in self.mon:
                m = self.mon[g]
                m.ema = d.get("ema")
                m.samples.clear(); m.samples.extend(d.get("samples", []))
                m.medians.clear(); m.medians.extend(d.get("medians", []))


HEALTH_TERMS = ("ode1", "ode2", "ode3", "charge_s", "charge_e")     # the terms that blew up in every wave-1 excursion


class HealthMonitor:
    """Rollback trigger: EMA of the summed kinetics losses against a baseline built from samples of that EMA (one
    every `every` steps).

    horizon = 0 (wave 2): baseline = median of the last `window` samples (2000 steps).  It follows any ramp slower
        than ~x100 per 1000 steps, so the slow kinetics ramps of wave 2 (x3-10 per 1000 steps) never fired.
    horizon > 0 (wave 3): baseline = the LOWEST 500-step running median over the last `horizon` steps, i.e. the best
        recent level.  A ramp of any speed fires once it is `factor` above that level; a lasting level shift below
        `factor` does not fire and is absorbed once it is older than the horizon (until then no snapshots are taken,
        because "healthy" means within healthy_factor of the baseline)."""

    def __init__(self, decay=0.98, factor=100.0, floor=1e-2, healthy_factor=3.0, horizon=0, every=50, window=40, short=10):
        self.decay, self.factor, self.floor, self.healthy_factor = decay, factor, floor, healthy_factor
        self.horizon, self.every, self.short = int(horizon), every, short
        self.ema, self.samples = None, deque(maxlen=max(window, short))
        self.medians = deque(maxlen=max(int(horizon) // every, 1))

    def _add(self, v):
        self.samples.append(v)
        if self.horizon > 0 and len(self.samples) >= self.short:
            self.medians.append(float(np.median(list(self.samples)[-self.short:])))

    def seed(self, past):
        """past = summed kinetics losses of the stored history records (every history_every steps)."""
        if len(past) >= 25:
            for j in range(0, len(past) - 4, 5):
                self._add(float(np.median(past[j:j + 5])))
            self.ema = float(np.median(past[-5:]))

    def baseline(self):
        if len(self.samples) < 10:
            return None
        if self.horizon > 0:
            return min(self.medians) if self.medians else None
        return float(np.median(self.samples))

    def observe(self, step, h):
        if not math.isfinite(h):
            return
        self.ema = h if self.ema is None else self.decay * self.ema + (1.0 - self.decay) * h
        if step % self.every == 0:
            self._add(self.ema)

    def status(self):
        """(fire, healthy, baseline); fire = the EMA exceeds factor x baseline and the absolute floor."""
        base = self.baseline()
        if base is None or self.ema is None:
            return False, None, base
        return (self.ema > self.factor * base and self.ema > self.floor), self.ema <= self.healthy_factor * base, base

    def reset(self):
        self.ema = self.baseline()


# ====================================================================== model I/O
def build_model(cfg: RateTrainConfig, params: LiSPANParams, window_coef) -> LiSPANRatePINN:
    return LiSPANRatePINN(params, window_coef, cfg.rate_min, cfg.rate_max, cfg.ramp_s, cfg.window_fraction, cfg.width,
                          cfg.depth, cfg.act, cfg.fourier_t, cfg.fourier_period, tuple(cfg.short_t), cfg.ic_tau_s,
                          cfg.quad_order, cfg.salt_sep_natural_scale)


def params_from_dict(d: dict) -> LiSPANParams:
    known = set(LiSPANParams.__dataclass_fields__)
    return LiSPANParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known})


def load_rate_model(checkpoint, device="cpu"):
    """(model, cfg, params) from a checkpoint written by train_rate (final.pt, latest.pt, step_*.pt)."""
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = RateTrainConfig.from_dict(ck["train"])
    params = params_from_dict(ck["params"])
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    model = build_model(cfg, params, ck["window_coef"]).to(device=device, dtype=dtype)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, params


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
    return torch.device(name)


# ====================================================================== training
def train_rate(cfg: RateTrainConfig, params: LiSPANParams, window_coef, out_dir: Path, refs: dict | None = None,
               log=print, resume=None):
    """Forward training of the rate-parametric PINN.  refs = {rate: FV reference} for the evaluation (optional).
    SIGTERM / SIGUSR1 (e.g. the SLURM time limit) save latest.pt and return; resume=<latest.pt> continues a run.
    cfg.weight_guard / cfg.rollback switch on the wave-2 stabilization (GuardedWeights; rollback to the last healthy
    in-memory snapshot with a reduced learning rate when the kinetics residuals blow up)."""
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    model = build_model(cfg, params, window_coef).to(device=device, dtype=dtype)
    model.salt_total_scale = cfg.salt_total_scale
    sampler = RateSampler(cfg, dtype, gen, device)
    net_params = [p_ for n, p_ in model.named_parameters() if not n.startswith("log_mult")]
    opt = torch.optim.Adam(net_params, lr=cfg.lr)
    gamma = (cfg.lr_final / cfg.lr) ** (1 / max(cfg.adam_steps, 1))
    if cfg.weight_guard:
        aw = GuardedWeights(dict(GROUPS), cfg.adaptive_alpha, cfg.weight_guard_rise, cfg.weight_guard_boost, cfg.loss_ema_decay,
                            ref=cfg.weight_guard_ref, horizon=cfg.weight_guard_horizon)
    else:
        aw = AdaptiveWeights(dict(GROUPS), cfg.adaptive_alpha)
    history, evals, start_step = [], [], 1
    guard = {"lr_scale": 1.0, "n_rollbacks": 0, "cooldown_until": 0}
    if resume is not None:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        aw.w.update(ck.get("group_weights", {}))
        if cfg.weight_guard and ck.get("weight_guard"):
            aw.load(ck["weight_guard"])
        guard.update(ck.get("guard", {}))
        start_step = int(ck.get("step", 0)) + 1
        for group in opt.param_groups:      # closed form (no double decay on resume), times the rollback reductions
            group["lr"] = cfg.lr * gamma ** (start_step - 1) * guard["lr_scale"]
        if ck.get("gen_state") is not None:
            gen.set_state(ck["gen_state"].cpu())
        prev = Path(resume).parent / "history.json"
        if prev.exists():
            saved = json.loads(prev.read_text()); history, evals = saved.get("history", []), saved.get("evals", [])
        log(f"Resumed from {resume} at step {start_step}")
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"kind": "lispan_rate", "train": cfg.to_dict(), "params": params.to_dict(),
                                                     "window_coef": list(window_coef)}, indent=2))
    log(f"device {device} | dtype {dtype} | threads {torch.get_num_threads()} | rates [{cfg.rate_min:g}, {cfg.rate_max:g}] C"
        f" | weight guard {cfg.weight_guard} | rollback {cfg.rollback}")
    t0 = time.perf_counter() - (history[-1]["time_s"] if history else 0.0)
    step_holder = [start_step - 1]
    stop = {"signal": None}
    n_skip = [0]

    # ---- health monitor for the rollback: EMA of the unweighted kinetics losses vs its running median
    health = HealthMonitor(cfg.loss_ema_decay, cfg.rollback_factor, cfg.rollback_floor, horizon=cfg.rollback_horizon)
    health.seed([sum(r[k] for k in HEALTH_TERMS if k in r) for r in history if r["step"] > start_step - 2000])
    if cfg.weight_guard and cfg.weight_guard_ref == "recent":
        aw.seed(history, start_step)
    # ---- best evaluation so far (anchor rates), kept in the guard dict so that it survives a resume
    guard.setdefault("best_score", None)
    guard.setdefault("best_step", None)
    anchors = [r for r in cfg.window_rates if refs and any(abs(r - q) < 1e-9 for q in refs)]

    snap = {}

    def take_snapshot(step):
        snap.update(step=step, model=copy.deepcopy(model.state_dict()), opt=copy.deepcopy(opt.state_dict()),
                    aw=copy.deepcopy(aw.state() if cfg.weight_guard else {"w": dict(aw.w)}))

    def on_signal(signum, _frame):
        stop["signal"] = signum
    old_handlers = {}
    for sig in (signal.SIGTERM, getattr(signal, "SIGUSR1", None)):
        if sig is not None:
            try:
                old_handlers[sig] = signal.signal(sig, on_signal)
            except ValueError:                       # not in the main thread (tests)
                pass

    def save(name):
        torch.save({"kind": "lispan_rate", "model": model.state_dict(), "optimizer": opt.state_dict(), "train": cfg.to_dict(),
                    "params": params.to_dict(), "window_coef": list(window_coef), "group_weights": aw.w,
                    "weight_guard": aw.state() if cfg.weight_guard else None, "guard": dict(guard),
                    "step": step_holder[0], "gen_state": gen.get_state()}, out_dir / name)
        (out_dir / "history.json").write_text(json.dumps({"history": history, "evals": evals}))

    if cfg.rollback:
        take_snapshot(start_step - 1)
    try:
        for step in range(start_step, cfg.adam_steps + 1):
            step_holder[0] = step
            model.charge_total_scale = cfg.charge_total_scale if step >= cfg.charge_total_start else 0.0
            terms = rate_residuals(model, sampler.draw())
            losses = {k: (huber(v, cfg.huber_delta) if k in ("charge_s", "charge_e") else v.square()).mean() for k, v in terms.items()}
            # ---- monitors (unweighted losses)
            if cfg.weight_guard:
                aw.observe(losses)
            health.observe(step, sum(float(losses[k].detach()) for k in HEALTH_TERMS if k in losses))
            fire, healthy, base = health.status()
            if cfg.rollback and step >= cfg.rollback_start and base is not None and health.ema is not None:
                recent = [s_ for s_ in guard.setdefault("rollback_steps", [])
                          if cfg.rollback_budget_window <= 0 or s_ > step - cfg.rollback_budget_window]
                budget_left = (len(recent) if cfg.rollback_budget_window > 0 else guard["n_rollbacks"]) < cfg.max_rollbacks
                if fire and budget_left and snap:
                    guard["n_rollbacks"] += 1
                    guard["rollback_steps"].append(step)
                    # halve the learning rate for the cooldown; a further rollback inside the cooldown halves it again
                    in_cooldown = step < guard.get("cooldown_until", 0)
                    guard["lr_scale"] = (guard["lr_scale"] if in_cooldown else 1.0) * cfg.rollback_lr_factor
                    guard["cooldown_until"] = step + cfg.rollback_cooldown
                    log(f"step {step}: ROLLBACK #{guard['n_rollbacks']} to step {snap['step']} (kinetics EMA "
                        f"{health.ema:.2e} > {cfg.rollback_factor:g} x median {base:.2e}); lr scale {guard['lr_scale']:g}")
                    model.load_state_dict(snap["model"])
                    opt.load_state_dict(copy.deepcopy(snap["opt"]))
                    if cfg.adaptive_stop and step >= cfg.adaptive_stop:
                        pass                                 # frozen weights are part of the recipe: keep them
                    elif cfg.weight_guard and cfg.weight_guard_ref == "recent":
                        aw.restore_weights(copy.deepcopy(snap["aw"]))
                    elif cfg.weight_guard:
                        aw.load(copy.deepcopy(snap["aw"]))
                    else:
                        aw.w.update(snap["aw"]["w"])
                    for group in opt.param_groups:
                        group["lr"] = cfg.lr * gamma ** (step - 1) * guard["lr_scale"]
                    health.reset()
                    continue
                if healthy and step % cfg.snapshot_every == 0:
                    take_snapshot(step)
            if guard["lr_scale"] != 1.0 and step >= guard.get("cooldown_until", 0):
                guard["lr_scale"] = 1.0                      # cooldown over: back to the schedule
                for group in opt.param_groups:
                    group["lr"] = cfg.lr * gamma ** (step - 1)
                log(f"step {step}: learning rate back to the schedule after the rollback cooldown")
            if cfg.adaptive_stop and step == cfg.adaptive_stop:
                recs = [r["w"] for r in history if "w" in r and r["step"] > step - cfg.adaptive_freeze_window]
                if recs:
                    aw.w.update({g: float(np.median([r[g] for r in recs if g in r])) for g in aw.w if any(g in r for r in recs)})
                log(f"step {step}: adaptive weights frozen at " + ", ".join(f"{g} {v:.3g}" for g, v in aw.w.items()))
            adaptive_on = cfg.adaptive and not (cfg.adaptive_stop and step >= cfg.adaptive_stop)
            if adaptive_on and (step == start_step or step % cfg.adaptive_every == 0):
                aw.update(model, losses)
                if cfg.charge_weight_floor > 0 and "charge" in aw.w and "ode" in aw.w:
                    aw.w["charge"] = max(aw.w["charge"], cfg.charge_weight_floor * aw.w["ode"])
            loss = sum(aw.w[g] * sum(losses[n] for n in names if n in losses) for g, names in GROUPS.items())
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_skip[0] += 1
                if n_skip[0] <= 10:
                    log(f"step {step}: non-finite loss, batch skipped ({n_skip[0]} so far)")
                continue
            loss.backward()
            if any((p_.grad is not None and not torch.isfinite(p_.grad).all()) for p_ in net_params):
                n_skip[0] += 1
                if n_skip[0] <= 10:
                    log(f"step {step}: non-finite gradients, batch skipped ({n_skip[0]} so far)")
                opt.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(net_params, 10.0)
            opt.step(); sched.step()
            rec = {"step": step, "loss": float(loss.detach()), "time_s": time.perf_counter() - t0,
                   **{k: float(v.detach()) for k, v in losses.items()}}
            logged = step % cfg.log_every == 0 or step == start_step
            if logged or step % max(cfg.history_every, 1) == 0:
                rec["w"] = {g: round(float(v), 4) for g, v in aw.w.items()}
                history.append(rec)
            if logged:
                log(f"step {step:6d} loss {rec['loss']:.3e} | " + " ".join(f"{k} {rec[k]:.2e}" for k in losses) + f" | {rec['time_s']:.0f}s")
            if refs and (step % cfg.eval_every == 0 or step == cfg.adam_steps):
                ev = evaluate_rates(model, refs); ev["step"] = step; ev["time_s"] = rec["time_s"]
                evals.append(ev)
                log("  eval V rmse [mV]: " + " ".join(f"{r:g}C {ev[f'V_rmse_mV@{r:g}']:.3g}" for r in sorted(refs))
                    + f" | mean {ev['V_rmse_mean_mV']:.3g} worst {ev['V_rmse_worst_mV']:.3g} | V max worst {ev['V_max_worst_mV']:.3g}")
                if cfg.keep_best:
                    score = (float(np.mean([ev[f"V_rmse_mV@{r:g}"] for r in anchors])) if anchors
                             else float(ev["V_rmse_mean_mV"]))
                    if math.isfinite(score) and (guard["best_score"] is None or score < guard["best_score"]):
                        guard["best_score"], guard["best_step"] = score, step
                        save("best.pt")
                        (out_dir / "best_metrics.json").write_text(json.dumps(
                            {**ev, "selected_on": [float(r) for r in anchors] or "all eval rates",
                             "selection_score_mV": score}, indent=1))
                        log(f"  best so far (mean V rmse at the anchor rates {score:.3g} mV): best.pt")
            if cfg.latest_every and step % cfg.latest_every == 0:
                save("latest.pt")
            if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                save(f"step_{step}.pt")
            if stop["signal"] is not None:
                save("latest.pt")
                log(f"signal {stop['signal']} received: latest.pt saved at step {step}; resubmit to continue")
                return model, history, evals
            if cfg.stop_step and step >= cfg.stop_step and step < cfg.adam_steps:
                save("latest.pt")
                log(f"stop_step {cfg.stop_step} reached: latest.pt saved")
                return model, history, evals
        save("final.pt")
        log(f"finished: {n_skip[0]} batches skipped (non-finite loss or gradients), {guard['n_rollbacks']} rollbacks"
            + (f", best evaluation at step {guard['best_step']}" if guard.get("best_step") else ""))
    finally:
        for sig, h_ in old_handlers.items():
            signal.signal(sig, h_)
    return model, history, evals
