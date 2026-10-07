"""Forward (and later inverse) PINN for the Li-SPAN cell of Simanjuntak et al. 2024.

Design (docs/LI_SPAN_MODEL_FORMULATION.md section 6; same philosophy as the v2 DFN PINN):

* Inputs (y, t) only; time features as in v2 (2T-1, 2g-1, Fourier modes, short-time exponentials).
* Hard structure wherever the physics allows it:
    - reaction extents xi_1 >= xi_2 >= xi_3 >= 0 (chain S4 -> S3 -> S2 -> S1), all zero at t = 0 and
      xi_1 <= 1 by construction: the four SPAN species, the released sulfide and (through
      eps_Li2S = eps_L0 + V_m c_S4,0 (xi_2 + xi_3)) the porosity and the SPAN area follow exactly; sulfur
      conservation and positivity are exact.  The dissolved S2- inventory is neglected (< 1e-8 of the
      sulfide) and the S2- activity in the kinetics is fixed at saturation (checked against the
      finite-volume reference: 0.03 mV on the voltage, 0.04 mol/m3 on the species).
    - total reduction rate R(y, t) = (I/F) rho / int a rho dy with rho = exp(net) > 0: the electrode
      integral of the faradaic current equals the applied current exactly and the ionic current
      i_e(y, t) = -I int_0^y a rho / int_0^L a rho follows (hard projection, as in v2).
    - kinetics exact: Delta phi(y, t) is the root of sum_m f_m(Delta phi) = R (three generalized
      Butler-Volmer rates sharing one potential difference; monotone, bracketed bisection + Newton,
      gradients by the implicit function theorem).  The individual rates f_m then split R.
    - electrolyte potential with the anode condition built in: phi_e = phi_e,anode(t) + (1 - Y) net.
    - salt concentration with the initial condition built in.
* Residuals (soft): the three extent ODEs d xi_m/dt = a f_m / (2 c_S4,0); the solid/electrolyte Ohm
  consistency d(Delta phi)/dy = (I + i_e)/kappa_s + (i_e + B)/kappa_e in the cathode; the electrolyte
  Ohm law d phi_e/dy = -(i_e + B)/kappa_e in both regions; the salt balance (binary electrolyte,
  transference form) with its collector and anode flux conditions and the interface flux continuity.
  The double layer is not included (its time constant is < 1 s); the reference for the evaluation is
  run with a negligible c_DL.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from ..v2.model import MLP, gauss_legendre
from ..v2.train import AdaptiveWeights
from .params import F, R as RGAS, LiSPANParams, LiSPANProtocol

GROUPS = {"ode": ["ode1", "ode2", "ode3"], "inventory": ["charge_total"], "charge": ["charge_s", "charge_e"], "salt": ["salt_c", "salt_s", "salt_total"],
          "bc": ["bc_c0", "bc_an", "bc_if"]}


@dataclass
class LiSPANTrainConfig:
    seed: int = 0
    dtype: str = "float32"
    threads: int = 0
    width: int = 64
    depth: int = 4
    act: str = "tanh"
    fourier_t: int = 4
    fourier_period: float = 1.0
    short_t: tuple = (60.0, 600.0)        # seconds
    ic_tau_s: float = 60.0
    quad_order: int = 16
    adam_steps: int = 10000
    lr: float = 1e-3
    lr_final: float = 1e-5
    n_cathode: int = 1024
    n_separator: int = 512
    n_boundary: int = 128
    early_fraction: float = 0.15
    early_window: float = 0.05
    huber_delta: float = 3.0
    salt_total_scale: float = 1.0         # mol/m3 of mean salt inventory error per unit residual (0 = off)
    charge_total_scale: float = 0.0       # mol/m3 of extent-inventory error per unit residual (0 = off; see residuals)
    salt_sep_natural_scale: bool = False  # separator salt residual on its own scale (1-t+) I/(F L_sep) instead of the cathode's
    ema_decay: float = 0.0                # exponential moving average of the network weights (0 = off)
    ema_start: float = 0.5                # fraction of adam_steps after which the average starts
    adaptive: bool = True
    adaptive_alpha: float = 0.9
    adaptive_every: int = 50
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_every: int = 2500
    latest_every: int = 250
    inverse_params: list = field(default_factory=list)
    inverse_init: dict = field(default_factory=dict)
    data_path: str = ""
    data_noise_mV: float = 0.0
    data_scale_mV: float = 1.0
    param_lr: float = 5e-3
    param_warmup_steps: int = 0
    net_freeze_steps: int = 0             # inverse: first steps move only the physical parameters (fields frozen)
    param_lr_final: float = 0.0           # final lr of the physical parameters (0 = same decay as the networks)
    gn_every: int = 0                     # multi-rate inverse: Levenberg-Marquardt update of the parameters every n steps
    gn_iters: int = 2                     #   (fields frozen, finite-difference Jacobian of the data residuals) - 0 = off
    gn_max_step: float = 0.3              #   max change per iteration of a log-multiplier (U0: in units of U0_SCALE)
    gn_until: int = 0                     #   last step with LM updates (0 = all)
    gn_mode: str = "frozen"               # "frozen": LM steps accepted on the frozen-field misfit (exact model data);
                                          # "tr": trust region - a step is kept only if the misfit after the fields have
                                          # re-equilibrated (gn_every physics steps) decreased, else fields+parameters are
                                          # restored and the damping raised (model-form error: real data)
    data_to_fields: bool = True           # multi-rate inverse: False = fields trained by the physics only, parameters by
                                          # the data only (fields stay the forward solution at the current parameters)
    param_log_bound: float = 2.0
    fixed_multipliers: dict = field(default_factory=dict)   # forward run at non-nominal parameters (inverse stage 1)

    def to_dict(self):
        d = asdict(self)
        d["short_t"] = list(self.short_t)
        return d


class LiSPANPINN(nn.Module):
    # log-multipliers (k0, b, D_salt, kappa0, Z_CC) and additive offsets in volts (U0_m, scale U0_SCALE)
    PARAMETERS = ("k0_1", "k0_2", "k0_3", "b_1", "b_2", "b_3", "D_salt", "kappa0", "Z_CC", "U0_1", "U0_2", "U0_3")
    U0_SCALE = 0.1

    def __init__(self, params: LiSPANParams, protocol: LiSPANProtocol, t_end: float, width=64, depth=4, act="tanh",
                 fourier_t=4, fourier_period=1.0, short_t=(60.0, 600.0), ic_tau_s=60.0, quad_order=16):
        super().__init__()
        self.p, self.prot, self.t_end = params, protocol, float(t_end)
        p = params
        self.RT = RGAS * p.T
        self.f2 = F / (2.0 * self.RT)
        self.fourier_t, self.fourier_period = fourier_t, float(fourier_period)
        self.short_tau_hat = tuple(float(tau) / self.t_end for tau in short_t)
        self.ic_tau_hat = ic_tau_s / self.t_end
        self.ramp_hat = max(protocol.ramp_s, 1e-6) / self.t_end
        n_t = 2 + 2 * fourier_t + len(self.short_tau_hat)
        self.net_R = MLP(1 + n_t, width, depth, n_out=1, act=act)
        self.net_xi = MLP(1 + n_t, width, depth, n_out=3, act=act)
        self.net_ce = MLP(3 + n_t, width, depth, n_out=1, act=act)
        self.net_phie = MLP(3 + n_t, width, depth, n_out=1, act=act)
        with torch.no_grad():                       # extents: P1 ~ 3 (phase 1 in a third of the time), P2, P3 lagging
            self.net_xi.net[-1].bias.copy_(torch.tensor([math.log(3.0), math.log(0.3), math.log(0.1)]))
        q, w = gauss_legendre(quad_order)
        self.register_buffer("q_nodes", q)
        self.register_buffer("q_weights", w)
        # scales
        self.Yi = p.L_cat / p.L_tot                                        # interface in the global coordinate
        I0 = protocol.current
        kap_e0 = p.kappa0 * p.eps_e_sep ** p.beta_sep
        self.ce_scale = min(0.5, 0.03 * max(I0, 0.3))                      # relative deviation ~ 3 % per A/m2
        self.phie_scale = 2.0 * I0 * p.L_tot / kap_e0                      # V
        self.charge_s_scale = I0 / p.kappa_s_eff                           # V/m
        self.charge_e_scale = I0 / kap_e0                                  # V/m
        self.salt_scale = (1.0 - p.t_plus) * I0 / (F * p.L_cat)            # mol/m3/s
        self.D_ion = {"Li": p.D_salt / (2 * (1 - p.t_plus)), "PF6": p.D_salt / (2 * p.t_plus)}
        self.salt_total_scale = 1.0
        self.charge_total_scale = 0.0
        self.salt_scale_sep = self.salt_scale                              # set by train(): natural separator scale optional
        self.log_mult = nn.ParameterDict({n: nn.Parameter(torch.zeros(()), requires_grad=False) for n in self.PARAMETERS})

    # ------------------------------------------------------------------ parameters
    def set_parameter_values(self, values):
        """Set multipliers (offsets in V for U0_m) without changing trainability."""
        with torch.no_grad():
            for n, v in values.items():
                self.log_mult[n].fill_(float(v) / self.U0_SCALE if n.startswith("U0") else math.log(float(v)))

    def set_trainable_parameters(self, names, initial=None):
        initial = initial or {}
        for n in names:
            self.log_mult[n].requires_grad_(True)
        self.set_parameter_values({n: v for n, v in initial.items() if n in names})

    def mult(self, name):
        return torch.exp(self.log_mult[name])

    def k0(self, m):
        return self.p.kin_scale * self.p.k0[m] * self.mult(f"k0_{m + 1}")

    def U0(self, m):
        return self.p.U0[m] + self.U0_SCALE * self.log_mult[f"U0_{m + 1}"]

    def b(self, m):
        return self.p.b[m] * self.mult(f"b_{m + 1}")

    def parameter_values(self):
        out = {}
        for n in self.PARAMETERS:
            v = float(self.log_mult[n].detach())
            out[n] = self.U0_SCALE * v if n.startswith("U0") else math.exp(v)
        return out

    # ------------------------------------------------------------------ features
    def g(self, T):
        return torch.tanh(T / self.ramp_hat)

    def I(self, T):
        return self.prot.current * self.g(T)

    def charge_passed(self, T):
        """Q(t) = int_0^t I dt = I0 tau ln cosh(t/tau) [C/m2], written overflow-free."""
        tau = self.ramp_hat
        return self.prot.current * self.t_end * (T + tau * (torch.log1p(torch.exp(-2.0 * T / tau)) - math.log(2.0)))

    def ic_factor(self, T):
        return 1 - torch.exp(-T / self.ic_tau_hat)

    def tf(self, T):
        feats = [2 * T - 1, 2 * self.g(T) - 1]
        for k in range(1, self.fourier_t + 1):
            w = 2 * math.pi * k / self.fourier_period
            feats += [torch.sin(w * T), torch.cos(w * T)]
        for tau in self.short_tau_hat:
            feats.append(2 * torch.exp(-T / tau) - 1)
        return torch.cat(feats, dim=1)

    def efeat(self, Y):
        """Electrolyte space features on the global coordinate Y = y/L_tot: 2Y-1, kink at the interface,
        cos(pi Y) (zero slope at the collector)."""
        return torch.cat([2 * Y - 1, 2 * torch.abs(Y - self.Yi) / max(self.Yi, 1 - self.Yi) - 1, torch.cos(math.pi * Y)], dim=1)

    # ------------------------------------------------------------------ cathode fields
    def extents(self, Yc, T):
        """xi_1 >= xi_2 >= xi_3 >= 0, zero at T = 0, xi_1 < 1."""
        e1, e2, e3 = self._remaining(Yc, T)
        xi1 = 1 - e1
        xi2 = xi1 * (1 - e2)
        xi3 = xi2 * (1 - e3)
        return xi1, xi2, xi3

    def _remaining(self, Yc, T):
        """Remaining fractions e_m = exp(-T P_m) of the educt of each reaction (P_m = exp(net) > 0); exponents
        capped at 60 (e^-60 = 9e-27 is representable in float32 and leaves no phantom Tafel rate)."""
        P = torch.exp(torch.clamp(self.net_xi(torch.cat([2 * Yc - 1, self.tf(T)], dim=1)), -16.0, 6.0))
        return tuple(torch.exp(-torch.clamp(T * P[:, m:m + 1], max=60.0)) for m in range(3))

    def species(self, Yc, T):
        p = self.p
        c40 = p.c_init[0]
        e1, e2, e3 = self._remaining(Yc, T)
        xi1 = 1 - e1
        xi2 = xi1 * (1 - e2)
        xi3 = xi2 * (1 - e3)
        # species from the remaining fractions directly (no 1 - xi cancellation in float32)
        c_S4 = c40 * e1
        c_S3 = c40 * xi1 * e2
        c_S2 = c40 * xi2 * e3
        c_S1 = c40 * (xi1 + xi3)
        eps_L = p.eps_L0 + p.V_m_L * c40 * (xi2 + xi3)
        eps_e = 1 - p.eps_SPAN - p.eps_CB - eps_L
        a = p.a_SPAN0 * (eps_e / p.eps_e0) ** p.xi
        return {"xi": (xi1, xi2, xi3), "c_S4": c_S4, "c_S3": c_S3, "c_S2": c_S2, "c_S1": c_S1, "eps_L": eps_L, "eps_e": eps_e, "a": a}

    def rho(self, Yc, T):
        return torch.exp(torch.clamp(self.net_R(torch.cat([2 * Yc - 1, self.tf(T)], dim=1)), -12.0, 12.0))

    def _a_rho_nodes(self, Yc_upper, T):
        """int_0^{Yc_upper} a rho dY by Gauss-Legendre on [0, Yc_upper] (Yc_upper broadcast per sample)."""
        n, q = T.shape[0], self.q_nodes.shape[0]
        nodes = self.q_nodes.to(T.dtype).view(1, q)
        Yq = (Yc_upper.view(n, 1) * nodes).reshape(-1, 1)
        Tq = T.view(n, 1).expand(n, q).reshape(-1, 1)
        val = (self.species(Yq, Tq)["a"] * self.rho(Yq, Tq)).view(n, q)
        return Yc_upper.view(n, 1) * (val * self.q_weights.to(T.dtype).view(1, q)).sum(1, keepdim=True)

    def current_fields(self, Yc, T, sp=None):
        """Total reduction rate R [mol/m2/s per SPAN area] (hard projection) and ionic current i_e [A/m2]."""
        sp = sp or self.species(Yc, T)
        denom = self._a_rho_nodes(torch.ones_like(Yc), T)                     # int_0^1 a rho dY
        cum = self._a_rho_nodes(Yc, T)                                        # int_0^Y a rho dY
        I = self.I(T)
        R = I / (F * self.p.L_cat) * self.rho(Yc, T) / denom
        i_e = -I * cum / denom
        return R, i_e

    def rates(self, dphi, sp, a_Li):
        """Generalized BV rates f_m(Delta phi) [mol/m2/s]; S2- activity fixed at saturation."""
        p = self.p
        # activity floor 1e-30: keeps sqrt differentiable at zero species but is small enough that an exhausted
        # educt carries no phantom rate even with the e^{-x} ~ 1e5 Tafel amplification at low Delta phi (a floor
        # of 1e-12 mol/m3 gave reaction 1 1.4 % of the current after phase 1 and a 24 mV plateau, run A)
        fl = 1e-30
        aS4, aS3, aS2, aS1 = (sp["c_S4"] / p.c_ref[0] + fl, sp["c_S3"] / p.c_ref[1] + fl,
                              sp["c_S2"] / p.c_ref[2] + fl, sp["c_S1"] / p.c_ref[3] + fl)
        aS = p.c_S_sat / p.c_S_ref
        zeta = [torch.clamp(1 - aS4, 0, 1), torch.clamp(1 - aS3, 0, 1), torch.clamp(1 - aS2, 0, 1)]
        out = []
        eds = [torch.sqrt(aS4) * a_Li, torch.sqrt(aS3), torch.sqrt(aS2)]
        prods = [torch.sqrt(aS3 * aS1), torch.sqrt(aS2 * aS), torch.sqrt(aS1 * aS)]
        for m in range(3):
            x = torch.clamp(self.f2 * (dphi - (self.U0(m) - self.b(m) * zeta[m])), -60.0, 60.0)
            rev = 1.0 if p.reversible[m] else 0.0
            out.append(self.k0(m) * (eds[m] * torch.exp(-x) - rev * prods[m] * torch.exp(x)))
        return out

    def dphi(self, R, sp, a_Li, lo=-0.5, hi=3.6, n_bisect=42, n_newton=3):
        """Delta phi solving sum_m f_m(Delta phi) = R (monotone decreasing); implicit-function gradients.

        The slope dg/d(Delta phi) is floored at -F R/(2RT) (its value when the forward rates carry R): where the
        educts are exhausted during training the root sits at the bracket edge and the sensitivity would
        otherwise blow up; the floor keeps d(Delta phi)/d(.) at the Tafel scale (2RT/F) d ln(.)."""
        f2R = self.f2 * R.detach()

        def gfun(d):
            return sum(self.rates(d, sp, a_Li)) - R

        def slope_floor(dg):
            return torch.minimum(dg, -f2R)
        with torch.no_grad():
            a = torch.full_like(R, lo); b = torch.full_like(R, hi)
            for _ in range(n_bisect):
                mid = 0.5 * (a + b)
                pos = gfun(mid) > 0
                a = torch.where(pos, mid, a); b = torch.where(pos, b, mid)
            d = 0.5 * (a + b)
            for _ in range(n_newton):
                dd = d.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    gv = gfun(dd)
                    dg = torch.autograd.grad(gv, dd, torch.ones_like(gv))[0]
                step = torch.clamp(gv / slope_floor(dg), -0.05, 0.05)
                d = d - torch.nan_to_num(step)
        # attach gradients: one Newton step with the graph (value unchanged where converged, derivative bounded)
        d0 = d.detach()
        with torch.enable_grad():
            d0g = d0.clone().requires_grad_(True)
            gv = gfun(d0g)
            dg = slope_floor(torch.autograd.grad(gv, d0g, torch.ones_like(gv))[0].detach())
        if torch.is_grad_enabled():
            corr = gfun(d0) / dg
            # value: Newton correction clamped to 0.1 V (non-converged points); gradient: unclamped (straight-through)
            return d0 - (corr.clamp(-0.1, 0.1).detach() + corr - corr.detach())
        return d0

    # ------------------------------------------------------------------ electrolyte fields
    def ce(self, Y, T):
        """Salt concentration; the deviation is bounded (tanh) so that kappa_e never collapses."""
        dev = torch.tanh(self.net_ce(torch.cat([self.efeat(Y), self.tf(T)], dim=1)))
        return self.p.c_Li0 * (1 + self.ce_scale * self.ic_factor(T) * dev)

    def phie_anode(self, T):
        """phi_e at the Li surface from the anode kinetics (phi_s(Li) = 0)."""
        Y1 = torch.ones_like(T)
        a_Li = torch.clamp(self.ce(Y1, T) / self.p.c_Li0, 1e-6)
        I = self.I(T)
        eta = (2 * self.RT / F) * torch.asinh(I / (2 * F * self.p.k0_Li * torch.sqrt(a_Li)))
        return -(self.RT / F) * torch.log(a_Li) - eta

    def phie(self, Y, T):
        return self.phie_anode(T) + (1 - Y) * self.phie_scale * self.net_phie(torch.cat([self.efeat(Y), self.tf(T)], dim=1))

    def kappa_e(self, c, eps_e, beta):
        return self.p.kappa0 * self.mult("kappa0") * torch.clamp(c, 1.0) / self.p.c_Li0 * eps_e ** beta

    def D_eff(self, eps_e, beta):
        return self.p.D_salt * self.mult("D_salt") * eps_e ** beta

    # ------------------------------------------------------------------ voltage
    def voltage(self, T):
        Y0 = torch.zeros_like(T)
        sp = self.species(Y0, T)
        a_Li = torch.clamp(self.ce(Y0, T) / self.p.c_Li0, 1e-6)
        R, _ = self.current_fields(Y0, T, sp)
        d = self.dphi(R, sp, a_Li)
        return d + self.phie(Y0, T) - self.p.Z_CC * self.mult("Z_CC") * self.I(T)

    def terminal_voltage(self, T):
        return self.voltage(T)


# ====================================================================== residuals
def _grad(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]


def huber(v, delta):
    a = v.abs()
    return torch.where(a <= delta, v * v, delta * (2 * a - delta))


def residuals(model: LiSPANPINN, batch):
    """Residual tensors (dimensionless, O(1) when the problem is unsolved)."""
    p, t_end = model.p, model.t_end
    out = {}
    # ---------------- cathode: extents, charge, salt
    Yc, T = batch["c"]
    Yc = Yc.clone().requires_grad_(True); T = T.clone().requires_grad_(True)
    y = Yc * p.L_cat
    Yg = y / p.L_tot
    sp = model.species(Yc, T)
    c = model.ce(Yg, T)
    a_Li = torch.clamp(c / p.c_Li0, 1e-6)
    R, i_e = model.current_fields(Yc, T, sp)
    d = model.dphi(R, sp, a_Li)
    f = model.rates(d, sp, a_Li)
    c40 = p.c_init[0]
    for m, xi in enumerate(sp["xi"]):
        dxi = _grad(xi, T)                                   # d xi / dT
        out[f"ode{m + 1}"] = dxi - t_end * sp["a"] * f[m] / (2 * c40)
    # charge: solid and electrolyte Ohm (B = F (D+ - D-) dc/dy)
    dc_dy = _grad(c, Yc) / p.L_cat
    B = F * (model.D_ion["Li"] - model.D_ion["PF6"]) * sp["eps_e"] ** p.beta_cat * dc_dy * model.mult("D_salt")
    kap = model.kappa_e(c, sp["eps_e"], p.beta_cat)
    phie = model.phie(Yg, T)
    dphie_dy = _grad(phie, Yc) / p.L_cat
    ddphi_dy = _grad(d, Yc) / p.L_cat
    I = model.I(T)
    out["charge_s"] = (ddphi_dy - ((I + i_e) / p.kappa_s_eff + (i_e + B) / kap)) / model.charge_s_scale
    out["charge_e"] = (dphie_dy + (i_e + B) / kap) / model.charge_e_scale
    # salt balance in the cathode: d(eps c)/dt = d/dy(D dc/dy) - (1 - t+) a R
    D = model.D_eff(sp["eps_e"], p.beta_cat)
    flux_diff = D * dc_dy
    dflux_dy = _grad(flux_diff, Yc) / p.L_cat
    dc_dt = _grad(c, T) / t_end
    deps_dt = _grad(sp["eps_e"], T) / t_end
    out["salt_c"] = (sp["eps_e"] * dc_dt + c * deps_dt - dflux_dy + (1 - p.t_plus) * sp["a"] * R) / model.salt_scale
    # ---------------- separator: charge_e and salt (i_e = -I)
    Ys, Ts = batch["s"]
    Ys = Ys.clone().requires_grad_(True); Ts = Ts.clone().requires_grad_(True)
    Yg_s = (p.L_cat + Ys * p.L_sep) / p.L_tot
    cs = model.ce(Yg_s, Ts)
    dcs_dy = _grad(cs, Ys) / p.L_sep
    eps_s = torch.full_like(cs, p.eps_e_sep)
    Bs = F * (model.D_ion["Li"] - model.D_ion["PF6"]) * p.eps_e_sep ** p.beta_sep * dcs_dy * model.mult("D_salt")
    kap_s = model.kappa_e(cs, eps_s, p.beta_sep)
    phie_s = model.phie(Yg_s, Ts)
    Is = model.I(Ts)
    out["charge_e"] = torch.cat([out["charge_e"], (_grad(phie_s, Ys) / p.L_sep + (-Is + Bs) / kap_s) / model.charge_e_scale])
    Ds = model.D_eff(eps_s, p.beta_sep)
    dflux_s = _grad(Ds * dcs_dy, Ys) / p.L_sep
    out["salt_s"] = (p.eps_e_sep * _grad(cs, Ts) / t_end - dflux_s) / model.salt_scale_sep
    # ---------------- boundary / interface conditions on the salt
    Tb = batch["b"].clone().requires_grad_(True)
    Y0 = torch.zeros_like(Tb).requires_grad_(True)
    c0 = model.ce(Y0, Tb)
    out["bc_c0"] = _grad(c0, Y0) / p.L_tot / (model.salt_scale * p.L_cat / p.D_salt)        # zero slope at the collector
    Y1 = torch.ones_like(Tb).requires_grad_(True)
    c1 = model.ce(Y1, Tb)
    D1 = model.D_eff(torch.full_like(c1, p.eps_e_sep), p.beta_sep)
    Ib = model.I(Tb)
    out["bc_an"] = (D1 * _grad(c1, Y1) / p.L_tot - (1 - p.t_plus) * Ib / F) / (model.salt_scale * p.L_cat)   # anion flux zero
    # interface: continuity of the diffusive salt flux (c itself is continuous by construction)
    eps_if = 1e-4
    Ym = torch.full_like(Tb, model.Yi - eps_if).requires_grad_(True)
    Yp = torch.full_like(Tb, model.Yi + eps_if).requires_grad_(True)
    cm, cp = model.ce(Ym, Tb), model.ce(Yp, Tb)
    sp_if = model.species(torch.ones_like(Tb), Tb)
    Dm = model.D_eff(sp_if["eps_e"], p.beta_cat); Dp = model.D_eff(torch.full_like(cp, p.eps_e_sep), p.beta_sep)
    out["bc_if"] = (Dm * _grad(cm, Ym) - Dp * _grad(cp, Yp)) / p.L_tot / (model.salt_scale * p.L_cat)
    # global salt inventory (the anion has no source and no boundary flux): int eps c dy = eps_e0 c0 L_cat + eps_sep c0 L_sep
    if model.salt_total_scale > 0:
        q = model.q_nodes.shape[0]; nb = Tb.shape[0]
        nodes = model.q_nodes.to(Tb.dtype).view(1, q); w = model.q_weights.to(Tb.dtype).view(1, q)
        Tq = Tb.view(nb, 1).expand(nb, q).reshape(-1, 1)
        Yc_q = nodes.expand(nb, q).reshape(-1, 1)
        spq = model.species(Yc_q, Tq)
        inv_c = ((spq["eps_e"] * model.ce(Yc_q * p.L_cat / p.L_tot, Tq)).view(nb, q) * w).sum(1, keepdim=True) * p.L_cat
        Ys_q = (p.L_cat + nodes * p.L_sep) / p.L_tot
        inv_s = (model.ce(Ys_q.expand(nb, q).reshape(-1, 1), Tq).view(nb, q) * w).sum(1, keepdim=True) * p.eps_e_sep * p.L_sep
        inv0 = p.eps_e0 * p.c_Li0 * p.L_cat + p.eps_e_sep * p.c_Li0 * p.L_sep
        # the Li2S growth removes pore volume: the salt displaced is small (eps_L <= 0.03) and is in the balance already
        out["salt_total"] = (inv_c + inv_s - inv0) / (model.salt_total_scale * p.L_tot)
    # global charge inventory: every reaction transfers 2 e- per chain, so the charge passed fixes the extent sum,
    # 2 F c_S4,0 L_cat int_0^1 (xi_1 + xi_2 + xi_3) dY = Q(t) = int_0^t I dt.  The hard current projection makes
    # this exact only if the extent ODEs are; their small mean residual otherwise integrates into an inventory
    # drift that ends up in c_S2 = c_S4,0 (xi_2 - xi_3) during phase 3 (1 mV per mol/m3 through b_3, run B).
    if model.charge_total_scale > 0:
        q = model.q_nodes.shape[0]; nb = Tb.shape[0]
        nodes = model.q_nodes.to(Tb.dtype).view(1, q); w = model.q_weights.to(Tb.dtype).view(1, q)
        Tq = Tb.view(nb, 1).expand(nb, q).reshape(-1, 1)
        Yc_q = nodes.expand(nb, q).reshape(-1, 1)
        xi1, xi2, xi3 = model.extents(Yc_q, Tq)
        inv = p.c_init[0] * ((xi1 + xi2 + xi3).view(nb, q) * w).sum(1, keepdim=True)
        out["charge_total"] = (inv - model.charge_passed(Tb) / (2.0 * F * p.L_cat)) / model.charge_total_scale
    return out


# ====================================================================== sampling / training
class Sampler:
    def __init__(self, cfg: LiSPANTrainConfig, dtype, gen):
        self.cfg, self.dtype, self.gen = cfg, dtype, gen

    def u(self, n):
        return torch.rand(n, 1, generator=self.gen, dtype=self.dtype)

    def times(self, n):
        t = self.u(n)
        m = int(round(self.cfg.early_fraction * n))
        t[:m] *= self.cfg.early_window
        return t

    def draw(self):
        c = self.cfg
        return {"c": (self.u(c.n_cathode), self.times(c.n_cathode)),
                "s": (self.u(c.n_separator), self.times(c.n_separator)),
                "b": self.times(c.n_boundary)}


def evaluate(model: LiSPANPINN, ref: dict, dtype):
    """Compare with a finite-volume reference (dict from dfn_pinn.lispan.model.solve/assemble)."""
    p = model.p
    t = np.asarray(ref["t"]); keep = t <= model.t_end
    t = t[keep]
    T = torch.as_tensor(t / model.t_end, dtype=dtype).view(-1, 1)
    with torch.no_grad():
        V = model.voltage(T).view(-1).numpy()
    e = V - np.asarray(ref["V"])[keep]
    m = t > 100.0
    out = {"V_rmse_mV": 1e3 * float(np.sqrt(np.mean(e[m] ** 2))), "V_max_mV": 1e3 * float(np.abs(e[m]).max())}
    # species and eps_L on the cathode grid
    yc = np.asarray(ref["y_c"]); Yc = torch.as_tensor(yc / p.L_cat, dtype=dtype).view(-1, 1)
    nt, ny = len(t), len(yc)
    Yq = Yc.view(1, ny, 1).expand(nt, ny, 1).reshape(-1, 1); Tq = T.view(nt, 1, 1).expand(nt, ny, 1).reshape(-1, 1)
    with torch.no_grad():
        sp = model.species(Yq, Tq)
        for key in ("c_S4", "c_S3", "c_S2", "c_S1"):
            pred = sp[key].view(nt, ny).numpy().T
            out[key + "_rmse"] = float(np.sqrt(np.mean((pred - np.asarray(ref[key])[:, keep]) ** 2)))
        out["eps_L_rmse"] = float(np.sqrt(np.mean((sp["eps_L"].view(nt, ny).numpy().T - np.asarray(ref["eps_L"])[:, keep]) ** 2)))
        y = np.asarray(ref["y"]); Yg = torch.as_tensor(y / p.L_tot, dtype=dtype).view(1, -1, 1).expand(nt, len(y), 1).reshape(-1, 1)
        Tg = T.view(nt, 1, 1).expand(nt, len(y), 1).reshape(-1, 1)
        ce = model.ce(Yg, Tg).view(nt, len(y)).numpy().T
        out["c_e_rmse"] = float(np.sqrt(np.mean((ce - np.asarray(ref["c_Li"])[:, keep]) ** 2)))
        out["c_e_max"] = float(np.abs(ce - np.asarray(ref["c_Li"])[:, keep]).max())
        phie = model.phie(Yg, Tg).view(nt, len(y)).numpy().T
        out["phi_e_rmse_mV"] = 1e3 * float(np.sqrt(np.mean((phie - np.asarray(ref["phi_e"])[:, keep]) ** 2)))
        # Delta phi at the collector
        Y0 = torch.zeros_like(T)
        spc = model.species(Y0, T); a_Li = torch.clamp(model.ce(Y0, T) / p.c_Li0, 1e-6)
        Rr, _ = model.current_fields(Y0, T, spc)
        d = model.dphi(Rr, spc, a_Li).view(-1).numpy()
        out["dphi0_rmse_mV"] = 1e3 * float(np.sqrt(np.mean((d[m] - np.asarray(ref["dphi"])[0, keep][m]) ** 2)))
    return out


def train(cfg: LiSPANTrainConfig, params: LiSPANParams, protocol: LiSPANProtocol, t_end: float, out_dir: Path,
          reference: dict | None = None, log=print, init_state=None, resume=None):
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    model = LiSPANPINN(params, protocol, t_end, cfg.width, cfg.depth, cfg.act, cfg.fourier_t, cfg.fourier_period,
                       tuple(cfg.short_t), cfg.ic_tau_s, cfg.quad_order).to(dtype)
    model.salt_total_scale = cfg.salt_total_scale
    model.charge_total_scale = cfg.charge_total_scale
    if cfg.salt_sep_natural_scale:
        # the separator has no source: its transient balances D d2c/dy2 ~ (1-t+) I/(F L_sep), 6x smaller than the cathode
        # source scale, so on the cathode's scale its residual looked converged while the 1 C salt transient was 25 % off
        model.salt_scale_sep = model.salt_scale * params.L_cat / params.L_sep
    if init_state is not None:
        model.load_state_dict(init_state, strict=False)
    if cfg.fixed_multipliers:
        model.set_parameter_values(cfg.fixed_multipliers)
        log(f"Fixed parameter multipliers: {cfg.fixed_multipliers}")
    data = None
    if cfg.data_path:
        raw = np.load(cfg.data_path, allow_pickle=False)
        t_d, V_d = np.asarray(raw["t"], float), np.asarray(raw["V"], float)
        if cfg.data_noise_mV > 0:
            V_d = V_d + np.random.default_rng(cfg.seed + 7).normal(0.0, cfg.data_noise_mV * 1e-3, V_d.shape)
        keep = t_d <= t_end
        data = (torch.as_tensor(t_d[keep] / t_end, dtype=dtype).view(-1, 1), torch.as_tensor(V_d[keep], dtype=dtype).view(-1, 1))
        model.set_trainable_parameters(cfg.inverse_params, cfg.inverse_init)
        log(f"Inverse problem: learning {cfg.inverse_params}, initial {model.parameter_values()}")
    sampler = Sampler(cfg, dtype, gen)
    net_params = [p_ for n, p_ in model.named_parameters() if not n.startswith("log_mult")]
    phys_params = [model.log_mult[n] for n in cfg.inverse_params]
    groups = [{"params": net_params, "lr": cfg.lr}]
    if phys_params:
        groups.append({"params": phys_params, "lr": cfg.param_lr})
    opt = torch.optim.Adam(groups)
    gamma = (cfg.lr_final / cfg.lr) ** (1 / max(cfg.adam_steps, 1))
    group_names = dict(GROUPS)
    if data is not None:
        group_names["data"] = ["data_V"]
    aw = AdaptiveWeights(group_names, cfg.adaptive_alpha)
    history, evals, start_step = [], [], 1
    if resume is not None:
        ck = torch.load(resume, weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["optimizer"])
        aw.w.update(ck.get("group_weights", {}))
        start_step = int(ck.get("step", 0)) + 1
        for group, base in zip(opt.param_groups, [cfg.lr, cfg.param_lr]):
            group["lr"] = base * gamma ** (start_step - 1)
        prev = Path(resume).parent / "history.json"
        if prev.exists():
            saved = json.loads(prev.read_text()); history, evals = saved.get("history", []), saved.get("evals", [])
        log(f"Resumed from {resume} at step {start_step}")
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    # exponential moving average of the network weights (evaluated and saved next to the raw weights)
    ema_names = [n for n, p_ in model.named_parameters() if not n.startswith("log_mult")]
    ema = None
    if resume is not None and cfg.ema_decay > 0:
        ck_ema = torch.load(resume, weights_only=False).get("ema")
        if ck_ema is not None:
            ema = {n: v.clone() for n, v in ck_ema.items()}
    ema_first = int(cfg.ema_start * cfg.adam_steps)

    def ema_update():
        nonlocal ema
        with torch.no_grad():
            cur = dict(model.named_parameters())
            if ema is None:
                ema = {n: cur[n].detach().clone() for n in ema_names}
            else:
                for n in ema_names:
                    ema[n].lerp_(cur[n].detach(), 1.0 - cfg.ema_decay)

    def with_ema(fn):
        """Run fn() with the averaged weights loaded, then restore the raw weights."""
        cur = dict(model.named_parameters())
        raw = {n: cur[n].detach().clone() for n in ema_names}
        with torch.no_grad():
            for n in ema_names:
                cur[n].copy_(ema[n])
        try:
            return fn()
        finally:
            with torch.no_grad():
                for n in ema_names:
                    cur[n].copy_(raw[n])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"train": cfg.to_dict(), "params": params.to_dict(),
                                                     "protocol": protocol.to_dict(), "t_end": t_end}, indent=2))
    t0 = time.perf_counter() - (history[-1]["time_s"] if history else 0.0)
    step_holder = [start_step - 1]
    n_nan = [0]

    def save(name):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "train": cfg.to_dict(),
                    "params": params.to_dict(), "protocol": protocol.to_dict(), "t_end": t_end,
                    "group_weights": aw.w, "parameters": model.parameter_values(), "step": step_holder[0],
                    "ema": ema}, out_dir / name)
        (out_dir / "history.json").write_text(json.dumps({"history": history, "evals": evals}))

    for step in range(start_step, cfg.adam_steps + 1):
        step_holder[0] = step
        batch = sampler.draw()
        terms = residuals(model, batch)
        # potential residuals get a Huber loss (quadratic below huber_delta, linear above): the Tafel relation turns
        # any spatial roughness of the species into very large d(Delta phi)/dy early in training
        losses = {k: (huber(v, cfg.huber_delta) if k in ("charge_s", "charge_e") else v.square()).mean() for k, v in terms.items()}
        if data is not None:
            Vp = model.voltage(data[0])
            losses["data_V"] = ((Vp - data[1]) / (1e-3 * cfg.data_scale_mV)).square().mean()
        if cfg.adaptive and (step == start_step or step % cfg.adaptive_every == 0):
            aw.update(model, losses)
        loss = sum(aw.w[g] * sum(losses[n] for n in names if n in losses) for g, names in group_names.items())
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            n_nan[0] += 1
            if n_nan[0] <= 5:
                log(f"step {step}: non-finite loss, batch skipped")
            continue
        loss.backward()
        if any((p_.grad is not None and not torch.isfinite(p_.grad).all()) for p_ in net_params):
            n_nan[0] += 1
            if n_nan[0] <= 5:
                log(f"step {step}: non-finite gradients, batch skipped")
            opt.zero_grad(set_to_none=True)
            continue
        if data is not None and step <= cfg.param_warmup_steps:
            for p_ in phys_params:
                p_.grad = None
        if data is not None and step <= cfg.net_freeze_steps:
            for p_ in net_params:
                p_.grad = None
        torch.nn.utils.clip_grad_norm_(net_params, 10.0)
        opt.step(); sched.step()
        if cfg.ema_decay > 0 and step >= ema_first:
            ema_update()
        with torch.no_grad():
            for n in cfg.inverse_params:
                model.log_mult[n].clamp_(-cfg.param_log_bound, cfg.param_log_bound)
        rec = {"step": step, "loss": float(loss), "time_s": time.perf_counter() - t0, **{k: float(v) for k, v in losses.items()}}
        if data is not None:
            rec["params"] = model.parameter_values()
        history.append(rec)
        if step % cfg.log_every == 0 or step == start_step:
            msg = f"step {step:6d} loss {rec['loss']:.3e} | " + " ".join(f"{k} {rec[k]:.2e}" for k in losses) + f" | {rec['time_s']:.0f}s"
            if data is not None:
                msg += " | " + ", ".join(f"{k} {v:.4g}" for k, v in rec["params"].items() if k in cfg.inverse_params)
            log(msg)
        if reference is not None and (step % cfg.eval_every == 0 or step == cfg.adam_steps):
            ev = evaluate(model, reference, dtype); ev["step"] = step
            if ema is not None:
                ev_ema = with_ema(lambda: evaluate(model, reference, dtype))
                ev.update({f"ema_{k}": v for k, v in ev_ema.items()})
            evals.append(ev)
            log("  eval: " + ", ".join(f"{k} {v:.4g}" for k, v in ev.items() if k != "step" and not k.startswith("ema_")))
            if ema is not None:
                log("  eval (EMA): " + ", ".join(f"{k[4:]} {v:.4g}" for k, v in ev.items() if k.startswith("ema_")))
        if cfg.latest_every and step % cfg.latest_every == 0:
            save("latest.pt")
        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            save(f"step_{step}.pt")
    save("final.pt")
    if ema is not None:
        with_ema(lambda: save("final_ema.pt"))
    return model, history, evals
