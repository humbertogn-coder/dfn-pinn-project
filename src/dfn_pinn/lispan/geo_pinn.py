"""Network v5: the chemistry-conditioned Li-SPAN PINN with the cathode geometry as two more inputs (docs/GEOMETRY.md).

Condition z = (r, theta_1..theta_k, ln L_cat [um], f): C-rate per gram of sulfur, the chemistry inputs of
chem_pinn (v2 box), the cathode thickness and the SPAN scale factor f of geometry.py (eps_SPAN, eps_CB, c_S4,0, c_ref
and a_SPAN0 all x f; porosity 1 - 0.1265 f).  The separator is fixed (geometry.SEP_POLYOLEFIN, a 25 um Celgard-type
membrane): the network's LiSPANParams p carry that separator and the reference cathode (100 um, f = 1); every
geometry-dependent quantity is computed per row from z.

Scaling used (geometry.py docstring): the species are kept in reference units (c_S4,0 of f = 1) - the activities
c / c_ref, the extent equations d xi / dt = a f_m / (2 c_S4,0) and the charge per chain do not depend on f; the true
specific surface is a = f a_SPAN0 (eps_e / eps_e0)^xi, the Li2S volume grows with f, the current density at C-rate r
is I = ONE_C r f L_cat / 100 um, the solid conductivity kappa_SPAN (f eps_SPAN)^1.5.

Electrolyte coordinate: the electrolyte networks see a LOGICAL coordinate Y in [0, 1] with the interface fixed at
Y_i = 0.8 (cathode [0, Y_i], separator [Y_i, 1]) whatever the thicknesses; the residuals carry the chain-rule factors
dY/dy = Y_i / L_cat (cathode) and (1 - Y_i) / L_sep (separator).  The salt deviation is exponential,
c = c0 exp(s(z) dev) with |dev| < 1, because dense cathodes concentrate the salt up to ~2x as Li2S fills the pores
(FV, f = 5) and thick ones polarize it; s(z) is a physics estimate of the largest |ln c / c0|.

Window: ln(r t_cut) = P6(ln r, theta_hat) [the v3 window, kept] + C_d(ln r, theta_hat, gamma_hat) fitted on the v5 FV
design (scripts/lispan_geo_design.py).
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from ..v2.model import MLP, gauss_legendre
from .chem_pinn import (CHEM_INPUTS_V2, ChemSampler, ChemTrainConfig, LiSPANChemPINN, check_spec, chem_vector,
                        cliff_metrics, normalize, window_degree, window_features, window_features_torch)
from .geometry import GEO_INPUTS_V5, L_REF_UM, SEP_POLYOLEFIN, in_domain, polyolefin_cell
from .params import F, R as RGAS, LiSPANParams
from .rate_pinn import ONE_C, _grad, params_from_dict, resolve_device

Y_INTERFACE = 0.8


def network_params(p: LiSPANParams) -> LiSPANParams:
    """Reference cell of the v5 network: 100 um, f = 1 cathode and the polyolefin separator."""
    return polyolefin_cell(p, L_REF_UM, 1.0)


def geo_window_t_end(window, spec, geo_spec, rate, chem_values, geo_values, fraction):
    th = normalize(spec, chem_values).reshape(1, -1)
    g = normalize(geo_spec, geo_values).reshape(1, -1)
    x = np.log([rate])
    y = window_features(x, th, window["degree_base"]) @ np.asarray(window["coef_base"])
    y = y + window_features(x, np.column_stack([th, g]), window["degree"]) @ np.asarray(window["coef"])
    return fraction * float(np.exp(y)[0]) / rate


class LiSPANGeoPINN(LiSPANChemPINN):
    """LiSPANChemPINN with two geometry columns at the end of the condition z (module docstring)."""

    def __init__(self, params: LiSPANParams, window: dict, chem_inputs=CHEM_INPUTS_V2, geo_inputs=GEO_INPUTS_V5,
                 rate_min=0.08, rate_max=3.0, ramp_s=30.0, window_fraction=0.95, width=96, depth=4, act="tanh",
                 fourier_t=8, fourier_period=1.0, short_t=(60.0, 600.0), ic_tau_s=60.0, quad_order=12,
                 salt_sep_natural_scale=False):
        nn.Module.__init__(self)
        self.chem_inputs = check_spec(chem_inputs)
        self.geo_inputs = [dict(s) for s in geo_inputs]
        if [s["name"] for s in self.geo_inputs] != ["lnL_cat", "f"]:
            raise ValueError("geometry inputs must be (lnL_cat, f)")
        self.k = len(self.chem_inputs)
        self.p = p = params
        if abs(p.L_cat - L_REF_UM * 1e-6) > 1e-12:
            raise ValueError("the network's params must be the reference cathode (100 um, f = 1; network_params())")
        self.RT = RGAS * p.T
        self.f2 = F / (2.0 * self.RT)
        self.rate_min, self.rate_max = float(rate_min), float(rate_max)
        self.ramp_s = max(float(ramp_s), 1e-6)
        self.window_fraction = float(window_fraction)
        self.fourier_t, self.fourier_period = int(fourier_t), float(fourier_period)
        self.short_t = tuple(float(s) for s in short_t)
        self.ic_tau_s = float(ic_tau_s)
        self.window = {k: window[k] for k in ("coef_base", "degree_base", "coef", "degree")}
        self.window_deg_base, self.window_deg = int(window["degree_base"]), int(window["degree"])
        self.register_buffer("window_coef", torch.as_tensor(window["coef_base"], dtype=torch.float64))
        self.register_buffer("window_corr", torch.as_tensor(window["coef"], dtype=torch.float64))
        self._t_end_cache = {}
        self.register_buffer("chem_lo", torch.as_tensor([s["lo"] for s in self.chem_inputs], dtype=torch.float64))
        self.register_buffer("chem_hi", torch.as_tensor([s["hi"] for s in self.chem_inputs], dtype=torch.float64))
        self.register_buffer("geo_lo", torch.as_tensor([s["lo"] for s in self.geo_inputs], dtype=torch.float64))
        self.register_buffer("geo_hi", torch.as_tensor([s["hi"] for s in self.geo_inputs], dtype=torch.float64))
        n_t = 2 + 2 * self.fourier_t + len(self.short_t) + 1 + self.k + 2
        self.net_R = MLP(1 + n_t, width, depth, n_out=1, act=act)
        self.net_xi = MLP(1 + n_t, width, depth, n_out=3, act=act)
        self.net_ce = MLP(3 + n_t, width, depth, n_out=1, act=act)
        self.net_phie = MLP(3 + n_t, width, depth, n_out=1, act=act)
        with torch.no_grad():
            self.net_xi.net[-1].bias.copy_(torch.tensor([math.log(3.0), math.log(0.3), math.log(0.1)]))
        q, w = gauss_legendre(quad_order)
        self.register_buffer("q_nodes", q)
        self.register_buffer("q_weights", w)
        self.Yi = Y_INTERFACE
        self.kap_e0 = p.kappa0 * p.eps_e_sep ** p.beta_sep
        self.D_ion = {"Li": p.D_salt / (2 * (1 - p.t_plus)), "PF6": p.D_salt / (2 * p.t_plus)}
        self.salt_total_scale = 1.0
        self.charge_total_scale = 0.0
        self.salt_sep_natural = bool(salt_sep_natural_scale)
        self.log_mult = nn.ParameterDict({n: nn.Parameter(torch.zeros(()), requires_grad=False) for n in self.PARAMETERS})

    # ------------------------------------------------------------------ condition z = (rate, chemistry, geometry)
    def theta_hat(self, z):
        lo, hi = self.chem_lo.to(z.dtype), self.chem_hi.to(z.dtype)
        return 2.0 * (z[:, 1:1 + self.k] - lo) / (hi - lo) - 1.0

    def gamma_hat(self, z):
        lo, hi = self.geo_lo.to(z.dtype), self.geo_hi.to(z.dtype)
        return 2.0 * (z[:, 1 + self.k:3 + self.k] - lo) / (hi - lo) - 1.0

    def L_cat(self, z):
        """Cathode thickness [m] per row."""
        return torch.exp(z[:, 1 + self.k:2 + self.k]) * 1e-6

    def f_span(self, z):
        return z[:, 2 + self.k:3 + self.k]

    def eps_e0_rows(self, z):
        p = self.p
        return 1.0 - self.f_span(z) * (p.eps_SPAN + p.eps_CB) - p.eps_L0

    def kappa_s_rows(self, z):
        p = self.p
        return p.kappa_SPAN * (self.f_span(z) * p.eps_SPAN) ** p.beta_cat

    def mass_ratio(self, z):
        return self.f_span(z) * self.L_cat(z) / (L_REF_UM * 1e-6)

    def I0(self, z):
        return ONE_C * z[:, :1] * self.mass_ratio(z)

    def _t_end_rows(self, z):
        r = z[:, :1]
        x = torch.log(r)
        th = self.theta_hat(z)
        y = window_features_torch(x, th, self.window_deg_base) @ self.window_coef.to(z.dtype).view(-1, 1)
        y = y + window_features_torch(x, torch.cat([th, self.gamma_hat(z)], dim=1), self.window_deg) @ \
            self.window_corr.to(z.dtype).view(-1, 1)
        return self.window_fraction * torch.exp(y) / r

    def rate_feature(self, z):
        lo, hi = math.log(self.rate_min), math.log(self.rate_max)
        rf = 2.0 * (torch.log(z[:, :1]) - lo) / max(hi - lo, 1e-12) - 1.0
        return torch.cat([rf, self.theta_hat(z), self.gamma_hat(z)], dim=1)

    # ------------------------------------------------------------------ per-row scales
    def ce_scale(self, z):
        """Largest expected |ln c / c0| (bound of the exponential salt deviation): polarization across the cathode and
        the separator at the full current plus the concentration by the Li2S volume, x 1.3, in [0.05, 2]."""
        p = self.p
        eps0 = self.eps_e0_rows(z)
        L = self.L_cat(z)
        resist = 0.5 * L / eps0 ** p.beta_cat + p.L_sep / p.eps_e_sep ** p.beta_sep
        pol = (1.0 - p.t_plus) * self.I0(z) * resist / (F * p.D_salt * p.c_Li0)
        eps_L_max = p.V_m_L * p.c_init[0] * self.f_span(z) * 2.0
        shrink = torch.log(eps0 / torch.clamp(eps0 - eps_L_max, min=0.05))
        return torch.clamp(1.3 * (pol + shrink) + 0.05, max=2.0)

    def phie_scale(self, z):
        p = self.p
        k_cat = p.kappa0 * self.eps_e0_rows(z) ** p.beta_cat
        return 2.0 * self.I0(z) * (self.L_cat(z) / k_cat + p.L_sep / self.kap_e0)

    def charge_s_scale(self, z):
        return self.I0(z) / self.kappa_s_rows(z)

    def charge_e_scale(self, z):
        return self.I0(z) / self.kap_e0

    def salt_scale(self, z):
        return (1.0 - self.p.t_plus) * self.I0(z) / (F * self.L_cat(z))

    def salt_scale_sep(self, z):
        s = self.salt_scale(z)
        return s * self.L_cat(z) / self.p.L_sep if self.salt_sep_natural else s

    # ------------------------------------------------------------------ cathode
    def species(self, Yc, T, z):
        """Species in reference units (c_S4,0 of f = 1); eps_L, eps_e and the TRUE specific surface a per row."""
        p = self.p
        c40 = p.c_init[0]
        x1, x2, x3 = self._exponents(Yc, T, z)
        e1, e2, e3 = torch.exp(-x1), torch.exp(-x2), torch.exp(-x3)
        xi1 = -torch.expm1(-x1)
        xi2 = xi1 * -torch.expm1(-x2)
        xi3 = xi2 * -torch.expm1(-x3)
        f = self.f_span(z)
        eps_L = p.eps_L0 + p.V_m_L * c40 * f * (xi2 + xi3)
        eps_e = torch.clamp(1.0 - f * (p.eps_SPAN + p.eps_CB) - eps_L, min=1e-3)
        a = f * p.a_SPAN0 * (eps_e / self.eps_e0_rows(z)) ** p.xi
        return {"xi": (xi1, xi2, xi3), "c_S4": c40 * e1, "c_S3": c40 * xi1 * e2, "c_S2": c40 * xi2 * e3,
                "c_S1": c40 * (xi1 + xi3), "eps_L": eps_L, "eps_e": eps_e, "a": a, "f": f,
                "chem": self.chem_values(z)}

    def current_fields(self, Yc, T, z, sp=None):
        sp = sp or self.species(Yc, T, z)
        denom = self._a_rho_nodes(torch.ones_like(Yc), T, z)
        cum = self._a_rho_nodes(Yc, T, z)
        I = self.I(T, z)
        Rr = I / (F * self.L_cat(z)) * self.rho(Yc, T, z) / denom
        return Rr, -I * cum / denom

    def inventory_target(self, T, z):
        """Electrode-averaged sum of the extents: Q / (2 F L_cat f c_S4,0)."""
        return self.charge_passed(T, z) / (2.0 * F * self.L_cat(z) * self.f_span(z) * self.p.c_init[0])

    # ------------------------------------------------------------------ electrolyte
    def ce(self, Y, T, z):
        dev = torch.tanh(self.net_ce(torch.cat([self.efeat(Y), self.tf(T, z)], dim=1)))
        return self.p.c_Li0 * torch.exp(self.ce_scale(z) * self.ic_factor(T, z) * dev)

    # ------------------------------------------------------------------ helpers for users
    def check_inside(self, rate, chem_values, geo_values=None):
        super().check_inside(rate, chem_values)
        if geo_values is None:
            raise ValueError("v5 network: geometry (lnL_cat, f) required")
        for s, v in zip(self.geo_inputs, geo_values):
            if not (s["lo"] - 1e-9 <= v <= s["hi"] + 1e-9):
                raise ValueError(f"{s['name']} = {v:g} outside the trained box [{s['lo']:g}, {s['hi']:g}]")
        if not in_domain(LiSPANParams(), rate, math.exp(geo_values[0]), geo_values[1]):
            raise ValueError(f"rate {rate:g} C with L_cat {math.exp(geo_values[0]):.0f} um, f {geo_values[1]:g} is outside "
                             "the transport domain of the network (geometry.in_domain)")

    def make_z(self, rate, chem_values, n=1, geo_values=None):
        ref = next(self.parameters())
        row = torch.as_tensor([float(rate)] + [float(v) for v in chem_values] + [float(v) for v in geo_values],
                              dtype=ref.dtype, device=ref.device)
        return row.view(1, -1).expand(n, -1).contiguous()

    def window_end(self, rate, chem_values, geo_values=None):
        return geo_window_t_end(self.window, self.chem_inputs, self.geo_inputs, rate, chem_values, geo_values,
                                self.window_fraction)

    def discharge_curve(self, rate: float, chem: dict | None = None, geo: dict | None = None, n: int = 400, t=None,
                        U0_shift: float = 0.0, Z_CC: float | None = None, V_cut: float | None = None) -> dict:
        vals = chem_vector(self.chem_inputs, chem)
        gv = geo_vector(geo)
        self.check_inside(rate, vals, gv)
        te = self.window_end(rate, vals, gv)
        t = np.linspace(0.0, te, n) if t is None else np.asarray(t, float)
        ref = next(self.parameters())
        T = torch.as_tensor(t / te, dtype=ref.dtype, device=ref.device).view(-1, 1)
        z = self.make_z(rate, vals, len(t), gv)
        with torch.no_grad():
            V = self.voltage(T, z, U0_shift=U0_shift, Z_CC=Z_CC).view(-1).cpu().numpy()
            Q = self.charge_passed(T, z).view(-1).cpu().numpy()
        m_S = self.p.m_S_model * float(self.mass_ratio(z[:1]))
        cap = Q / 3.6 / (m_S * 1e3)
        if V_cut is not None:
            below = np.nonzero((V < V_cut) & (t > 100.0))[0]
            if below.size:
                k = below[0] + 1
                t, V, cap = t[:k], V[:k], cap[:k]
        return {"t": t, "V": V, "capacity_mAh_gS": cap, "t_end": te, "rate": float(rate), "geo": geo}


def geo_vector(geo: dict | None):
    """(ln L_cat [um], f) from {'L_cat_um' or 'lnL_cat', 'f'}."""
    if geo is None:
        raise ValueError("geometry required: {'L_cat_um': ..., 'f': ...}")
    lnL = geo["lnL_cat"] if "lnL_cat" in geo else math.log(float(geo["L_cat_um"]))
    return [float(lnL), float(geo["f"])]


# ====================================================================== residuals
def geo_residuals(model: LiSPANGeoPINN, batch):
    """rate_pinn.rate_residuals with per-row geometry (module docstring); same residual names and normalizations."""
    p = model.p
    Yi = model.Yi
    out = {}
    # ---------------- cathode
    Yc, T, z = batch["c"]
    Yc = Yc.clone().requires_grad_(True); T = T.clone().requires_grad_(True)
    te = model.t_end(z)
    L = model.L_cat(z)
    Yg = Yi * Yc
    sp = model.species(Yc, T, z)
    c = model.ce(Yg, T, z)
    a_Li = torch.clamp(c / p.c_Li0, 1e-6)
    Rr, i_e = model.current_fields(Yc, T, z, sp)
    d = model.dphi(Rr, sp, a_Li)
    rr = model.rates(d, sp, a_Li)
    c40 = p.c_init[0]
    a_ref = sp["a"] / sp["f"]                       # a / (f c_S4,0) = a_ref / c_S4,0: f drops out of the extents
    for m, xi in enumerate(sp["xi"]):
        out[f"ode{m + 1}"] = _grad(xi, T) - te * a_ref * rr[m] / (2 * c40)
    dc_dy = _grad(c, Yc) / L
    B = F * (model.D_ion["Li"] - model.D_ion["PF6"]) * sp["eps_e"] ** p.beta_cat * dc_dy * model.mult("D_salt")
    kap = model.kappa_e(c, sp["eps_e"], p.beta_cat)
    phie = model.phie(Yg, T, z)
    dphie_dy = _grad(phie, Yc) / L
    ddphi_dy = _grad(d, Yc) / L
    I = model.I(T, z)
    out["charge_s"] = (ddphi_dy - ((I + i_e) / model.kappa_s_rows(z) + (i_e + B) / kap)) / model.charge_s_scale(z)
    out["charge_e"] = (dphie_dy + (i_e + B) / kap) / model.charge_e_scale(z)
    D = model.D_eff(sp["eps_e"], p.beta_cat)
    dflux_dy = _grad(D * dc_dy, Yc) / L
    dc_dt = _grad(c, T) / te
    deps_dt = _grad(sp["eps_e"], T) / te
    out["salt_c"] = (sp["eps_e"] * dc_dt + c * deps_dt - dflux_dy + (1 - p.t_plus) * sp["a"] * Rr) / model.salt_scale(z)
    # ---------------- separator (i_e = -I)
    Ys, Ts, zs = batch["s"]
    Ys = Ys.clone().requires_grad_(True); Ts = Ts.clone().requires_grad_(True)
    Yg_s = Yi + (1.0 - Yi) * Ys
    cs = model.ce(Yg_s, Ts, zs)
    dcs_dy = _grad(cs, Ys) / p.L_sep
    eps_s = torch.full_like(cs, p.eps_e_sep)
    Bs = F * (model.D_ion["Li"] - model.D_ion["PF6"]) * p.eps_e_sep ** p.beta_sep * dcs_dy * model.mult("D_salt")
    kap_s = model.kappa_e(cs, eps_s, p.beta_sep)
    phie_s = model.phie(Yg_s, Ts, zs)
    Is = model.I(Ts, zs)
    out["charge_e"] = torch.cat([out["charge_e"], (_grad(phie_s, Ys) / p.L_sep + (-Is + Bs) / kap_s) / model.charge_e_scale(zs)])
    Ds = model.D_eff(eps_s, p.beta_sep)
    dflux_s = _grad(Ds * dcs_dy, Ys) / p.L_sep
    out["salt_s"] = (p.eps_e_sep * _grad(cs, Ts) / model.t_end(zs) - dflux_s) / model.salt_scale_sep(zs)
    # ---------------- boundary / interface conditions on the salt (physical slopes via the logical coordinate)
    Tb, zb = batch["b"]
    ss_b = model.salt_scale(zb)
    Lb = model.L_cat(zb)
    Y0 = torch.zeros_like(Tb).requires_grad_(True)
    out["bc_c0"] = _grad(model.ce(Y0, Tb, zb), Y0) * (Yi / Lb) / (ss_b * Lb / p.D_salt)
    Y1 = torch.ones_like(Tb).requires_grad_(True)
    c1 = model.ce(Y1, Tb, zb)
    D1 = model.D_eff(torch.full_like(c1, p.eps_e_sep), p.beta_sep)
    out["bc_an"] = (D1 * _grad(c1, Y1) * ((1.0 - Yi) / p.L_sep) - (1 - p.t_plus) * model.I(Tb, zb) / F) / (ss_b * Lb)
    eps_if = 1e-4
    Ym = torch.full_like(Tb, Yi - eps_if).requires_grad_(True)
    Yp = torch.full_like(Tb, Yi + eps_if).requires_grad_(True)
    cm, cp = model.ce(Ym, Tb, zb), model.ce(Yp, Tb, zb)
    sp_if = model.species(torch.ones_like(Tb), Tb, zb)
    Dm = model.D_eff(sp_if["eps_e"], p.beta_cat); Dp = model.D_eff(torch.full_like(cp, p.eps_e_sep), p.beta_sep)
    out["bc_if"] = (Dm * _grad(cm, Ym) * (Yi / Lb) - Dp * _grad(cp, Yp) * ((1.0 - Yi) / p.L_sep)) / (ss_b * Lb)
    # ---------------- global inventories (quadrature at the boundary sample times)
    if model.salt_total_scale > 0 or model.charge_total_scale > 0:
        q = model.q_nodes.shape[0]; nb = Tb.shape[0]
        nodes = model.q_nodes.to(Tb.dtype).view(1, q); w = model.q_weights.to(Tb.dtype).view(1, q)
        Tq = Tb.view(nb, 1).expand(nb, q).reshape(-1, 1)
        zq = zb.reshape(nb, 1, -1).expand(nb, q, zb.shape[-1]).reshape(nb * q, -1)
        Yc_q = nodes.expand(nb, q).reshape(-1, 1)
    if model.salt_total_scale > 0:
        spq = model.species(Yc_q, Tq, zq)
        inv_c = ((spq["eps_e"] * model.ce(Yi * Yc_q, Tq, zq)).view(nb, q) * w).sum(1, keepdim=True) * Lb
        Ys_q = (Yi + (1.0 - Yi) * nodes).expand(nb, q).reshape(-1, 1)
        inv_s = (model.ce(Ys_q, Tq, zq).view(nb, q) * w).sum(1, keepdim=True) * p.eps_e_sep * p.L_sep
        inv0 = model.eps_e0_rows(zb) * p.c_Li0 * Lb + p.eps_e_sep * p.c_Li0 * p.L_sep
        out["salt_total"] = (inv_c + inv_s - inv0) / (model.salt_total_scale * (Lb + p.L_sep))
    if model.charge_total_scale > 0:
        xi1, xi2, xi3 = model.extents(Yc_q, Tq, zq)
        inv = c40 * ((xi1 + xi2 + xi3).view(nb, q) * w).sum(1, keepdim=True)
        target = model.charge_passed(Tb, zb) / (2.0 * F * Lb * model.f_span(zb))
        out["charge_total"] = (inv - target) / model.charge_total_scale
    return out


# ====================================================================== training pieces
@dataclass
class GeoTrainConfig(ChemTrainConfig):
    chem_inputs: tuple = CHEM_INPUTS_V2
    geo_inputs: tuple = GEO_INPUTS_V5
    rate_min: float = 0.08
    rate_max: float = 3.0
    window_fraction: float = 0.95
    geo_resample_max: int = 50            # rejection sampling of the transport domain: max rounds

    def to_dict(self):
        d = super().to_dict()
        d["geo_inputs"] = [dict(s) for s in self.geo_inputs]
        return d


class GeoSampler(ChemSampler):
    """ChemSampler plus (ln L_cat, f) uniform in their box, restricted to the transport domain (geometry.in_domain:
    rejection, the rate redrawn with the geometry)."""

    def __init__(self, cfg: GeoTrainConfig, dtype, gen, device):
        super().__init__(cfg, dtype, gen, device)
        self.glo = torch.as_tensor([s["lo"] for s in cfg.geo_inputs], dtype=dtype)
        self.ghi = torch.as_tensor([s["hi"] for s in cfg.geo_inputs], dtype=dtype)
        p0 = LiSPANParams()
        self.solid = p0.eps_SPAN + p0.eps_CB
        self.eps_L0 = p0.eps_L0
        self.m_ref = p0.m_S_model * 1e2              # mg S/cm2 of the reference cathode

    def valid(self, r, g):
        from .geometry import LOADING_MAX, THETA_MAX
        L, f = torch.exp(g[:, :1]), g[:, 1:2]
        por = 1.0 - self.solid * f - self.eps_L0
        I = 10.0 * r * f * L / L_REF_UM
        theta = I * L * 1e-6 / por ** 1.5
        load = self.m_ref * f * L / L_REF_UM
        return ((theta <= THETA_MAX) & (load <= LOADING_MAX)).view(-1)

    def conds(self, n):
        out_r, out_g, have = [], [], 0
        for _ in range(self.cfg.geo_resample_max):
            m = 2 * (n - have) + 8
            r = self.rates(m)
            g = self.glo + (self.ghi - self.glo) * torch.rand(m, 2, generator=self.gen, dtype=self.dtype)
            ok = self.valid(r, g)
            out_r.append(r[ok]); out_g.append(g[ok]); have += int(ok.sum())
            if have >= n:
                break
        r, g = torch.cat(out_r)[:n], torch.cat(out_g)[:n]
        if r.shape[0] < n:
            raise RuntimeError("could not draw enough conditions inside the transport domain")
        th = self.lo_t + (self.hi_t - self.lo_t) * torch.rand(n, len(self.lo_t), generator=self.gen, dtype=self.dtype)
        return torch.cat([r, th, g], dim=1)


def evaluate_geo(model: LiSPANGeoPINN, cases: dict) -> dict:
    """evaluate_chem for v5 cases ({'rate', 'chem', 'geo', 'ref', 'select'}): same metric names."""
    prm = next(model.parameters())
    out, rms, mx, sel, test, sel_o, test_o, shifts = {}, [], [], [], [], [], [], []
    for cid in sorted(cases):
        c = cases[cid]
        vals = chem_vector(model.chem_inputs, c["chem"])
        gv = geo_vector(c["geo"])
        te = model.window_end(c["rate"], vals, gv)
        t_all = np.asarray(c["ref"]["t"]); V_all = np.asarray(c["ref"]["V"])
        keep = t_all <= te
        t = t_all[keep]
        T = torch.as_tensor(t / te, dtype=prm.dtype, device=prm.device).view(-1, 1)
        with torch.no_grad():
            V = model.voltage(T, model.make_z(c["rate"], vals, len(t), gv)).view(-1).cpu().numpy()
        e = V - V_all[keep]
        m = t > 100.0
        r_ = 1e3 * float(np.sqrt(np.mean(e[m] ** 2)))
        x_ = 1e3 * float(np.abs(e[m]).max())
        _, r_out, sh, _ = cliff_metrics(t[m] / t_all[-1], V_all[keep][m], V[m].astype(float))
        out[f"V_rmse_mV@{cid}"], out[f"V_max_mV@{cid}"] = r_, x_
        out[f"V_rmse_out_mV@{cid}"] = r_out
        out[f"cliff_shift_pct@{cid}"] = max((abs(v) for v in sh), default=0.0)
        out[f"coverage@{cid}"] = float(te / t_all[-1])
        rms.append(r_); mx.append(x_); shifts += [abs(v) for v in sh]
        (sel if c.get("select") else test).append(r_)
        (sel_o if c.get("select") else test_o).append(r_out)
    if rms:
        out.update({"V_rmse_mean_mV": float(np.mean(rms)), "V_rmse_worst_mV": float(np.max(rms)),
                    "V_max_worst_mV": float(np.max(mx)),
                    "V_rmse_select_mV": float(np.mean(sel)) if sel else float(np.mean(rms)),
                    "V_rmse_test_mV": float(np.mean(test)) if test else float("nan"),
                    "V_rmse_out_select_mV": float(np.mean(sel_o)) if sel_o else float("nan"),
                    "V_rmse_out_test_mV": float(np.mean(test_o)) if test_o else float("nan"),
                    "V_rmse_out_worst_test_mV": float(np.max(test_o)) if test_o else float("nan"),
                    "cliff_shift_max_pct": float(max(shifts, default=0.0))})
    return out


def build_geo_model(cfg: GeoTrainConfig, params: LiSPANParams, window: dict) -> LiSPANGeoPINN:
    m = LiSPANGeoPINN(params, window, cfg.chem_inputs, cfg.geo_inputs, cfg.rate_min, cfg.rate_max, cfg.ramp_s,
                      cfg.window_fraction, cfg.width, cfg.depth, cfg.act, cfg.fourier_t, cfg.fourier_period,
                      tuple(cfg.short_t), cfg.ic_tau_s, cfg.quad_order, cfg.salt_sep_natural_scale)
    m.inventory_hard = bool(getattr(cfg, "inventory_hard", False))
    return m


def warm_start_from_chem(model: LiSPANGeoPINN, checkpoint) -> dict:
    """Copy a v2-box chem checkpoint (e.g. chem_v4_warm_salt_s0) into a v5 model: every layer as it is, the two
    geometry columns of the first layers set to zero.  NOT the same function: the separator, the electrolyte
    coordinate (fixed interface) and the salt form (exponential) differ, so this is only a starting point."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"]
    copied = 0
    with torch.no_grad():
        for name in ("net_R", "net_xi", "net_ce", "net_phie"):
            for i, layer in enumerate(getattr(model, name).net):
                if not isinstance(layer, nn.Linear):
                    continue
                w, b = sd[f"{name}.net.{i}.weight"], sd[f"{name}.net.{i}.bias"]
                if i == 0:
                    if w.shape[1] != layer.weight.shape[1] - 2 or w.shape[0] != layer.weight.shape[0]:
                        raise ValueError(f"{name}: chem layer {tuple(w.shape)} does not fit {tuple(layer.weight.shape)}")
                    layer.weight.zero_(); layer.weight[:, :w.shape[1]].copy_(w.to(layer.weight.dtype))
                else:
                    layer.weight.copy_(w.to(layer.weight.dtype))
                layer.bias.copy_(b.to(layer.bias.dtype))
                copied += 1
    return {"layers": copied, "rate_step": ck.get("step"), "group_weights": ck.get("group_weights", {}), "source": "chem"}


def load_geo_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = GeoTrainConfig.from_dict(ck["train"])
    params = params_from_dict(ck["params"])
    model = build_geo_model(cfg, params, ck["window"])
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    model.load_state_dict(ck["model"])
    model.to(device=resolve_device(device), dtype=dtype).eval()
    return model, cfg, params


__all__ = ["LiSPANGeoPINN", "GeoTrainConfig", "GeoSampler", "geo_residuals", "evaluate_geo", "build_geo_model",
           "load_geo_model", "warm_start_from_chem", "network_params", "geo_window_t_end", "geo_vector", "Y_INTERFACE",
           "SEP_POLYOLEFIN"]
