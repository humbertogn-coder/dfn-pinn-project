"""Phase 3b: the Li-SPAN PINN conditioned on the C-rate AND on chemistry parameters (docs/PHASE3B_CHEM_PINN.md).

One trained network returns the discharge (voltage and internal fields) of the Simanjuntak cell for any C-rate in
[rate_min, rate_max] and any chemistry inside a box of k parameters, so that a two-curve calibration needs no new
training.

Condition vector z, shape (n, 1 + k): z[:, 0] = C-rate r; z[:, 1 + j] = chemistry input j in its own unit:
    "dU0_m"     offset of the equilibrium potential of step m [V]
    "dU0_3rel2" offset of step 3 relative to step 2 [V]: U0_3 moves by dU0_2 + dU0_3rel2, so the step-2/3 gap is
                0.24 V - dU0_3rel2 and a box on it keeps the steps in order (v2)
    "b"         natural log of a common multiplier of the three OCV slopes b_1..b_3   ("b_m": of b_m alone;
                "b_12": of b_1 and b_2 together)
    "k0_m"      natural log of a multiplier of the rate constant of step m
The networks see each input mapped to [-1, 1] over its box (appended to the time features, like the rate).

Parameters handled analytically, after the network (exact invariances of the model; the first two checked with the
finite-volume solver to 1e-6 V, scripts/lispan_chem_design.py --check-invariances):
    common shift d of U0_1..U0_3   -> V + d   (every rate depends on Delta phi - U0_m only)
    contact resistance Z_CC         -> V - Z_CC I   (a series term; no field depends on it)
    effective sulfur fraction w_S   -> capacity x w_S, current / w_S (a scaling of the data axes, lispan_lit_rates.py)
For the irreversible steps 2-3 only k0_m exp(F U0_m / 2RT) enters, so "dU0_m" also stands for their k0_m.

Time window: T = t / t_end(r, theta), ln(r t_cut) a polynomial in (ln r, theta_hat) (degree 2-4; v1 and v2 use 4)
fitted to FV cut-off times over a design of chemistries (scripts/lispan_chem_design.py), computed at the network's own
contact resistance params.Z_CC (the window value: no larger than any Z_CC to be fitted, so that the window covers their
discharges; v1 0.010 Ohm m2, v2 0).

Training aids (v2): ChemRARSampler (residual-based sampling at the voltage cliffs) and cliff_metrics (voltage error
outside the cliffs and the shift of each cliff), reported by evaluate_chem.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import functools
import itertools
import math

import numpy as np
import torch
from torch import nn

from ..v2.model import MLP, gauss_legendre
from .params import F, R as RGAS, LiSPANParams
from .rate_pinn import ONE_C, LiSPANRatePINN, RateSampler, RateTrainConfig, params_from_dict, resolve_device

CHEM_INPUTS_V1 = (
    {"name": "dU0_2", "lo": -0.12, "hi": 0.12},
    {"name": "dU0_3", "lo": -0.12, "hi": 0.12},
    {"name": "b", "lo": math.log(0.6), "hi": math.log(1.5)},
    {"name": "k0_1", "lo": math.log(0.1), "hi": math.log(10.0)},
)
# v2 (2026-10-25): the box of the literature fits (docs/PHASE3B_CHEM_PINN.md, "v1 results", point 5): step 2 up to
# +0.28 V (gap 1-2 >= 0.02 V), the step-2/3 gap 0.04-0.50 V (always in order), separate OCV slopes for steps 1-2 and 3,
# k0_1 up to x100.  Used with the network's contact resistance 0 (window covers every Z_CC >= 0).
CHEM_INPUTS_V2 = (
    {"name": "dU0_2", "lo": -0.15, "hi": 0.28},
    {"name": "dU0_3rel2", "lo": -0.26, "hi": 0.20},
    {"name": "b_12", "lo": math.log(0.5), "hi": math.log(1.6)},
    {"name": "b_3", "lo": math.log(0.4), "hi": math.log(3.5)},
    {"name": "k0_1", "lo": math.log(0.1), "hi": math.log(100.0)},
)
KNOWN_INPUTS = ("dU0_1", "dU0_2", "dU0_3", "dU0_3rel2", "b", "b_1", "b_2", "b_3", "b_12", "k0_1", "k0_2", "k0_3")
CHEM_SPECS = {"v1": CHEM_INPUTS_V1, "v2": CHEM_INPUTS_V2}


def check_spec(spec):
    """Validated list of {'name', 'lo', 'hi'} (floats); the nominal chemistry (all inputs 0) must lie in the box."""
    out, seen = [], set()
    for s in spec:
        name, lo, hi = str(s["name"]), float(s["lo"]), float(s["hi"])
        if name not in KNOWN_INPUTS:
            raise ValueError(f"unknown chemistry input {name!r} (known: {KNOWN_INPUTS})")
        if name in seen:
            raise ValueError(f"chemistry input {name!r} given twice")
        if not lo < hi or not lo <= 0.0 <= hi:
            raise ValueError(f"chemistry input {name!r}: need lo < hi and 0 (nominal) inside [{lo}, {hi}]")
        if name == "b" and any(n.startswith("b_") for n in seen) or name.startswith("b_") and "b" in seen:
            raise ValueError("use either the common 'b' or the individual 'b_m', not both")
        if name == "b_12" and seen & {"b_1", "b_2"} or name in ("b_1", "b_2") and "b_12" in seen:
            raise ValueError("'b_12' already moves b_1 and b_2")
        if {name} | seen >= {"dU0_3", "dU0_3rel2"}:
            raise ValueError("use either 'dU0_3' or 'dU0_3rel2', not both")
        seen.add(name)
        out.append({"name": name, "lo": lo, "hi": hi})
    return out


def chem_vector(spec, chem: dict | None):
    """Values of the inputs in spec order (missing = 0 = nominal)."""
    chem = chem or {}
    unknown = set(chem) - {s["name"] for s in spec}
    if unknown:
        raise ValueError(f"chemistry inputs {sorted(unknown)} are not inputs of this network")
    return [float(chem.get(s["name"], 0.0)) for s in spec]


def chem_effects(spec, values):
    """Offsets of U0_1..U0_3 [V] and natural-log multipliers of b_1..b_3 and k0_1..k0_3 caused by the inputs; values may
    be floats or (n, 1) tensors (the same linear map serves the FV solver and the network)."""
    dU, lb, lk = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    named = {s["name"]: v for s, v in zip(spec, values)}
    for name, v in named.items():
        if name == "dU0_3rel2":
            dU[2] = dU[2] + v + named.get("dU0_2", 0.0)
        elif name.startswith("dU0_"):
            m = int(name[-1]) - 1
            dU[m] = dU[m] + v
        elif name == "b":
            lb = [x + v for x in lb]
        elif name == "b_12":
            lb[0], lb[1] = lb[0] + v, lb[1] + v
        elif name.startswith("b_"):
            m = int(name[-1]) - 1
            lb[m] = lb[m] + v
        elif name.startswith("k0_"):
            m = int(name[-1]) - 1
            lk[m] = lk[m] + v
    return dU, lb, lk


def apply_chem(params: LiSPANParams, spec, values) -> LiSPANParams:
    """LiSPANParams with the chemistry inputs applied (for the finite-volume solver)."""
    dU, lb, lk = chem_effects(spec, [float(v) for v in values])
    U0 = tuple(u + d for u, d in zip(params.U0, dU))
    b = tuple(x * math.exp(v) for x, v in zip(params.b, lb))
    k0 = tuple(x * math.exp(v) for x, v in zip(params.k0, lk))
    return replace(params, U0=U0, b=b, k0=k0)


def normalize(spec, values):
    """Box values -> [-1, 1] (numpy, shape (..., k))."""
    lo = np.array([s["lo"] for s in spec]); hi = np.array([s["hi"] for s in spec])
    return 2.0 * (np.asarray(values, float) - lo) / (hi - lo) - 1.0


# ====================================================================== window
def _monomials(n_vars: int, degree: int):
    """Index tuples of all monomials of total degree <= degree in n_vars variables, ordered by degree and then
    lexicographically: (), (0,), (1,), ..., (0, 0), (0, 1), ...  For degree 2 and variables (x, th_1..th_k) this is
    [1, x, th_j, x^2, x th_j, th_i th_j (i <= j)]."""
    return [c for d in range(degree + 1) for c in itertools.combinations_with_replacement(range(n_vars), d)]


def n_window_features(k: int, degree: int = 2) -> int:
    return math.comb(k + 1 + degree, degree)


def window_degree(n_coef: int, k: int) -> int:
    for d in range(1, 9):
        if n_window_features(k, d) == n_coef:
            return d
    raise ValueError(f"{n_coef} window coefficients fit no polynomial degree for k = {k}")


def window_features(x, th, degree=2):
    """Polynomial of total degree `degree` in (x = ln r, th = normalized chemistry (N, k)); see _monomials."""
    x = np.asarray(x, float).reshape(-1)
    v = np.column_stack([x, np.asarray(th, float).reshape(len(x), -1)])
    cols = [np.prod(v[:, list(m)], axis=1) if m else np.ones_like(x) for m in _monomials(v.shape[1], degree)]
    return np.stack(cols, axis=1)


@functools.lru_cache(maxsize=None)
def _monomial_plan(n_vars: int, degree: int):
    """For each degree d >= 1, (parent column, variable) of every degree-d monomial in the _monomials order: monomial
    m = parent * v[m[-1]] with parent = m[:-1], which sits in an earlier column."""
    mons = _monomials(n_vars, degree)
    pos = {m: i for i, m in enumerate(mons)}
    plan = []
    for d in range(1, degree + 1):
        block = [m for m in mons if len(m) == d]
        plan.append((tuple(pos[m[:-1]] for m in block), tuple(m[-1] for m in block)))
    return tuple(plan)


def window_features_torch(x, th, degree=2):
    """Torch version of window_features, built degree by degree (one gather and one product per degree; the
    window is evaluated ~60 times per training step on ~2e5 rows, so a loop over the 126 monomials is too slow)."""
    v = torch.cat([x, th], dim=1)
    feats = torch.ones_like(x)
    for parent, var in _monomial_plan(v.shape[1], degree):
        idx_p = torch.tensor(parent, device=v.device)
        idx_v = torch.tensor(var, device=v.device)
        feats = torch.cat([feats, feats.index_select(1, idx_p) * v.index_select(1, idx_v)], dim=1)
    return feats


def fit_chem_window(rates, th_hat, t_cut, degree=2):
    """Least squares of ln(r t_cut) on window_features; returns (coef list, max and rms relative error of t_cut)."""
    r = np.asarray(rates, float)
    A = window_features(np.log(r), th_hat, degree)
    y = np.log(r * np.asarray(t_cut, float))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rel = np.exp(A @ coef - y) - 1.0
    return [float(c) for c in coef], float(np.max(np.abs(rel))), float(np.sqrt(np.mean(rel ** 2)))


def chem_window_t_end(coef, spec, rate, chem_values, fraction=0.98):
    """t_end [s] of the network window (numpy/float)."""
    th = normalize(spec, chem_values).reshape(1, -1)
    feats = window_features(np.log([rate]), th, window_degree(len(coef), th.shape[1]))
    return fraction * float(np.exp(feats @ np.asarray(coef, float))[0]) / rate


# ====================================================================== model
class LiSPANChemPINN(LiSPANRatePINN):
    """LiSPANRatePINN whose condition argument z carries the C-rate and k chemistry inputs (module docstring)."""

    def __init__(self, params: LiSPANParams, window_coef, chem_inputs=CHEM_INPUTS_V1, rate_min=0.1, rate_max=1.0,
                 ramp_s=30.0, window_fraction=0.98, width=96, depth=4, act="tanh", fourier_t=8, fourier_period=1.0,
                 short_t=(60.0, 600.0), ic_tau_s=60.0, quad_order=12, salt_sep_natural_scale=False):
        nn.Module.__init__(self)
        self.chem_inputs = check_spec(chem_inputs)
        self.k = len(self.chem_inputs)
        self.p = p = params
        self.RT = RGAS * p.T
        self.f2 = F / (2.0 * self.RT)
        self.rate_min, self.rate_max = float(rate_min), float(rate_max)
        self.ramp_s = max(float(ramp_s), 1e-6)
        self.window_fraction = float(window_fraction)
        self.fourier_t, self.fourier_period = int(fourier_t), float(fourier_period)
        self.short_t = tuple(float(s) for s in short_t)
        self.ic_tau_s = float(ic_tau_s)
        coef = list(window_coef)
        self.window_deg = window_degree(len(coef), self.k)
        self.register_buffer("window_coef", torch.as_tensor(coef, dtype=torch.float64))
        self._t_end_cache = {}
        self.register_buffer("chem_lo", torch.as_tensor([s["lo"] for s in self.chem_inputs], dtype=torch.float64))
        self.register_buffer("chem_hi", torch.as_tensor([s["hi"] for s in self.chem_inputs], dtype=torch.float64))
        n_t = 2 + 2 * self.fourier_t + len(self.short_t) + 1 + self.k      # time features + rate + chemistry
        self.net_R = MLP(1 + n_t, width, depth, n_out=1, act=act)
        self.net_xi = MLP(1 + n_t, width, depth, n_out=3, act=act)
        self.net_ce = MLP(3 + n_t, width, depth, n_out=1, act=act)
        self.net_phie = MLP(3 + n_t, width, depth, n_out=1, act=act)
        with torch.no_grad():
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

    # ------------------------------------------------------------------ condition z = (rate, chemistry)
    def theta_hat(self, z):
        lo, hi = self.chem_lo.to(z.dtype), self.chem_hi.to(z.dtype)
        return 2.0 * (z[:, 1:] - lo) / (hi - lo) - 1.0

    def I0(self, z):
        return ONE_C * z[:, :1]

    def _t_end(self, z):
        q = self.q_nodes.shape[0]
        n, d = z.shape
        if n >= 4 * q and n % q == 0 and not z.requires_grad:
            # quadrature-expanded conditions (rate_pinn._a_rho_nodes and the inventory terms) repeat each row q times
            # in a row: evaluate the window once per row (the degree-6 window of v3 has 924 features; 2 x 18 432
            # rows cost ~0.3 s per step otherwise).  The check costs one comparison per element.
            zq = z.view(n // q, q, d)
            if torch.equal(zq, zq[:, :1].expand(n // q, q, d)):
                return self._t_end_rows(zq[:, 0]).view(n // q, 1, 1).expand(n // q, q, 1).reshape(n, 1)
        return self._t_end_rows(z)

    def _t_end_rows(self, z):
        r = z[:, :1]
        feats = window_features_torch(torch.log(r), self.theta_hat(z), self.window_deg)
        return self.window_fraction * torch.exp(feats @ self.window_coef.to(z.dtype).view(-1, 1)) / r

    def t_end(self, z):
        """Window end per point.  One residual evaluation asks ~60 times for the window of the same ~6 condition
        tensors, so the result is cached per tensor (key: storage address, shape, strides, in-place version; the
        cache holds the tensor itself, so its address cannot be reused by another tensor while the entry lives)."""
        if z.requires_grad or torch.is_grad_enabled() and self.window_coef.requires_grad:
            return self._t_end(z)
        key = (z.data_ptr(), tuple(z.shape), z.stride(), z._version, z.dtype, str(z.device), self.window_fraction,
               self.window_coef._version)
        hit = self._t_end_cache.get(key)
        if hit is not None and hit[0].data_ptr() == key[0]:
            return hit[1]
        out = self._t_end(z)
        if len(self._t_end_cache) >= 16:
            self._t_end_cache.pop(next(iter(self._t_end_cache)))
        self._t_end_cache[key] = (z, out)
        return out

    def rate_feature(self, z):
        lo, hi = math.log(self.rate_min), math.log(self.rate_max)
        rf = 2.0 * (torch.log(z[:, :1]) - lo) / max(hi - lo, 1e-12) - 1.0
        return torch.cat([rf, self.theta_hat(z)], dim=1)

    def chem_values(self, z):
        """Per-point U0_m [V], b_m [V], k0_m [mol/m2/s] (each a float or an (n, 1) tensor)."""
        cols = [z[:, 1 + j:2 + j] for j in range(self.k)]
        dU, lb, lk = chem_effects(self.chem_inputs, cols)
        U0 = [self.U0(m) + dU[m] for m in range(3)]
        b = [self.b(m) * (torch.exp(lb[m]) if torch.is_tensor(lb[m]) else math.exp(lb[m])) for m in range(3)]
        k0 = [self.k0(m) * (torch.exp(lk[m]) if torch.is_tensor(lk[m]) else math.exp(lk[m])) for m in range(3)]
        return {"U0": U0, "b": b, "k0": k0}

    def species(self, Yc, T, z):
        sp = super().species(Yc, T, z)
        sp["chem"] = self.chem_values(z)
        return sp

    def rates(self, dphi, sp, a_Li):
        """LiSPANPINN.rates with per-point U0, b, k0 (sp['chem'])."""
        ch = sp.get("chem")
        if ch is None:
            return super().rates(dphi, sp, a_Li)
        p = self.p
        fl = 1e-30
        aS4, aS3, aS2, aS1 = (sp["c_S4"] / p.c_ref[0] + fl, sp["c_S3"] / p.c_ref[1] + fl,
                              sp["c_S2"] / p.c_ref[2] + fl, sp["c_S1"] / p.c_ref[3] + fl)
        aS = p.c_S_sat / p.c_S_ref
        zeta = [torch.clamp(1 - aS4, 0, 1), torch.clamp(1 - aS3, 0, 1), torch.clamp(1 - aS2, 0, 1)]
        eds = [torch.sqrt(aS4) * a_Li, torch.sqrt(aS3), torch.sqrt(aS2)]
        prods = [torch.sqrt(aS3 * aS1), torch.sqrt(aS2 * aS), torch.sqrt(aS1 * aS)]
        out = []
        for m in range(3):
            x = torch.clamp(self.f2 * (dphi - (ch["U0"][m] - ch["b"][m] * zeta[m])), -60.0, 60.0)
            rev = 1.0 if p.reversible[m] else 0.0
            out.append(ch["k0"][m] * (eds[m] * torch.exp(-x) - rev * prods[m] * torch.exp(x)))
        return out

    # ------------------------------------------------------------------ voltage and curves
    def voltage(self, T, z, U0_shift=0.0, Z_CC=None):
        """Cell voltage; U0_shift (common shift of U0_1..3, V) and Z_CC (Ohm m2, default the network's) analytic."""
        Y0 = torch.zeros_like(T)
        sp = self.species(Y0, T, z)
        a_Li = torch.clamp(self.ce(Y0, T, z) / self.p.c_Li0, 1e-6)
        Rr, _ = self.current_fields(Y0, T, z, sp)
        d = self.dphi(Rr, sp, a_Li)
        Z = self.p.Z_CC * self.mult("Z_CC") if Z_CC is None else Z_CC
        return d + self.phie(Y0, T, z) - Z * self.I(T, z) + U0_shift

    def terminal_voltage(self, T, z):
        return self.voltage(T, z)

    def check_inside(self, rate, chem_values):
        if not (self.rate_min * (1 - 1e-9) <= rate <= self.rate_max * (1 + 1e-9)):
            raise ValueError(f"C-rate {rate:g} outside the trained range [{self.rate_min:g}, {self.rate_max:g}]")
        for s, v in zip(self.chem_inputs, chem_values):
            if not (s["lo"] - 1e-9 <= v <= s["hi"] + 1e-9):
                raise ValueError(f"{s['name']} = {v:g} outside the trained box [{s['lo']:g}, {s['hi']:g}]")

    def make_z(self, rate, chem_values, n=1):
        ref = next(self.parameters())
        row = torch.as_tensor([float(rate)] + [float(v) for v in chem_values], dtype=ref.dtype, device=ref.device)
        return row.view(1, -1).expand(n, -1).contiguous()

    def window_end(self, rate, chem_values):
        return chem_window_t_end(self.window_coef.tolist(), self.chem_inputs, rate, chem_values, self.window_fraction)

    def discharge_curve(self, rate: float, chem: dict | None = None, n: int = 400, t=None, U0_shift: float = 0.0,
                        Z_CC: float | None = None, V_cut: float | None = None) -> dict:
        """Voltage curve inside the trained box: {'t' [s], 'V' [V], 'capacity_mAh_gS', 't_end'}; with V_cut the curve
        stops where V first falls below it (the window may extend past the cut-off of this Z_CC / U0_shift)."""
        vals = chem_vector(self.chem_inputs, chem)
        self.check_inside(rate, vals)
        te = self.window_end(rate, vals)
        t = np.linspace(0.0, te, n) if t is None else np.asarray(t, float)
        ref = next(self.parameters())
        T = torch.as_tensor(t / te, dtype=ref.dtype, device=ref.device).view(-1, 1)
        z = self.make_z(rate, vals, len(t))
        with torch.no_grad():
            V = self.voltage(T, z, U0_shift=U0_shift, Z_CC=Z_CC).view(-1).cpu().numpy()
            Q = self.charge_passed(T, z).view(-1).cpu().numpy()
        cap = Q / 3.6 / (self.p.m_S_model * 1e3)
        if V_cut is not None:
            below = np.nonzero((V < V_cut) & (t > 100.0))[0]
            if below.size:
                k = below[0] + 1
                t, V, cap = t[:k], V[:k], cap[:k]
        return {"t": t, "V": V, "capacity_mAh_gS": cap, "t_end": te, "rate": float(rate), "chem": dict(zip(
            [s["name"] for s in self.chem_inputs], vals)), "U0_shift": float(U0_shift),
            "Z_CC": float(self.p.Z_CC if Z_CC is None else Z_CC)}


# ====================================================================== training pieces
@dataclass
class ChemTrainConfig(RateTrainConfig):
    chem_inputs: tuple = CHEM_INPUTS_V1
    design_file: str = ""                 # window design + evaluation cases (scripts/lispan_chem_design.py)
    init_from: str = ""                   # rate (Phase 3a) or chem checkpoint to warm-start from (rate: chemistry columns 0)
    # residual-based sampling (v2; ChemRARSampler): every rar_every steps the kinetics residuals (ode1-3 + charge_s) are
    # evaluated on rar_candidates uniform cathode points; the rar_pool worst are kept and rar_fraction of every cathode
    # batch is drawn from them with a small jitter (the voltage cliffs of v1 sit where these residuals peak)
    rar: bool = False
    rar_every: int = 250
    rar_candidates: int = 8192
    rar_pool: int = 1024
    rar_fraction: float = 0.25
    rar_chunk: int = 2048
    rar_jitter_T: float = 0.003           # in T = t / t_end
    rar_jitter_Y: float = 0.05
    rar_jitter_lnr: float = 0.02
    rar_jitter_theta: float = 0.01        # fraction of each box width

    def to_dict(self):
        d = super().to_dict()
        d["chem_inputs"] = [dict(s) for s in self.chem_inputs]
        return d


class ChemSampler(RateSampler):
    """RateSampler plus the chemistry inputs, uniform over their box (independent per point)."""

    def __init__(self, cfg: ChemTrainConfig, dtype, gen, device):
        super().__init__(cfg, dtype, gen, device)
        spec = check_spec(cfg.chem_inputs)
        self.lo_t = torch.as_tensor([s["lo"] for s in spec], dtype=dtype)
        self.hi_t = torch.as_tensor([s["hi"] for s in spec], dtype=dtype)

    def conds(self, n):
        th = self.lo_t + (self.hi_t - self.lo_t) * torch.rand(n, len(self.lo_t), generator=self.gen, dtype=self.dtype)
        return torch.cat([self.rates(n), th], dim=1)

    def draw(self):
        c, dev = self.cfg, self.device
        return {"c": (self.u(c.n_cathode).to(dev), self.times(c.n_cathode).to(dev), self.conds(c.n_cathode).to(dev)),
                "s": (self.u(c.n_separator).to(dev), self.times(c.n_separator).to(dev), self.conds(c.n_separator).to(dev)),
                "b": (self.times(c.n_boundary).to(dev), self.conds(c.n_boundary).to(dev))}


class ChemRARSampler(ChemSampler):
    """ChemSampler plus residual-based refinement of the cathode points (cfg.rar_*).  train_rate calls
    before_step(step, model, residual_fn) before every draw; the pool of high-residual points is rebuilt every
    rar_every steps (and on the first step, also after a resume).  Separator and boundary points stay uniform."""

    KIN = ("ode1", "ode2", "ode3", "charge_s")

    def __init__(self, cfg: ChemTrainConfig, dtype, gen, device):
        super().__init__(cfg, dtype, gen, device)
        self.pool = None
        self.last_refresh = None
        self.stats = {}

    def before_step(self, step, model, residual_fn):
        c = self.cfg
        if not c.rar or (self.pool is not None and step - self.last_refresh < c.rar_every):
            return
        self.refresh(model, residual_fn)
        self.last_refresh = step

    def refresh(self, model, residual_fn):
        c, dev = self.cfg, self.device
        Y, T, Z = self.u(c.rar_candidates), self.u(c.rar_candidates), self.conds(c.rar_candidates)
        scores = []
        for i in range(0, c.rar_candidates, c.rar_chunk):
            sl = slice(i, i + c.rar_chunk)
            n = Y[sl].shape[0]
            batch = {"c": (Y[sl].to(dev), T[sl].to(dev), Z[sl].to(dev)),
                     "s": (self.u(8).to(dev), self.times(8).to(dev), self.conds(8).to(dev)),
                     "b": (self.times(4).to(dev), self.conds(4).to(dev))}
            terms = residual_fn(model, batch)
            sc = sum(terms[k][:n].detach().square() for k in self.KIN if k in terms)
            scores.append(sc.view(-1).cpu())
            del terms
        score = torch.cat(scores)
        score = torch.where(torch.isfinite(score), score, torch.full_like(score, -1.0))
        top = torch.topk(score, min(c.rar_pool, score.numel())).indices
        self.pool = (Y[top], T[top], Z[top])
        med = float(score.median())
        self.stats = {"pool_min": float(score[top].min()), "median": med,
                      "ratio": float(score[top].min()) / med if med > 0 else float("nan")}

    @staticmethod
    def _reflect(x):
        """Jittered coordinates back into (0, 1] by reflection at the ends (no pile-up at an end, never exactly 0)."""
        x = 1.0 - torch.abs(1.0 - torch.abs(x))
        return torch.clamp(x, 1e-6, 1.0)

    def _from_pool(self, n):
        c = self.cfg
        Y, T, Z = self.pool
        idx = torch.randint(0, Y.shape[0], (n,), generator=self.gen)
        g = lambda: torch.randn(n, 1, generator=self.gen, dtype=self.dtype)          # noqa: E731
        Yn = self._reflect(Y[idx] + c.rar_jitter_Y * g())
        Tn = self._reflect(T[idx] + c.rar_jitter_T * g())         # T = 0 exactly gives non-finite time derivatives
        Zn = Z[idx].clone()
        Zn[:, :1] = torch.clamp(Zn[:, :1] * torch.exp(c.rar_jitter_lnr * g()), c.rate_min, c.rate_max)
        k = Zn.shape[1] - 1
        noise = torch.randn(n, k, generator=self.gen, dtype=self.dtype) * c.rar_jitter_theta * (self.hi_t - self.lo_t)
        Zn[:, 1:] = torch.minimum(torch.maximum(Zn[:, 1:] + noise, self.lo_t), self.hi_t)
        return Yn, Tn, Zn

    def draw(self):
        out = super().draw()
        c, dev = self.cfg, self.device
        if not c.rar or self.pool is None:
            return out
        n_r = int(round(c.rar_fraction * c.n_cathode))
        if n_r <= 0:
            return out
        Yu, Tu, Zu = out["c"]
        Yr, Tr, Zr = self._from_pool(n_r)
        out["c"] = (torch.cat([Yu[n_r:], Yr.to(dev)]), torch.cat([Tu[n_r:], Tr.to(dev)]), torch.cat([Zu[n_r:], Zr.to(dev)]))
        return out


def cliff_metrics(x, V_ref, V_mod, slope=5.0, pad=0.01):
    """Voltage errors split at the voltage cliffs of the reference (docs/PHASE3B_CHEM_PINN.md, "v1 results"):
    x = t / FV cut-off time, voltages in V.  A cliff = where the reference falls faster than `slope` V per unit x;
    points within `pad` of one are left out of rms_out.  shifts = x-shift [% of the discharge] of the mid-voltage
    crossing of each cliff (model minus reference).  Returns (rms_all_mV, rms_out_mV, shifts, share_masked)."""
    x, V_ref, V_mod = (np.asarray(a, float) for a in (x, V_ref, V_mod))
    e = 1e3 * (V_mod - V_ref)
    fast = np.nonzero(-np.gradient(V_ref, x) > slope)[0]
    mask = np.zeros(len(x), bool)
    for i in fast:
        mask |= np.abs(x - x[i]) < pad
    rms_all = float(np.sqrt(np.mean(e ** 2)))
    rms_out = float(np.sqrt(np.mean(e[~mask] ** 2))) if (~mask).any() else rms_all
    shifts = []
    if fast.size:
        for g in np.split(fast, np.nonzero(np.diff(fast) > 3)[0] + 1):
            v_mid = 0.5 * (V_ref[g[0]] + V_ref[g[-1]])
            lo_, hi_ = max(g[0] - 20, 0), min(g[-1] + 20, len(x) - 1)
            xs = x[lo_:hi_ + 1]

            def cross(v):
                d = v[lo_:hi_ + 1] - v_mid
                j = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]
                if not j.size:
                    return float("nan")
                j = j[0]
                return xs[j] - d[j] * (xs[j + 1] - xs[j]) / (d[j + 1] - d[j])

            sft = 100.0 * (cross(V_mod) - cross(V_ref))
            if np.isfinite(sft):
                shifts.append(float(sft))
    return rms_all, rms_out, shifts, float(mask.mean())


def evaluate_chem(model: LiSPANChemPINN, cases: dict) -> dict:
    """cases = {id: {'rate', 'chem' (dict), 'ref' (FV result at the network's Z_CC), 'select' (bool)}} -> per-case
    voltage errors ('V_rmse_mV@<id>', 'V_max_mV@<id>', 'coverage@<id>' = window end / FV cut-off, and the cliff-aware
    'V_rmse_out_mV@<id>' / 'cliff_shift_pct@<id>' of cliff_metrics) with their summaries: mean / worst, 'V_rmse_select_mV'
    (mean over the select cases: best.pt) and 'V_rmse_test_mV' (the others), 'V_rmse_out_select_mV',
    'V_rmse_out_test_mV', 'V_rmse_out_worst_test_mV', 'cliff_shift_max_pct'."""
    prm = next(model.parameters())
    out, rms, mx, sel, test, sel_o, test_o, shifts = {}, [], [], [], [], [], [], []
    for cid in sorted(cases):
        c = cases[cid]
        vals = chem_vector(model.chem_inputs, c["chem"])
        te = model.window_end(c["rate"], vals)
        t_all = np.asarray(c["ref"]["t"]); V_all = np.asarray(c["ref"]["V"])
        keep = t_all <= te
        t = t_all[keep]
        T = torch.as_tensor(t / te, dtype=prm.dtype, device=prm.device).view(-1, 1)
        with torch.no_grad():
            V = model.voltage(T, model.make_z(c["rate"], vals, len(t))).view(-1).cpu().numpy()
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


def chem_eval_summary(ev: dict) -> str:
    out = (f"V rmse [mV]: select {ev['V_rmse_select_mV']:.3g} test {ev['V_rmse_test_mV']:.3g} | mean "
           f"{ev['V_rmse_mean_mV']:.3g} worst {ev['V_rmse_worst_mV']:.3g} | V max worst {ev['V_max_worst_mV']:.3g}")
    if "V_rmse_out_select_mV" in ev:
        out += (f" | outside cliffs select {ev['V_rmse_out_select_mV']:.3g} test {ev['V_rmse_out_test_mV']:.3g} "
                f"(worst test {ev['V_rmse_out_worst_test_mV']:.3g}) | cliff shift max {ev['cliff_shift_max_pct']:.2g} %")
    return out


def build_chem_model(cfg: ChemTrainConfig, params: LiSPANParams, window_coef) -> LiSPANChemPINN:
    return LiSPANChemPINN(params, window_coef, cfg.chem_inputs, cfg.rate_min, cfg.rate_max, cfg.ramp_s,
                          cfg.window_fraction, cfg.width, cfg.depth, cfg.act, cfg.fourier_t, cfg.fourier_period,
                          tuple(cfg.short_t), cfg.ic_tau_s, cfg.quad_order, cfg.salt_sep_natural_scale)


def warm_start_from_rate(model: LiSPANChemPINN, rate_checkpoint) -> dict:
    """Copy the networks of a Phase 3a rate checkpoint into a chem model: every layer as it is, the first layer into
    the columns shared with the rate model (space, time and rate features), the chemistry columns set to zero.  At the
    nominal chemistry the chem model then reproduces the rate network exactly where the two windows agree (equality
    test); only the time window differs (window Z_CC and chemistry fit), which training absorbs.
    A chem checkpoint with the same inputs (e.g. v2 -> a wider rate range) is copied as it is; the rate feature and the
    window then refer to the new range, so the copy is a starting point, not the same function."""
    ck = torch.load(rate_checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"]
    copied = 0
    with torch.no_grad():
        for name in ("net_R", "net_xi", "net_ce", "net_phie"):
            for i, layer in enumerate(getattr(model, name).net):
                if not isinstance(layer, nn.Linear):
                    continue
                w, b = sd[f"{name}.net.{i}.weight"], sd[f"{name}.net.{i}.bias"]
                if i == 0 and w.shape == layer.weight.shape:
                    layer.weight.copy_(w.to(layer.weight.dtype))          # a chem checkpoint (same inputs): as it is
                elif i == 0:
                    if w.shape[0] != layer.weight.shape[0] or w.shape[1] != layer.weight.shape[1] - model.k:
                        raise ValueError(f"{name}: rate layer {tuple(w.shape)} does not fit chem layer {tuple(layer.weight.shape)}")
                    layer.weight.zero_(); layer.weight[:, :w.shape[1]].copy_(w.to(layer.weight.dtype))
                else:
                    if w.shape != layer.weight.shape:
                        raise ValueError(f"{name}.{i}: shapes {tuple(w.shape)} vs {tuple(layer.weight.shape)}")
                    layer.weight.copy_(w.to(layer.weight.dtype))
                layer.bias.copy_(b.to(layer.bias.dtype))
                copied += 1
    return {"layers": copied, "rate_step": ck.get("step"), "group_weights": ck.get("group_weights", {})}


def load_chem_model(checkpoint, device="cpu"):
    """(model, cfg, params) from a chem checkpoint (final.pt / best.pt / latest.pt)."""
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ChemTrainConfig.from_dict(ck["train"])
    params = params_from_dict(ck["params"])
    model = build_chem_model(cfg, params, ck["window_coef"])
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    model.load_state_dict(ck["model"])
    model.to(device=resolve_device(device), dtype=dtype).eval()
    return model, cfg, params
