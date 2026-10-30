"""Two-curve tool (Phase 3b): fit measured Li-SPAN discharge curves with the chemistry-conditioned PINN.

The network (chem_pinn.LiSPANChemPINN, v2 inputs) gives the cell voltage as a function of time, model C-rate and five
chemistry inputs.  Four more parameters are analytic or a change of variables, so one network serves every cell:

    w_S   effective active-sulfur factor: Q_data = w_S Q_model.  A specific current i [A/g of the data basis] is the
          model current I = i m_S_model / w_S [A/m2], i.e. the model C-rate r = I / 10 A/m2.  The time at a given
          capacity does not depend on w_S (t = 3.6 q / i), so w_S acts only through the rate input.
    d     common shift of U0_1..3 [V] (V + d; exact for the model, docs/PHASE3_PARAMETRIC_PINN.md)
    Z_CC  contact resistance [Ohm m2] (V - Z_CC I; the network is trained at Z_CC = 0)
    theta dU0_2, dU0_3rel2 [V], ln b_12, ln b_3, ln k0_1 (the network's inputs; box CHEM_INPUTS_V2)

Data: one curve per C-rate, capacity q [mAh/g, data basis] and voltage V [V] at a constant specific current i [A/g].
A capacity q is reached at model time t with I0 tau ln cosh(t / tau) = charge (the 30 s tanh ramp of the model; the
measured current is a step, the difference is a 21 s time offset that the inverse ramp removes).

Rate range: every curve must sit inside the network's rates [r_min, r_max] = [0.1, 1] C at the fitted w_S, so
    w_S in [m i_max / (10 r_max), m i_min / (10 r_min)]     (m = m_S_model in g/m2).
The span of the measured rates must therefore be at most r_max / r_min = 10; at exactly 10 w_S is fixed.

Fit: bounded least squares (trf) on all curves (each curve weighted equally), multi-start (box centre + Latin
hypercube), central-difference Jacobian with all perturbed parameter sets in one network call.  Uncertainty:
linearized covariance sigma^2 (J^T J)^-1 with sigma from the residuals (a lower bound when the residuals are model
error rather than noise).  Network bias: optional FV self-check (FV curves at the solution, refitted with the network:
the parameter change is the error the network itself causes there; scripts/lispan_chem_recovery.py does the same for
the evaluation chemistries).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np
import torch
from scipy.optimize import least_squares

from .chem_pinn import chem_effects
from .rate_pinn import ONE_C

W_LIMITS = (0.2, 3.0)               # absolute bounds of w_S (as the Phase 2 literature fits)
D_BOUND = 0.2                       # V, |d|
Z_MAX = 0.05                        # Ohm m2 (500 Ohm cm2)
EDGE = 0.02                         # parameter within 2 % of its box width from a bound -> flag
MISFIT_WARN, MISFIT_BAD = 5.0, 15.0 # mV: above ~2x / ~6x the network error vs FV the misfit is the model's, not the net's
CORR_WARN = 0.95
POOR_SD = 0.25                      # sd > 25 % of the box width -> poorly identified

# Network error floor per trained network, in the units of the reported sd (w_S, b, k0 %; dU0, d mV; Z_CC Ohm cm2):
# rms of the noise-free parameter errors of the known-truth test (scripts/lispan_two_curve_synthetic.py, 0.2 + 0.6 C,
# the 10 evaluation chemistries without box-edge flags; docs/TWO_CURVE_TOOL.md "Synthetic test").  Added in quadrature
# to the statistical sd ("sd_total"); with it 92-100 % of the 5 mV-noise errors lie within 2 sd for every parameter
# (w_S 81 % -> 96 % in that same test).
NETWORK_SD = {
    "chem_v2_norar_s0": {"w_S": 0.14, "dU0_2": 0.64, "dU0_3rel2": 2.1, "b_12": 0.22, "b_3": 0.71, "k0_1": 11.6,
                         "d": 0.56, "Z_CC": 1.25},
    # v3 (0.08-3 C), same test at 0.2 + 2 C over the 9 unflagged chemistries (results/lispan/two_curve/
    # synthetic_v3s1_0.2+2.json): 2-8x the v2 floor; with it 88-100 % of the 5 mV-noise errors lie within 2 sd
    "chem_v3_s1": {"w_S": 0.34, "dU0_2": 5.35, "dU0_3rel2": 5.10, "b_12": 2.41, "b_3": 2.87, "k0_1": 23.85,
                   "d": 4.52, "Z_CC": 2.89},
}


def network_sd_for(checkpoint) -> dict | None:
    """Network error floor of a checkpoint (by its run folder name), or None if the network has not been tested."""
    from pathlib import Path
    return NETWORK_SD.get(Path(checkpoint).parent.name)


@dataclass
class Curve:
    """One measured discharge at constant specific current."""
    i_sp: float                     # A/g of the data basis
    q: np.ndarray                   # mAh/g of the data basis
    V: np.ndarray                   # V
    label: str = ""

    def __post_init__(self):
        self.q, self.V = np.asarray(self.q, float), np.asarray(self.V, float)
        if self.q.shape != self.V.shape or self.q.ndim != 1:
            raise ValueError("q and V must be 1-d arrays of the same length")


def m_S_g(params) -> float:
    """Model sulfur loading [g/m2]."""
    return params.m_S_model * 1e3


def model_rate(params, i_sp, w_S):
    """Model C-rate of a specific current i_sp [A/g] at the active-sulfur factor w_S."""
    return i_sp * m_S_g(params) / (w_S * ONE_C)


def model_time(q, i_sp, ramp_s=30.0):
    """Model time [s] at which the capacity q [mAh/g, data basis] has passed at i_sp [A/g] (inverse of the tanh-ramp
    charge I0 tau ln cosh(t/tau)); independent of w_S."""
    t_data = 3.6 * np.asarray(q, float) / i_sp
    if ramp_s <= 0:
        return t_data
    y = t_data / ramp_s
    return ramp_s * (y + np.log1p(np.sqrt(-np.expm1(-2.0 * y))))


def w_bounds(params, curves, rate_min, rate_max, limits=W_LIMITS):
    """(lo, hi) of w_S that keep every curve inside the network's rate range."""
    i = [c.i_sp for c in curves]
    lo = max(limits[0], m_S_g(params) * max(i) / (ONE_C * rate_max))
    hi = min(limits[1], m_S_g(params) * min(i) / (ONE_C * rate_min))
    return lo, hi


class TwoCurveFit:
    """Least-squares fit of curves with the network.  Parameter order: w_S, the chemistry inputs, d, Z_CC; internally
    x = [ln w_S, theta..., d, Z_CC]."""

    def __init__(self, model, curves, fixed: dict | None = None, w_limits=W_LIMITS, network_sd: dict | None = None):
        self.model, self.curves = model, list(curves)
        self.network_sd = network_sd
        self.p = model.p
        self.spec = model.chem_inputs
        self.k = len(self.spec)
        self.names = ["w_S"] + [s["name"] for s in self.spec] + ["d", "Z_CC"]
        self.dtype = next(model.parameters()).dtype
        w_lo, w_hi = w_bounds(self.p, self.curves, model.rate_min, model.rate_max, w_limits)
        if w_lo > w_hi * (1 + 1e-9):
            span = max(c.i_sp for c in self.curves) / min(c.i_sp for c in self.curves)
            raise ValueError(f"the currents span x{span:.3g}, more than the network's rate range "
                             f"x{model.rate_max / model.rate_min:.3g} (or w_S would leave {w_limits})")
        self.lo = np.array([math.log(w_lo)] + [s["lo"] for s in self.spec] + [-D_BOUND, 0.0])
        self.hi = np.array([math.log(max(w_hi, w_lo))] + [s["hi"] for s in self.spec] + [D_BOUND, Z_MAX])
        self.fixed = {}
        for n, v in (fixed or {}).items():
            if n not in self.names:
                raise ValueError(f"unknown parameter {n!r} (parameters: {self.names})")
            self.fixed[self.names.index(n)] = self.to_internal(n, v)
        if self.hi[0] - self.lo[0] < 1e-9 and 0 not in self.fixed:
            self.fixed[0] = self.lo[0]                     # the rates span exactly the range: w_S has one value
            self.w_fixed_by_range = True
        else:
            self.w_fixed_by_range = False
        for j, v in self.fixed.items():
            if not self.lo[j] - 1e-9 <= v <= self.hi[j] + 1e-9:
                raise ValueError(f"fixed {self.names[j]} = {self.to_natural(j, v):g} outside "
                                 f"[{self.to_natural(j, self.lo[j]):g}, {self.to_natural(j, self.hi[j]):g}]")
        self.free = [j for j in range(len(self.names)) if j not in self.fixed]
        self.t = [model_time(c.q, c.i_sp, model.ramp_s) for c in self.curves]
        n = np.array([len(c.q) for c in self.curves], float)
        self.weights = np.sqrt(n.mean() / n)              # every curve counts equally
        self.steps = np.array([2e-3] + [2e-3 if s["name"].startswith("dU0_") else 1e-2 for s in self.spec]
                              + [2e-3, 2e-4])
        self.last_T_max = None

    # ------------------------------------------------------------------ units
    def to_internal(self, name, v):
        if name == "w_S":
            return math.log(v)
        if name in ("d", "Z_CC") or name.startswith("dU0_"):
            return float(v)
        return math.log(v)                                  # b_12, b_3, k0_1 multipliers

    def to_natural(self, j, v):
        name = self.names[j]
        return float(np.exp(v)) if name == "w_S" or not (name in ("d", "Z_CC") or name.startswith("dU0_")) \
            else float(v)

    def natural(self, x):
        return {n: self.to_natural(j, v) for j, (n, v) in enumerate(zip(self.names, x))}

    def full(self, xf):
        x = np.empty(len(self.names))
        x[self.free] = xf
        for j, v in self.fixed.items():
            x[j] = v
        return x

    # ------------------------------------------------------------------ model curves
    def rates(self, x):
        w = math.exp(x[0])
        return [model_rate(self.p, c.i_sp, w) for c in self.curves]

    def curves_batch(self, X, clip_rate=True):
        """Model voltages for several full parameter vectors in one network call: list over X of lists over curves.
        Also records T = t / t_end per curve (self.last_T)."""
        Ts, Zs, meta, Tm = [], [], [], []
        for x in X:
            theta = [float(v) for v in x[1:1 + self.k]]
            row = []
            for r, t in zip(self.rates(x), self.t):
                if clip_rate:                                  # FD steps may poke out of the range by ~1e-3
                    r = min(max(r, self.model.rate_min), self.model.rate_max)
                te = self.model.window_end(r, theta)
                Ts.append(torch.as_tensor(t / te, dtype=self.dtype).view(-1, 1))
                Zs.append(self.model.make_z(r, theta, len(t)))
                meta.append((r, len(t)))
                row.append(float(t[-1] / te))
            Tm.append(row)
        with torch.no_grad():
            V = self.model.voltage(torch.cat(Ts), torch.cat(Zs), U0_shift=0.0, Z_CC=0.0).view(-1).double().numpy()
        out, i, j = [], 0, 0
        for x in X:
            d, Z = x[1 + self.k], x[2 + self.k]
            row = []
            for t in self.t:
                r, n = meta[j]
                I = ONE_C * r * np.tanh(t / self.model.ramp_s) if self.model.ramp_s > 0 else ONE_C * r
                row.append(V[i:i + n] + d - Z * I)
                i += n; j += 1
            out.append(row)
        self.last_T = Tm[0]
        return out

    def model_curves(self, x):
        return self.curves_batch([x])[0]

    def _res(self, Vm):
        return np.concatenate([w * (v - c.V) for w, v, c in zip(self.weights, Vm, self.curves)])

    def residuals(self, xf):
        return self._res(self.model_curves(self.full(xf)))

    def jacobian(self, xf):
        x0 = self.full(xf)
        X = []
        for j in self.free:
            h = self.steps[j]
            xp, xm = x0.copy(), x0.copy()
            xp[j] = min(x0[j] + h, self.hi[j]); xm[j] = max(x0[j] - h, self.lo[j])
            if xp[j] - xm[j] < 1e-12:                       # fixed range of zero width
                xp[j], xm[j] = x0[j] + h, x0[j] - h
            X += [xp, xm]
        C = self.curves_batch(X)
        J = np.empty((sum(len(c.q) for c in self.curves), len(self.free)))
        for col, j in enumerate(self.free):
            J[:, col] = (self._res(C[2 * col]) - self._res(C[2 * col + 1])) / (X[2 * col][j] - X[2 * col + 1][j])
        return J

    # ------------------------------------------------------------------ fit
    def starts(self, n, seed=0):
        """Box centre (chemistry 0 = Simanjuntak where 0 is inside, w_S mid-range, d 0, Z small) + Latin hypercube."""
        lo, hi = self.lo[self.free], self.hi[self.free]
        centre = []
        for j in self.free:
            name = self.names[j]
            if name == "Z_CC":
                centre.append(min(0.002, self.hi[j]))
            elif name == "d":
                centre.append(0.0)
            elif name == "w_S":
                centre.append(0.5 * (self.lo[j] + self.hi[j]))
            else:
                centre.append(min(max(0.0, self.lo[j]), self.hi[j]))
        out = [np.array(centre)]
        if n > 1:
            rng = np.random.default_rng(seed)
            u = (np.argsort(rng.random((n - 1, len(lo))), axis=0) + rng.random((n - 1, len(lo)))) / (n - 1)
            out += list(lo + (0.05 + 0.9 * u) * (hi - lo))
        return out

    def fit_from(self, xf0, max_nfev=200):
        xf0 = np.clip(xf0, self.lo[self.free] + 1e-9, self.hi[self.free] - 1e-9)
        lo, hi = self.lo[self.free], self.hi[self.free]
        same = hi - lo < 1e-12
        lo, hi = np.where(same, lo - 1e-9, lo), np.where(same, hi + 1e-9, hi)
        return least_squares(self.residuals, xf0, jac=self.jacobian, bounds=(lo, hi), method="trf",
                             x_scale=self.steps[self.free] * 10, ftol=1e-10, xtol=1e-10, gtol=1e-10,
                             max_nfev=max_nfev)

    def fit(self, n_starts=8, seed=0, max_nfev=200, log=None):
        sols = []
        for s, x0 in enumerate(self.starts(n_starts, seed)):
            sol = self.fit_from(x0, max_nfev)
            sols.append(sol)
            if log:
                log(f"  start {s}: rms {1e3 * np.sqrt(np.mean(sol.fun ** 2)):.2f} mV, nfev {sol.nfev}, "
                    + ", ".join(f"{n} {v:.4g}" for n, v in self.natural(self.full(sol.x)).items()))
        sols.sort(key=lambda s: s.cost)
        return sols

    # ------------------------------------------------------------------ report
    def uncertainty(self, xf):
        """Linearized covariance (internal units, free parameters) with sigma^2 = SSR / (n - p)."""
        r = self.residuals(xf)
        J = self.jacobian(xf)
        dof = max(len(r) - len(xf), 1)
        s2 = float(r @ r) / dof
        cov = s2 * np.linalg.pinv(J.T @ J)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = cov / np.outer(np.where(sd > 0, sd, 1), np.where(sd > 0, sd, 1))
        return cov, sd, corr, math.sqrt(s2)

    def sd_scale(self, j):
        """Factor from internal units to the sd units (% for log parameters, mV, Ohm cm2)."""
        n = self.names[j]
        return 1e3 if (n == "d" or n.startswith("dU0_")) else (1e4 if n == "Z_CC" else 100.0)

    def natural_sd(self, x, sd_free):
        """sd in natural units: w_S, b, k0 as % of the value (log sd x 100), dU0 / d in mV, Z_CC in Ohm cm2."""
        return {self.names[j]: self.sd_scale(j) * s for j, s in zip(self.free, sd_free)}

    def report(self, sols, sd_rel_unique=3.0):
        best = sols[0]
        xf = best.x
        x = self.full(xf)
        Vm = self.model_curves(x)
        T_max = list(self.last_T)
        cov, sd, corr, sigma = self.uncertainty(xf)
        sd_nat = self.natural_sd(x, sd)
        net = self.network_sd or {}
        sd_total = {n: math.hypot(v, net.get(n, 0.0)) for n, v in sd_nat.items()}
        sd_tot_int = np.array([sd_total[self.names[j]] / self.sd_scale(j) for j in self.free])
        flags = []
        # box edges (theta, w_S at the rate-range bounds, d)
        for col, j in enumerate(self.free):
            n, width = self.names[j], self.hi[j] - self.lo[j]
            if width <= 0:
                continue
            at_lo, at_hi = x[j] - self.lo[j] < EDGE * width, self.hi[j] - x[j] < EDGE * width
            if n == "Z_CC":
                if at_lo:
                    flags.append({"flag": "Z_CC_zero", "level": "info", "msg": "contact resistance at its lower "
                                  "bound 0 (the data ask for no extra ohmic drop)"})
                elif at_hi:
                    flags.append({"flag": "box_edge", "level": "warn", "param": n, "msg": f"Z_CC at {Z_MAX} Ohm m2"})
            elif n == "w_S" and (at_lo or at_hi):
                which = "fastest" if at_lo else "slowest"
                flags.append({"flag": "rate_edge", "level": "warn", "param": n,
                              "msg": f"w_S at the bound where the {which} curve sits at the network's rate limit; the "
                                     f"optimum may lie outside the rate range"})
            elif at_lo or at_hi:
                flags.append({"flag": "box_edge", "level": "warn", "param": n,
                              "msg": f"{n} at the {'lower' if at_lo else 'upper'} edge of the trained box: the optimum "
                                     f"may lie outside it, and the network error is larger near the box corners"})
            if width > 0 and sd_tot_int[col] > POOR_SD * width and n != "Z_CC":
                flags.append({"flag": "poorly_identified", "level": "warn", "param": n,
                              "msg": f"sd of {n} is {100 * sd_tot_int[col] / width:.0f} % of the box width"})
        rms = [1e3 * float(np.sqrt(np.mean((v - c.V) ** 2))) for v, c in zip(Vm, self.curves)]
        rms_all = 1e3 * float(np.sqrt(np.mean(best.fun ** 2)))
        if rms_all > MISFIT_WARN:
            flags.append({"flag": "misfit", "level": "warn" if rms_all <= MISFIT_BAD else "bad",
                          "msg": f"rms misfit {rms_all:.1f} mV, above the network's own error (~1-2.5 mV vs FV): the "
                                 f"continuum model does not reproduce the data, the parameters are effective values"})
        beyond = [(c.label or f"curve {i}", tm) for i, (c, tm) in enumerate(zip(self.curves, T_max)) if tm > 1.0]
        for lab, tm in beyond:
            flags.append({"flag": "beyond_window", "level": "warn", "msg": f"{lab}: data reach {tm:.3f} x the network "
                          f"window (extrapolation in time); shorten the data (--frac) or treat the end as unreliable"})
        for a in range(len(self.free)):
            for b in range(a + 1, len(self.free)):
                if abs(corr[a, b]) > CORR_WARN:
                    flags.append({"flag": "correlated", "level": "info",
                                  "params": [self.names[self.free[a]], self.names[self.free[b]]],
                                  "msg": f"corr({self.names[self.free[a]]}, {self.names[self.free[b]]}) = {corr[a, b]:+.3f}"})
        # other local minima that fit as well (cost difference inside the 95 % chi-square region of the linearized
        # model with sigma from the residuals): far apart in sd units means the linearized sd understates the spread
        from scipy.stats import chi2
        dcost = 0.5 * chi2.ppf(0.95, len(xf)) * sigma ** 2            # least_squares cost = SSR / 2
        alt, accepted = [], [x]
        for s in sols[1:]:
            if s.cost - best.cost <= dcost:
                z = np.abs(s.x - xf) / np.where(sd > 0, sd, np.inf)
                accepted.append(self.full(s.x))
                if np.max(z) > sd_rel_unique:
                    alt.append({"rms_mV": 1e3 * float(np.sqrt(np.mean(s.fun ** 2))),
                                "params": self.natural(self.full(s.x)), "max_shift_sd": float(np.max(z)),
                                "shifted": [self.names[self.free[i]] for i in np.nonzero(z > 2.0)[0]]})
        if alt:
            shifted = sorted({n for a_ in alt for n in a_["shifted"]})
            flags.append({"flag": "non_unique", "level": "warn", "params": shifted,
                          "msg": f"{len(alt)} other start(s) fit as well (inside the 95 % chi-square region) but sit "
                                 f"> {sd_rel_unique:g} sd away: the linearized sd understates the uncertainty of "
                                 f"{', '.join(shifted)}; see 'spread'"})
        A = np.array(accepted)
        spread = {self.names[j]: [self.to_natural(j, A[:, j].min()), self.to_natural(j, A[:, j].max())]
                  for j in self.free}
        if self.w_fixed_by_range:
            flags.append({"flag": "w_S_fixed_by_range", "level": "info", "msg": "the currents span exactly the network's "
                          "rate range, so w_S is fixed at the value that maps them onto it"})
        return {
            "params": self.natural(x),
            "sd": sd_nat,
            "sd_network": {n: net.get(n, 0.0) for n in sd_nat} if self.network_sd else None,
            "sd_total": sd_total,
            "sd_units": "w_S, b, k0: % of the value; dU0, d: mV; Z_CC: Ohm cm2.  sd: statistical (linearized, sigma "
                        "from the residuals); sd_total: with the network error floor in quadrature",
            "fixed": [self.names[j] for j in self.fixed],
            "corr": {"names": [self.names[j] for j in self.free], "matrix": corr.round(4).tolist()},
            "sigma_mV": 1e3 * sigma,
            "rms_mV": rms_all, "rms_per_curve_mV": rms, "T_max": T_max, "model_rates": self.rates(x),
            "U0": [u + dd + x[1 + self.k] for u, dd in zip(self.p.U0, chem_effects(self.spec, list(x[1:1 + self.k]))[0])],
            "b": [bb * math.exp(lb) for bb, lb in zip(self.p.b, chem_effects(self.spec, list(x[1:1 + self.k]))[1])],
            "k0_1_mult": math.exp(chem_effects(self.spec, list(x[1:1 + self.k]))[2][0]),
            "flags": flags, "alternatives": alt, "spread": spread, "n_equivalent_starts": len(accepted),
            "starts": [{"rms_mV": 1e3 * float(np.sqrt(np.mean(s.fun ** 2))), "nfev": int(s.nfev),
                        "params": self.natural(self.full(s.x))} for s in sols],
            "x": x.tolist(),
        }

    # ------------------------------------------------------------------ prediction
    def predict(self, x, i_sp, q):
        """Network curve at another specific current (must map inside the rate range at the fitted w_S)."""
        r = model_rate(self.p, i_sp, math.exp(x[0]))
        if not self.model.rate_min * (1 - 1e-9) <= r <= self.model.rate_max * (1 + 1e-9):
            raise ValueError(f"{i_sp:g} A/g is {r:.3g} C in the model at w_S = {math.exp(x[0]):.3g}, outside the "
                             f"network's [{self.model.rate_min:g}, {self.model.rate_max:g}] C")
        other = TwoCurveFit.__new__(TwoCurveFit)
        other.__dict__.update(self.__dict__)
        other.curves = [Curve(i_sp, q, np.zeros(len(q)))]
        other.t = [model_time(q, i_sp, self.model.ramp_s)]
        V = other.curves_batch([np.asarray(x, float)], clip_rate=False)[0][0]
        return V, other.last_T[0]


# ====================================================================== finite-volume counterpart
def fv_params(net_params, spec, x):
    """LiSPANParams of the FV solver for a full parameter vector x (as TwoCurveFit): chemistry inputs applied, U0
    shifted by d, Z_CC set, no double layer (c_DL -> 0, as the network's references)."""
    from .chem_pinn import apply_chem
    k = len(spec)
    p = apply_chem(replace(net_params, c_DL=1e-6, Z_CC=float(x[2 + k])), spec, [float(v) for v in x[1:1 + k]])
    return replace(p, U0=tuple(u + float(x[1 + k]) for u in p.U0))


def fv_curve(net_params, spec, x, i_sp, q, N=(20, 10), rtol=1e-6, ramp_s=30.0):
    """FV voltage at the capacities q [mAh/g, data basis] for specific current i_sp [A/g]; points beyond the FV cut-off
    are NaN.  Returns (V, Q_end in the data basis) or (None, None) on solver failure."""
    from .model import solve
    from .params import LiSPANProtocol
    p = fv_params(net_params, spec, x)
    w = math.exp(x[0])
    prot = LiSPANProtocol(current=i_sp * m_S_g(p) / w, ramp_s=ramp_s)
    for tol in (rtol, 10 * rtol):
        try:
            _, res = solve(p, prot, N_c=N[0], N_s=N[1], n_out=1200, rtol=tol)
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        Q, V = np.asarray(res["Q_mAh_gS"]) * w, np.asarray(res["V"])
        q = np.asarray(q, float)
        return np.where(q <= Q[-1], np.interp(q, Q, V), np.nan), float(Q[-1])
    return None, None


def fv_check(fit: TwoCurveFit, x, N=(20, 10), refit=True, log=None):
    """FV curves at the solution x: misfit to the data and to the network; with refit, the network fit of those FV curves
    (started at x) gives the parameter error the network causes at this solution (bias = refit - x)."""
    out = {"curves": []}
    fv = []
    for c in fit.curves:
        V, Q_end = fv_curve(fit.p, fit.spec, x, c.i_sp, c.q, N=N, ramp_s=fit.model.ramp_s)
        fv.append(V)
        if V is None:
            out["curves"].append({"label": c.label, "failed": True})
            continue
        ok = np.isfinite(V)
        out["curves"].append({"label": c.label, "Q_end_fv": Q_end, "n_beyond_fv_cutoff": int((~ok).sum()),
                              "rms_fv_vs_data_mV": 1e3 * float(np.sqrt(np.mean((V[ok] - c.V[ok]) ** 2)))})
    if any(v is None for v in fv):
        return out
    Vn = fit.model_curves(x)
    for row, V, vn in zip(out["curves"], fv, Vn):
        ok = np.isfinite(V)
        row["rms_net_vs_fv_mV"] = 1e3 * float(np.sqrt(np.mean((vn[ok] - V[ok]) ** 2)))
    if refit:
        keep = [np.isfinite(V) for V in fv]
        synth = [Curve(c.i_sp, c.q[m], V[m], c.label) for c, V, m in zip(fit.curves, fv, keep)]
        f2 = TwoCurveFit(fit.model, synth, fixed={fit.names[j]: fit.to_natural(j, v) for j, v in fit.fixed.items()})
        f2.lo, f2.hi = fit.lo, fit.hi
        sol = f2.fit_from(np.asarray(x)[f2.free])
        x2 = f2.full(sol.x)
        bias = {}
        for j in fit.free:
            n = fit.names[j]
            dv = x2[j] - x[j]
            bias[n] = 1e3 * dv if (n == "d" or n.startswith("dU0_")) else (1e4 * dv if n == "Z_CC" else 100 * dv)
        out["network_bias"] = bias
        out["network_bias_units"] = "as sd (w_S, b, k0: %; dU0, d: mV; Z_CC: Ohm cm2)"
        out["refit_rms_mV"] = 1e3 * float(np.sqrt(np.mean(sol.fun ** 2)))
        if log:
            log("  network bias at the solution: " + ", ".join(f"{n} {v:+.2f}" for n, v in bias.items()))
    return out
