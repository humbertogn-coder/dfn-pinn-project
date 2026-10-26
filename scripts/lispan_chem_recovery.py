"""Parameter-recovery test of a chemistry-conditioned Li-SPAN PINN (Phase 3b): fit FV discharge curves of known
chemistry with the network, as the two-curve tool will fit measured curves, and compare with the truth.

    python scripts/lispan_chem_recovery.py results/lispan_chem_runs/chem_v1_s2/final.pt
    python scripts/lispan_chem_recovery.py <ckpt> --rates 0.1 1 --noise-mV 0 2 --draws 5 --json recovery.json

Data: the FV evaluation references of the design (data/lispan_chem_refs/<design>/, computed at the network's contact
resistance, common U0 shift 0, w_S 1).  Fitted: the network's chemistry inputs (dU0_2, dU0_3, b, k0_1 for v1) plus the
two analytic parameters, the common U0 shift d (V + d) and the extra contact resistance dZ (V - dZ I); w_S is held at 1
(the time axis of synthetic data is exact).  Points: 100 s < t < frac x the FV cut-off (default 0.93, inside the
network's window for every chemistry of the box), every `stride`-th reference point.

The noise-free fit gives the parameter error caused by the network alone (an exact solver would return the truth);
with noise, `draws` noise realizations give the scatter, compared with the linearized standard deviation from the
network Jacobian at the truth (a Cramer-Rao-type bound for that noise; the network Jacobian stands in for the FV one).
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from dfn_pinn.lispan.chem_pinn import chem_vector, load_chem_model  # noqa: E402
from dfn_pinn.lispan.rate_pinn import ONE_C  # noqa: E402

D_BOUND, DZ_HI = 0.2, 0.05          # V; Ohm m2.  Lower bound of dZ: total Z_CC >= 0.001 Ohm m2 for the 0.010 network
                                     # (v1), 0.002 below the network's value for the Z_CC = 0 network (v2)


class CurveFit:
    """Residuals [V] of the network (inputs theta, shift d, extra resistance dZ) against curves at several rates."""

    def __init__(self, model, data):
        self.model, self.data = model, data            # data: list of (rate, t [s], V [V])
        self.k = len(model.chem_inputs)
        self.dtype = next(model.parameters()).dtype

    def curves_batch(self, X):
        """Model curves for several parameter vectors in one network call: list over X of lists over rates."""
        Ts, Zs, meta = [], [], []
        tmax = 0.0
        for x in X:
            theta = [float(v) for v in x[:self.k]]
            for rate, t, _ in self.data:
                te = self.model.window_end(rate, theta)
                tmax = max(tmax, float(t[-1] / te))
                Ts.append(torch.as_tensor(t / te, dtype=self.dtype).view(-1, 1))
                Zs.append(self.model.make_z(rate, theta, len(t)))
                meta.append(len(t))
        with torch.no_grad():
            V = self.model.voltage(torch.cat(Ts), torch.cat(Zs)).view(-1).double().numpy()
        self.last_T_max = tmax
        out, i, j = [], 0, 0
        for x in X:
            d, dz = x[self.k], x[self.k + 1]
            row = []
            for rate, t, _ in self.data:
                row.append(V[i:i + meta[j]] + d - dz * ONE_C * rate)
                i += meta[j]; j += 1
            out.append(row)
        return out

    def curves(self, x):
        return self.curves_batch([x])[0]

    def _res(self, curves):
        return np.concatenate([V - Vd for V, (_, _, Vd) in zip(curves, self.data)])

    def residuals(self, x):
        return self._res(self.curves(x))

    def jacobian(self, x, steps):
        X = []
        for j, h in enumerate(steps):
            xp, xm = x.copy(), x.copy()
            xp[j] += h; xm[j] -= h
            X += [xp, xm]
        C = self.curves_batch(X)
        J = np.empty((sum(len(t) for _, t, _ in self.data), len(x)))
        for j, h in enumerate(steps):
            J[:, j] = (self._res(C[2 * j]) - self._res(C[2 * j + 1])) / (2 * h)
        return J


def fit_case(model, spec, data, x0, lo, hi, steps):
    cf = CurveFit(model, data)
    sol = least_squares(cf.residuals, x0, jac=lambda x: cf.jacobian(x, steps), bounds=(lo, hi), x_scale=steps * 10,
                        method="trf", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
    return sol, cf


def describe(spec, x):
    """Natural units: dU0 in mV, ln-multipliers as multipliers, d in mV, dZ in Ohm cm2."""
    out = {}
    for s, v in zip(spec, x[:len(spec)]):
        out[s["name"]] = 1e3 * v if s["name"].startswith("dU0_") else float(np.exp(v))
    out["d"], out["dZ"] = 1e3 * x[len(spec)], 1e4 * x[len(spec) + 1]
    return out


def errors(spec, x, x_true):
    """dU0 / d [mV], multipliers [%], dZ [Ohm cm2]."""
    e = {}
    for s, v, w in zip(spec, x[:len(spec)], x_true[:len(spec)]):
        e[s["name"]] = 1e3 * (v - w) if s["name"].startswith("dU0_") else 100 * (np.exp(v - w) - 1)
    k = len(spec)
    e["d"], e["dZ"] = 1e3 * (x[k] - x_true[k]), 1e4 * (x[k + 1] - x_true[k + 1])
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--design", default=str(ROOT / "configs" / "lispan_chem_design_v1.json"))
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 1.0])
    ap.add_argument("--cases", nargs="*", default=None, help="chemistry ids (p0 p1 ...); default all of the design")
    ap.add_argument("--noise-mV", type=float, nargs="+", default=[0.0, 2.0])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.93)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    model, cfg, params = load_chem_model(a.checkpoint)
    spec = model.chem_inputs
    design = json.loads(Path(a.design).read_text())
    by_chem = {}
    for c in design["eval_cases"]:
        by_chem.setdefault(c["id"].split("_")[0], {})[float(c["rate"])] = c
    chems = a.cases or sorted(by_chem)
    k = len(spec)
    dz_lo = -0.009 if params.Z_CC >= 0.009 else -(params.Z_CC + 0.002)
    lo = np.array([s["lo"] for s in spec] + [-D_BOUND, dz_lo])
    hi = np.array([s["hi"] for s in spec] + [D_BOUND, DZ_HI])
    x0 = np.array([0.0 if s["name"].startswith("dU0_") else 0.5 * (s["lo"] + s["hi"]) for s in spec] + [0.02, 0.003])
    steps = np.array([2e-3 if s["name"].startswith("dU0_") else 1e-2 for s in spec] + [2e-3, 2e-4])
    rng = np.random.default_rng(a.seed)
    print(f"{a.checkpoint}: fit {[s['name'] for s in spec] + ['d', 'dZ']} to FV curves at {a.rates} C "
          f"(t < {a.frac:g} x cut-off, every {a.stride}. point); start {describe(spec, x0)}")
    results = []
    for ch in chems:
        cases = by_chem[ch]
        missing = [r for r in a.rates if r not in cases]
        if missing:
            print(f"{ch}: no reference at {missing} C - skipped"); continue
        chem = cases[a.rates[0]]["chem"]
        x_true = np.array(chem_vector(spec, chem) + [0.0, 0.0])
        clean = []
        for r in a.rates:
            z = np.load(ROOT / cases[r]["ref"])
            t, V = np.asarray(z["t"], float), np.asarray(z["V"], float)
            m = (t > 100.0) & (t < a.frac * t[-1])
            clean.append((r, t[m][::a.stride], V[m][::a.stride]))
        cf_true = CurveFit(model, clean)
        J = cf_true.jacobian(x_true, steps)
        r_true = cf_true.residuals(x_true)
        cov = np.linalg.pinv(J.T @ J)
        for noise in a.noise_mV:
            fits = []
            for dr in range(a.draws if noise > 0 else 1):
                data = [(r, t, V + (rng.normal(0, 1e-3 * noise, len(V)) if noise > 0 else 0.0)) for r, t, V in clean]
                t0 = time.time()
                sol, cf = fit_case(model, spec, data, x0.copy(), lo, hi, steps)
                fits.append({"x": sol.x.tolist(), "err": errors(spec, sol.x, x_true),
                             "misfit_mV": 1e3 * float(np.sqrt(np.mean(sol.fun ** 2))), "nfev": int(sol.nfev),
                             "T_max": cf.last_T_max, "time_s": time.time() - t0})
            E = {p: [f["err"][p] for f in fits] for p in fits[0]["err"]}
            sd_lin = {}
            for j, p in enumerate(list(E)):
                s = float(np.sqrt(cov[j, j])) * 1e-3 * noise if noise > 0 else 0.0
                sd_lin[p] = 1e3 * s if (j < k and spec[j]["name"].startswith("dU0_")) or j == k else \
                    (100 * s if j < k else 1e4 * s)
            row = {"chem": ch, "truth": describe(spec, x_true), "noise_mV": noise, "fits": fits,
                   "err_mean": {p: float(np.mean(v)) for p, v in E.items()},
                   "err_sd": {p: float(np.std(v)) for p, v in E.items()} if len(fits) > 1 else None,
                   "sd_linear": sd_lin, "misfit_at_truth_mV": 1e3 * float(np.sqrt(np.mean(r_true ** 2)))}
            results.append(row)
            em = row["err_mean"]
            print(f"{ch} noise {noise:g} mV: misfit at truth {row['misfit_at_truth_mV']:.2f} mV, fitted "
                  f"{np.mean([f['misfit_mV'] for f in fits]):.2f} mV | errors: "
                  + ", ".join(f"{p} {em[p]:+.2f}" + (f"+-{row['err_sd'][p]:.2f}" if row["err_sd"] else "")
                              + (f" (lin sd {sd_lin[p]:.2f})" if noise > 0 else "") for p in em)
                  + f" | T_max {max(f['T_max'] for f in fits):.3f}")
    units = {s["name"]: ("mV" if s["name"].startswith("dU0_") else "% of the multiplier") for s in spec}
    units.update(d="mV", dZ="Ohm cm2")
    print("units:", units)
    if a.json:
        Path(a.json).write_text(json.dumps({"checkpoint": a.checkpoint, "rates": a.rates, "units": units,
                                            "results": results}, indent=1))
        print("written", a.json)


if __name__ == "__main__":
    main()
