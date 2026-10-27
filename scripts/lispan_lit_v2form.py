"""Validation of the two-curve tool: finite-volume fit of the same curves, with the same parameters and bounds as the
network fit (scripts/lispan_two_curve_fit.py; w_S, dU0_2, dU0_3rel2, b_12, b_3, k0_1, d, Z_CC), started at the
network's solution.  The FV optimum is what an exact surrogate would return; the difference is the error of the tool.

    python scripts/lispan_two_curve_fit.py --dataset hao2025_SPAN --rates 0.2 1 --fv-check     # network fit first
    python scripts/lispan_lit_v2form.py --dataset hao2025_SPAN --rates 0.2 1 --predict 0.5
    python scripts/lispan_lit_v2form.py --experiment 0.1 1 --predict 0.2

Same data points and curve weights as the tool (its loaders are imported), the same w_S bounds (the network's rate
range) and box.  FV grid 10 / 5 volumes and rtol 1e-5 during the fit (as scripts/lispan_lit_rates.py), 20 / 10 for the
reported curves; a model that reaches 1.0 V before a data point counts as 1.0 V there.  Forward-difference Jacobian
with absolute steps (src/dfn_pinn/lispan/fdjac.py); the best point is checkpointed (--resume continues a killed fit).
Output: results/lispan/two_curve/<name>_fv.json with the FV estimates and their difference from the network fit in
units of the network fit's linearized sd.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from dfn_pinn.lispan.chem_pinn import load_chem_model  # noqa: E402
from dfn_pinn.lispan.fdjac import FitCheckpoint, least_squares_fd  # noqa: E402
from dfn_pinn.lispan.two_curve import TwoCurveFit, fv_curve  # noqa: E402
import lispan_two_curve_fit as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset")
    src.add_argument("--experiment", nargs="+", type=float)
    ap.add_argument("--rates", nargs="+", type=float)
    ap.add_argument("--predict", nargs="*", type=float, default=[])
    ap.add_argument("--network", default=str(T.NETWORK))
    ap.add_argument("--name", default=None)
    ap.add_argument("--start-json", default=None, help="network fit to start from (default results/lispan/two_curve/"
                    "<name>.json); also supplies frac, n-points and fixed parameters")
    ap.add_argument("--fit-grid", nargs=2, type=int, default=[10, 5])
    ap.add_argument("--fit-rtol", type=float, default=1e-5)
    ap.add_argument("--max-nfev", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    model, cfg, params = load_chem_model(a.network)
    name = a.name or (f"{a.dataset}_{'+'.join(f'{r:g}' for r in a.rates)}" if a.dataset else
                      f"simanjuntak_{'+'.join(f'{r:g}' for r in a.experiment)}")
    start = json.loads(Path(a.start_json or T.OUT / f"{name}.json").read_text())
    frac, n_pts = start["args"]["frac"], start["args"]["n_points"]
    if a.dataset:
        curves = [T.dataset_curve(a.dataset, r, frac, n_pts)[0] for r in a.rates]
    else:
        curves = [T.experiment_curve(r, frac, params)[0] for r in a.experiment]
    fit = TwoCurveFit(model, curves, fixed=T.parse_fix(start["args"]["fix"]))
    x_net = np.asarray(start["x"], float)
    free = fit.free
    lo, hi = fit.lo[free], fit.hi[free]
    ck = FitCheckpoint(T.OUT / f"{name}_fv.ckpt.json")
    xf0 = ck.x() if (a.resume and ck.state) else x_net[free]
    xf0 = np.clip(xf0, lo + 1e-9, hi - 1e-9)
    steps = np.array([5e-3 if fit.names[j] == "w_S" else 2e-3 if (fit.names[j] == "d" or fit.names[j].startswith("dU0_"))
                      else 2e-4 if fit.names[j] == "Z_CC" else 1e-2 for j in free])
    n_eval, t0 = [ck.prior_solves], time.time()
    print(f"{name}: FV fit of {[fit.names[j] for j in free]} from the network solution "
          + ", ".join(f"{k} {v:.4g}" for k, v in fit.natural(fit.full(xf0)).items()), flush=True)

    def resid(xf):
        n_eval[0] += 1
        x = fit.full(xf)
        rr = []
        for c, w in zip(fit.curves, fit.weights):
            V, _ = fv_curve(fit.p, fit.spec, x, c.i_sp, c.q, N=tuple(a.fit_grid), rtol=a.fit_rtol,
                            ramp_s=model.ramp_s)
            V = np.full(len(c.q), 1.0) if V is None else np.where(np.isfinite(V), V, 1.0)
            rr.append(w * (V - c.V))
        r = np.concatenate(rr)
        rms = 1e3 * float(np.sqrt(np.mean(r ** 2)))
        ck.record(xf, rms, n_eval[0], ck.prior_wall_s + time.time() - t0)
        print(f"  eval {n_eval[0]:3d} ({time.time() - t0:5.0f} s): rms {rms:7.2f} mV | "
              + ", ".join(f"{k} {v:.4g}" for k, v in fit.natural(x).items()), flush=True)
        return r

    fun, jac = least_squares_fd(resid, steps, lo, hi)
    sol = least_squares(fun, xf0, jac=jac, method="trf", x_scale=steps * 5, max_nfev=a.max_nfev, bounds=(lo, hi),
                        ftol=1e-4, xtol=1e-4)
    x_fv = fit.full(sol.x)

    # reported curves on the fine grid; network at both solutions
    per = []
    Vn_fv = fit.model_curves(x_fv)
    Vn_net = fit.model_curves(x_net)
    for c, vn, vnn in zip(fit.curves, Vn_fv, Vn_net):
        Vf, Q_end = fv_curve(fit.p, fit.spec, x_fv, c.i_sp, c.q, ramp_s=model.ramp_s)
        Vg, _ = fv_curve(fit.p, fit.spec, x_net, c.i_sp, c.q, ramp_s=model.ramp_s)
        okf, okg = np.isfinite(Vf), np.isfinite(Vg)
        per.append({"label": c.label, "rms_fv_at_fv_mV": 1e3 * float(np.sqrt(np.mean((np.where(okf, Vf, 1.0) - c.V) ** 2))),
                    "rms_fv_at_net_mV": 1e3 * float(np.sqrt(np.mean((np.where(okg, Vg, 1.0) - c.V) ** 2))),
                    "rms_net_at_net_mV": 1e3 * float(np.sqrt(np.mean((vnn - c.V) ** 2))),
                    "rms_net_vs_fv_at_fv_mV": 1e3 * float(np.sqrt(np.mean((vn[okf] - Vf[okf]) ** 2)))})
    preds = []
    for r in a.predict:
        c = (T.dataset_curve(a.dataset, r, frac, n_pts) if a.dataset else T.experiment_curve(r, frac, params))[0]
        row = {"label": c.label}
        for tag, x in (("fv", x_fv), ("net", x_net)):
            V, _ = fv_curve(fit.p, fit.spec, x, c.i_sp, c.q, ramp_s=model.ramp_s)
            if V is not None:
                row[f"rms_fv_at_{tag}_mV"] = 1e3 * float(np.sqrt(np.mean((np.where(np.isfinite(V), V, 1.0) - c.V) ** 2)))
        try:
            V, T_max = fit.predict(x_net, c.i_sp, c.q)
            row["rms_net_at_net_mV"] = 1e3 * float(np.sqrt(np.mean((V - c.V) ** 2)))
        except ValueError as exc:
            row["net_error"] = str(exc)
        preds.append(row)
    p_fv, p_net = fit.natural(x_fv), fit.natural(x_net)
    sd = start["sd"]
    diff = {}
    for j in free:
        n = fit.names[j]
        dv = x_fv[j] - x_net[j]
        dn = 1e3 * dv if (n == "d" or n.startswith("dU0_")) else (1e4 * dv if n == "Z_CC" else 100 * dv)
        diff[n] = {"net_minus_fv": -dn, "in_sd": (-dn / sd[n]) if sd.get(n) else None}
    span = fit.hi - fit.lo
    at_bound = [fit.names[j] for j in free if span[j] > 0 and min(x_fv[j] - fit.lo[j], fit.hi[j] - x_fv[j]) < 0.02 * span[j]]
    rep = {"name": name, "fv_params": p_fv, "net_params": p_net, "difference": diff,
           "difference_units": "net - FV; w_S, b, k0: %; dU0, d: mV; Z_CC: Ohm cm2; in_sd: / the network fit's sd",
           "rms_fit_mV_coarse": 1e3 * float(np.sqrt(np.mean(sol.fun ** 2))), "per_curve": per, "predictions": preds,
           "at_bound": at_bound, "n_residual_evals": n_eval[0], "wall_s": ck.prior_wall_s + time.time() - t0,
           "status": int(sol.status), "x_fv": x_fv.tolist()}
    out = T.OUT / f"{name}_fv.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("fv_params", "difference", "per_curve", "predictions", "at_bound")}, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
