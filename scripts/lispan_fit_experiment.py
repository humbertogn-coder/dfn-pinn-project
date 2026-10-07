"""Classical least-squares fit of the Li-SPAN finite-volume model to measured discharge points (reference answer
for the PINN inverse on the same data).

    python scripts/lispan_fit_experiment.py --fit 0.1 1 --predict 0.05 0.2

Parameters: {k0_1, k0_2, k0_3, b_1, b_2, b_3, U0_1, Z_CC} (multipliers, U0_1 offset in V) on the PINN-benchmark
model (reaction 1 reversible, 2 and 3 irreversible, no double layer, 30 s ramp); data files from
scripts/lispan_make_inverse_data.py --experiment (results/lispan/inverse_data/V_<rate>C_experiment.npz).
scipy least_squares (trust region, forward differences of step 0.02 in log / 2 mV for U0_1).
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dfn_pinn.lispan.params import LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.model import solve  # noqa: E402
from lispan_make_inverse_data import apply_truth  # noqa: E402

NAMES = ["k0_1", "k0_2", "k0_3", "b_1", "b_2", "b_3", "U0_1", "Z_CC"]
U0_SCALE = 0.1   # same convention as the PINN (offset = U0_SCALE * x)


def values(x):
    return {n: (U0_SCALE * v if n.startswith("U0") else float(np.exp(v))) for n, v in zip(NAMES, x)}


def model_voltage(x, crate, t, base, N_c=20, N_s=10):
    p = apply_truth(base, values(x))
    prot = LiSPANProtocol.from_crate(crate, ramp_s=30.0)
    _, res = solve(p, prot, N_c=N_c, N_s=N_s, n_out=1500, t_end=float(t[-1]) * 1.05 + 100.0)
    tt, VV = res["t"], res["V"]
    # past the model's cut-off the cell is empty: hold V_min (a large residual, as it should be)
    return np.where(t <= tt[-1], np.interp(t, tt, VV), prot.V_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", type=float, default=[0.1, 1.0])
    ap.add_argument("--predict", nargs="*", type=float, default=[0.05, 0.2])
    ap.add_argument("--max-nfev", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "results" / "lispan" / "fit_experiment"))
    ap.add_argument("--init-json", default=None, help="start from the estimates of an earlier fit (its JSON report)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)
    data = {}
    for cr in set(args.fit) | set(args.predict):
        d = np.load(ROOT / "results" / "lispan" / "inverse_data" / f"V_{cr:g}C_experiment.npz")
        data[cr] = (np.asarray(d["t"]), np.asarray(d["V"]), np.asarray(d["Q_mAh_gS"]))
    t0 = time.time()
    n_eval = [0]

    def resid(x):
        n_eval[0] += 1
        r = np.concatenate([model_voltage(x, cr, data[cr][0], base) - data[cr][1] for cr in args.fit]) * 1e3
        print(f"  eval {n_eval[0]:3d} ({time.time() - t0:5.0f} s): rms {np.sqrt(np.mean(r ** 2)):7.2f} mV | "
              + ", ".join(f"{k} {v:.4g}" for k, v in values(x).items()), flush=True)
        return r

    x0 = np.zeros(len(NAMES))
    x_start = x0.copy()
    if args.init_json:
        est0 = json.loads(Path(args.init_json).read_text())["estimates"]
        x_start = np.array([est0[n] / U0_SCALE if n.startswith("U0") else np.log(est0[n]) for n in NAMES])
    r0 = resid(x0)
    sol = least_squares(resid, x_start, method="trf", diff_step=0.02, x_scale=1.0, max_nfev=args.max_nfev,
                        bounds=(-2.0 * np.ones(len(NAMES)), 2.0 * np.ones(len(NAMES))))
    J = sol.jac
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    # parameter uncertainty from the residual scatter (model error + digitization treated as white noise)
    dof = max(len(sol.fun) - len(NAMES), 1)
    s2 = float(np.sum(sol.fun ** 2) / dof)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        sd = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        sd = np.full(len(NAMES), np.nan)
    est = values(sol.x)
    report = {"fit_rates": args.fit, "estimates": est,
              "sd_log_or_U0scaled": dict(zip(NAMES, sd.tolist())),
              "rms_initial_mV": float(np.sqrt(np.mean(r0 ** 2))), "rms_fit_mV": rms, "nfev": int(sol.nfev),
              "per_rate_rms_mV": {}, "wall_s": time.time() - t0}
    for cr in sorted(data):
        r = (model_voltage(sol.x, cr, data[cr][0], base) - data[cr][1]) * 1e3
        r_nom = (model_voltage(x0, cr, data[cr][0], base) - data[cr][1]) * 1e3
        report["per_rate_rms_mV"][f"{cr:g}C"] = {"nominal": float(np.sqrt(np.mean(r_nom ** 2))),
                                                "fitted": float(np.sqrt(np.mean(r ** 2))),
                                                "role": "fit" if cr in args.fit else "prediction"}
    report["start"] = values(x_start)
    (out / f"fit_{'+'.join(f'{c:g}' for c in args.fit)}C{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
