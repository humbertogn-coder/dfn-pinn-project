"""Finite-volume reference for the SPAN500 PINN inverse: least squares on exactly the PINN's data window with the
PINN's fixed sulfur fraction (w_S, hence the same current), fitting the same six parameters (U0_1..3, b_1..3).

The PINN inverse (configs/lispan_inverse_span500_paper_wS.json) cannot fit w_S (its current is fixed) and its data
window stops before the cut-off collapse (Q <= 0.95 x the FV end of discharge), so the full-curve FV fit of
scripts/lispan_span500.py is not the like-for-like reference; this one is.

    python scripts/lispan_span500_window_fit.py   ->  results/lispan/span500/fit_cycle3_pinnwindow.json
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dfn_pinn.lispan.fdjac import FitCheckpoint, least_squares_fd  # noqa: E402
import lispan_span500 as sp5  # noqa: E402

NAMES6 = ["U0_1", "U0_2", "U0_3", "b_1", "b_2", "b_3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lispan_inverse_span500_paper_wS.json"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-nfev", type=int, default=30)
    args = ap.parse_args()
    spec = json.loads(Path(args.config).read_text())
    d = np.load(ROOT / spec["rates"][0]["data_path"])
    q_d, v_d = np.asarray(d["Q_mAh_gSPAN"]), np.asarray(d["V"])
    fv = json.loads(str(d["model_json"]))["fv_estimates"]          # the FV fit that produced the data file
    w_S = fv["w_S"]
    base = sp5.base_params(1.0)
    out = sp5.OUT / "fit_cycle3_pinnwindow.json"
    ck = FitCheckpoint(sp5.OUT / "ckpt_cycle3_pinnwindow.json")
    if not args.resume and ck.state:
        ck.path.rename(ck.path.with_suffix(f".old{int(time.time())}.json"))
        ck = FitCheckpoint(ck.path)
    n = [ck.prior_solves if args.resume else 0]
    t0 = time.time() - (ck.prior_wall_s if args.resume else 0.0)

    def vals(x):
        return {"w_S": w_S, **{k: (0.1 * v if k.startswith("U0") else float(np.exp(v))) for k, v in zip(NAMES6, x)}}

    def resid(x):
        n[0] += 1
        Q, V, _, _ = sp5.simulate(vals(x), base)
        r = (np.where(q_d <= Q[-1], np.interp(q_d, Q, V), 1.0) - v_d) * 1e3
        ck.record(x, float(np.sqrt(np.mean(r ** 2))), n[0], time.time() - t0)
        print(f"  eval {n[0]:3d} ({time.time() - t0:5.0f} s) rms {np.sqrt(np.mean(r ** 2)):7.2f} mV", flush=True)
        return r

    x0 = np.array([fv[k] / 0.1 if k.startswith("U0") else np.log(fv[k]) for k in NAMES6])
    if args.resume and ck.state:
        x0 = ck.x()
    lo, hi = np.array([-10.0] * 3 + [-1.5] * 3), np.array([10.0] * 3 + [1.5] * 3)
    fun, jac = least_squares_fd(resid, np.full(6, 0.01), lo, hi)
    sol = least_squares(fun, x0, jac=jac, method="trf", max_nfev=args.max_nfev, bounds=(lo, hi), ftol=1e-4, xtol=1e-4)
    rep = {"config": Path(args.config).name, "data": spec["rates"][0]["data_path"], "points": int(len(q_d)),
           "q_max_mAh_gSPAN": float(q_d[-1]), "w_S_fixed": w_S, "start": "full-curve FV estimates",
           "estimates": vals(sol.x), "rms_mV": float(np.sqrt(np.mean(sol.fun ** 2))), "solves": n[0],
           "status": int(sol.status)}
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    sys.exit(main())
