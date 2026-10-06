"""Calibrate the effective SPAN frequency factors of the Li-SPAN reference to the paper's simulated
discharge curves (Simanjuntak 2024, Fig. 6a, k0 variation panel: blue curves = Table 3 kinetics, Z_CC = 0,
0.1 C solid and 1 C dotted; digitized by scripts/lispan_digitize_paper.py).

    python scripts/lispan_fit_paper.py --out results/lispan/fit_k0.json

Only the products a_SPAN k0 enter the kinetics; a_SPAN is kept at the paper's 1e7 1/m and the three k0
are fitted (log10).  See docs/LI_SPAN_MODEL_FORMULATION.md section 5.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.lispan import LiSPANParams, LiSPANProtocol, solve  # noqa: E402

DIG = ROOT / "results" / "lispan" / "paper_digitized"


def load_curve(name):
    d = np.loadtxt(DIG / name, delimiter=",", skiprows=1)
    d = d[np.argsort(d[:, 0])]
    return d


def model_curve(k0, crate, zcc=0.0, **kw):
    p = LiSPANParams(k0=tuple(k0), Z_CC=zcc, **kw)
    _, res = solve(p, LiSPANProtocol.from_crate(crate), n_out=400)
    return res["Q_mAh_gS"], res["V"], res


def misfit(k0, targets, q_min=30.0, verbose=False, **kw):
    tot, n = 0.0, 0
    parts = {}
    for crate, curve in targets:
        Q, V, _ = model_curve(k0, crate, **kw)
        qs = curve[:, 0]; m = (qs >= q_min) & (qs <= Q[-1] + 150.0)
        qs, vs = qs[m], curve[m, 1]
        vm = np.interp(qs, Q, V, right=np.nan)
        # beyond the model's cut-off: the model is at V_min while the paper continues -> count as V_min
        vm = np.where(np.isnan(vm), 1.0, vm)
        e = vm - vs
        parts[crate] = float(np.sqrt(np.mean(e ** 2)))
        tot += np.sum(e ** 2); n += len(e)
    rms = float(np.sqrt(tot / n))
    if verbose:
        print(f"k0 = {k0}  rms {1e3 * rms:.1f} mV  " + "  ".join(f"{c} C: {1e3 * v:.1f}" for c, v in parts.items()), flush=True)
    return rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "lispan" / "fit_k0.json"))
    ap.add_argument("--x0", default="-7.6,-7.6,-7.6", help="log10 k0 start")
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--targets", default="fig6a", choices=["fig6a", "fig4b"])
    args = ap.parse_args()
    if args.targets == "fig6a":
        targets = [(0.1, load_curve("fig6a_k0_1e-2_0.1C.csv")), (1.0, load_curve("fig6a_k0_1e-2_1C.csv"))]
        zcc = 0.0
    else:
        targets = [(0.1, load_curve("fig4b_0.1C.csv")), (1.0, load_curve("fig4b_1C.csv"))]
        zcc = 0.025
    x0 = np.array([float(v) for v in args.x0.split(",")])
    t0 = time.perf_counter()

    def f(x):
        return misfit(10.0 ** x, targets, verbose=True, zcc=zcc)
    r = minimize(f, x0, method="Nelder-Mead", options={"maxiter": args.maxiter, "xatol": 0.02, "fatol": 2e-4,
                                                        "initial_simplex": np.array([x0, x0 + [0.5, 0, 0], x0 + [0, 0.5, 0], x0 + [0, 0, 0.5]])})
    k0 = 10.0 ** r.x
    print("best", k0, "rms", 1e3 * r.fun, "mV", f"({time.perf_counter() - t0:.0f} s)")
    out = {"k0": [float(v) for v in k0], "log10_k0": [float(v) for v in r.x], "rms_mV": 1e3 * float(r.fun),
           "targets": args.targets, "Z_CC": zcc, "nfev": int(r.nfev)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("written", args.out)


if __name__ == "__main__":
    main()
