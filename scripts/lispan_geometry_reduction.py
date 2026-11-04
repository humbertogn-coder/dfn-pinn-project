"""Which geometry inputs does network v5 need?  Linearized parameter bias of reduced network geometries (step 1).

A network trained for one cell geometry G_net, fitted to data from a cell with geometry G_data, returns biased
chemistry parameters (step 0, lispan_geometry_bias.py).  A network with geometry inputs removes that bias only for the
inputs it has.  This script measures, with the FV solver alone (no network), the bias that remains when the network
knows only part of the geometry:

  reduction   network geometry given the data geometry (L_cat, f, L_sep, sep porosity, beta)
  ref         the v4 network: 100 um, f = 1, 618 um glass fibre (should reproduce the step-0 network biases)
  sep_fixed   cathode exact (L_cat, f), separator fixed to a Celgard-type 25 um / 0.41 / beta 2.33
  load_f3     separator Celgard, f fixed to 3, L_cat from the loading (m_S exact): one cathode input
  load_f_cat  separator Celgard, (L_cat, f) exact (= two cathode inputs; same as sep_fixed for Celgard data)

Bias = Gauss-Newton fit (FV curves of G_net, FD Jacobian, 3 iterations) of the FV data curves of G_data minus the truth,
with the tool's parameters (w_S, the 5 v2 chemistry inputs, d, Z_CC); calibration rates 0.2 + 2 C as in step 0.

    python scripts/lispan_geometry_reduction.py --workers 2
Output: results/lispan/geometry/reduction_<case>.json
"""

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from dfn_pinn.lispan.chem_pinn import CHEM_SPECS, chem_vector  # noqa: E402
from dfn_pinn.lispan.model import solve  # noqa: E402
from dfn_pinn.lispan.params import LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.rate_pinn import ONE_C  # noqa: E402
from dfn_pinn.lispan.two_curve import fv_params, m_S_g  # noqa: E402
from lispan_geometry_bias import geometry  # noqa: E402

SPEC = CHEM_SPECS["v2"]
NAMES = ["w_S"] + [s["name"] for s in SPEC] + ["d", "Z_CC"]
CEL = (25.0, 0.41, 2.33)
REF = (100.0, 1.0, 618.0, 0.89, 1.0)
DATA = {   # name: (L_cat, f, L_sep, porosity, beta)
    "coin_33_f3": (33.0, 3.0, *CEL), "coin_67_f3": (67.0, 3.0, *CEL), "coin_100_f3": (100.0, 3.0, *CEL),
    "coin_50_f2": (50.0, 2.0, *CEL), "coin_150_f2": (150.0, 2.0, *CEL), "coin_40_f5": (40.0, 5.0, *CEL),
    "coin_80_f5": (80.0, 5.0, *CEL), "pe_67_f3": (67.0, 3.0, 20.0, 0.40, 2.33), "gf_67_f3": (67.0, 3.0, 260.0, 0.90, 1.0),
}


def reduce(name, g):
    L, f, *_ = g
    if name == "ref":
        return REF
    if name in ("sep_fixed", "load_f_cat"):
        return (L, f, *CEL)
    if name == "load_f3":
        return (L * f / 3.0, 3.0, *CEL)
    raise KeyError(name)


def scale(n):
    return 1e3 if (n == "d" or n.startswith("dU0_")) else (1e4 if n == "Z_CC" else 100.0)


def curves(x, g, rates, q_grids, N=(20, 10)):
    """FV voltage on the capacity grids (data basis, per g S of w_S = 1 data) for parameter vector x, geometry g."""
    p = geometry(fv_params(LiSPANParams(), SPEC, x), *g)
    w = math.exp(x[0])
    out = []
    for r, q in zip(rates, q_grids):
        i_sp = r * ONE_C / m_S_g(LiSPANParams())
        res = None
        for tol in (1e-6, 1e-5):
            try:
                _, res = solve(p, LiSPANProtocol(current=i_sp * m_S_g(p) / w, ramp_s=30.0), N_c=N[0], N_s=N[1],
                               n_out=1200, rtol=tol)
                break
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
                res = None
        if res is None:
            return None
        Q, V = np.asarray(res["Q_mAh_gS"]) * w, np.asarray(res["V"])
        out.append(np.where(q <= Q[-1], np.interp(q, Q, V), np.nan))
    return out


def job(args):
    case, x_true, dname, red, rates, iters = args
    gd = DATA[dname]
    gn = reduce(red, gd)
    t0 = time.time()
    x_true = np.asarray(x_true, float)
    # data: FV of the data geometry, capacity grid to 0.9 of each end capacity
    p_d = geometry(fv_params(LiSPANParams(), SPEC, x_true), *gd)
    grids, data = [], []
    for r in rates:
        i_sp = r * ONE_C / m_S_g(LiSPANParams())
        _, res = solve(p_d, LiSPANProtocol(current=i_sp * m_S_g(p_d), ramp_s=30.0), N_c=20, N_s=10, n_out=1200, rtol=1e-6)
        Q, V = np.asarray(res["Q_mAh_gS"]), np.asarray(res["V"])
        q = np.linspace(max(15.0, i_sp * 100.0 / 3.6), 0.9 * Q[-1], 80)
        grids.append(q)
        data.append(np.interp(q, Q, V))
    y = np.concatenate(data)
    x = x_true.copy()
    hist = []
    for it in range(iters):
        base = curves(x, gn, rates, grids)
        if base is None:
            return {"data": dname, "reduction": red, "error": "FV failed"}
        v0 = np.concatenate(base)
        ok = np.isfinite(v0)
        r0 = y - v0
        hist.append(1e3 * float(np.sqrt(np.mean(r0[ok] ** 2))))
        J = np.zeros((len(y), len(x)))
        for j, n in enumerate(NAMES):
            if n == "d":
                J[:, j] = 1.0
                continue
            if n == "Z_CC":
                p_n = geometry(fv_params(LiSPANParams(), SPEC, x), *gn)
                J[:, j] = -np.concatenate([np.full(len(q), r * ONE_C / m_S_g(LiSPANParams()) * m_S_g(p_n) / math.exp(x[0]))
                                            for r, q in zip(rates, grids)])
                continue
            h = 0.01 if n != "w_S" else 0.005
            xp = x.copy()
            xp[j] += h
            c = curves(xp, gn, rates, grids)
            if c is None:
                return {"data": dname, "reduction": red, "error": f"FV failed in the Jacobian ({n})"}
            J[:, j] = (np.concatenate(c) - v0) / h
            ok &= np.isfinite(J[:, j])
        step, *_ = np.linalg.lstsq(J[ok], r0[ok], rcond=None)
        x = x + step
    final = curves(x, gn, rates, grids)
    rms = None if final is None else 1e3 * float(np.sqrt(np.nanmean((y - np.concatenate(final)) ** 2)))
    return {"case": case, "data": dname, "reduction": red, "g_data": gd, "g_net": gn, "misfit_mV": rms,
            "misfit_history_mV": hist, "bias": {n: scale(n) * float(x[j] - x_true[j]) for j, n in enumerate(NAMES)},
            "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default=str(ROOT / "configs" / "lispan_chem_design_v3.json"))
    ap.add_argument("--case", default="p0")
    ap.add_argument("--data", nargs="*", default=list(DATA))
    ap.add_argument("--reductions", nargs="*", default=["ref", "sep_fixed", "load_f3"])
    ap.add_argument("--rates", nargs=2, type=float, default=[0.2, 2.0])
    ap.add_argument("--Z", type=float, default=0.002)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    design = json.loads(Path(a.design).read_text())
    chem = next(c["chem"] for c in design["eval_cases"] if c["id"].split("_")[0] == a.case)
    x_true = [0.0] + chem_vector(SPEC, chem) + [0.0, a.Z]
    jobs = [(a.case, x_true, d, red, a.rates, a.iters) for d in a.data for red in a.reductions
            if not (red == "sep_fixed" and DATA[d][2:] == CEL)]       # identical geometry -> no bias
    t0 = time.time()
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            if "error" in r:
                print(r["data"], r["reduction"], r["error"], flush=True)
                continue
            b = r["bias"]
            print(f"{r['data']:12s} {r['reduction']:10s} misfit {r['misfit_mV']:5.1f} | " +
                  " ".join(f"{n} {b[n]:+6.1f}" for n in NAMES) + f" ({r['wall_s']:.0f} s)", flush=True)
    path = Path(a.json) if a.json else ROOT / "results" / "lispan" / "geometry" / f"reduction_{a.case}.json"
    path.write_text(json.dumps({"args": vars(a), "data_geometries": DATA, "results": res}, indent=1, default=float))
    print("written", path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
