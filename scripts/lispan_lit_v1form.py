"""Coverage check of the Phase 3b v1 chemistry box on the Phase 2 literature data: FV fit of a two-rate dataset with
the v1 parameterization of the chemistry network (src/dfn_pinn/lispan/chem_pinn.py, CHEM_INPUTS_V1):

    U0_1 = d,  U0_2 = d + dU0_2,  U0_3 = d + dU0_3   (offsets in V),
    b_1 = b_2 = b_3 = b                               (one common OCV-slope multiplier),
    k0_1 multiplier, Z_CC, w_S                        (k0_2, k0_3 carried by the U0 offsets, as in scripts/lispan_lit_rates.py)

    python scripts/lispan_lit_v1form.py --dataset chen2019_ether --fit 0.4 2 --mode box
    python scripts/lispan_lit_v1form.py --dataset hao2025_SPAN --fit 0.1 2 --mode wide

--mode box: dU0_2, dU0_3 within +-0.12 V, b within x0.6-1.5, k0_1 within x0.1-10 (the v1 box); --mode wide: the
bounds of scripts/lispan_lit_rates.py.  Compare the misfit with the free fit of lispan_lit_rates.py (separate b_m) on
the same rates: the increase is what the v1 network's parameterization costs on that dataset.  Same data, model, grid
and residuals as lispan_lit_rates.py (imported); results in results/lispan/literature_rates/<dataset>_v1form_*.json.
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

from dfn_pinn.lispan.fdjac import least_squares_fd  # noqa: E402
import lispan_lit_rates as L  # noqa: E402

NAMES = ["w_S", "d", "dU0_2", "dU0_3", "b", "k0_1", "Z_CC"]
BOX = {"dU0": 0.12, "b": (0.6, 1.5), "k0_1": (0.1, 10.0)}


def to_vals(x):
    w, d, d2, d3, lb, lk, lz = x
    b = float(np.exp(lb))
    return {"w_S": float(np.exp(w)), "U0_1": L.U0_SCALE * d, "U0_2": L.U0_SCALE * (d + d2),
            "U0_3": L.U0_SCALE * (d + d3), "b_1": b, "b_2": b, "b_3": b, "k0_1": float(np.exp(lk)),
            "Z_CC": float(np.exp(lz))}


def bounds(mode):
    lo = [L.LO["w_S"], L.LO["U0"], L.LO["U0"], L.LO["U0"], L.LO["b"], L.LO["k0_1"], L.LO["Z_CC"]]
    hi = [L.HI["w_S"], L.HI["U0"], L.HI["U0"], L.HI["U0"], L.HI["b"], L.HI["k0_1"], L.HI["Z_CC"]]
    if mode == "box":
        lo[2] = lo[3] = -BOX["dU0"] / L.U0_SCALE
        hi[2] = hi[3] = BOX["dU0"] / L.U0_SCALE
        lo[4], hi[4] = np.log(BOX["b"][0]), np.log(BOX["b"][1])
        lo[5], hi[5] = np.log(BOX["k0_1"][0]), np.log(BOX["k0_1"][1])
    return np.array(lo), np.array(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(L.DATASETS))
    ap.add_argument("--fit", nargs="+", type=float, required=True)
    ap.add_argument("--mode", choices=["box", "wide"], default="box")
    ap.add_argument("--start-json", default=None, help="free fit (lispan_lit_rates.py report) to start from; default "
                    "results/lispan/literature_rates/<dataset>_fit<rates>.json if present")
    ap.add_argument("--fit-grid", nargs=2, type=int, default=[10, 5])
    ap.add_argument("--fit-rtol", type=float, default=1e-5)
    ap.add_argument("--max-nfev", type=int, default=40)
    a = ap.parse_args()
    tag = "+".join(f"{r:g}" for r in a.fit)
    data = {r: L.load(a.dataset, r) for r in sorted(L.DATASETS[a.dataset]["rates"])}
    free_json = Path(a.start_json) if a.start_json else L.OUT / f"{a.dataset}_fit{tag}.json"
    free = json.loads(free_json.read_text()) if free_json.exists() else None
    if free:
        e = free["estimates"]
        x0 = np.array([np.log(e["w_S"]), e["U0_1"] / L.U0_SCALE, (e["U0_2"] - e["U0_1"]) / L.U0_SCALE,
                       (e["U0_3"] - e["U0_1"]) / L.U0_SCALE, np.mean(np.log([e["b_1"], e["b_2"], e["b_3"]])),
                       np.log(e["k0_1"]), np.log(e["Z_CC"])])
    else:
        x0 = np.array([np.log(data[min(a.fit)]["q_end"] / 1250.0), 0, 0, 0, 0, 0, np.log(0.3)])
    lo, hi = bounds(a.mode)
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    n_eval, t0 = [0], time.time()

    def resid(x):
        n_eval[0] += 1
        rr = []
        for r in a.fit:
            vm, _ = L.model_on(to_vals(x), data[r], N=tuple(a.fit_grid), rtol=a.fit_rtol)
            rr.append(np.where(np.isfinite(vm), vm - data[r]["v"], 1.0))
        r_all = np.concatenate(rr) * 1e3
        print(f"  eval {n_eval[0]:3d} ({time.time() - t0:5.0f} s): rms {np.sqrt(np.mean(r_all ** 2)):7.2f} mV | "
              + ", ".join(f"{n} {v:.4g}" for n, v in zip(NAMES, x)), flush=True)
        return r_all

    fun, jac = least_squares_fd(resid, np.full(len(NAMES), 0.01), lo, hi)
    sol = least_squares(fun, x0, jac=jac, method="trf", x_scale=1.0, max_nfev=a.max_nfev, bounds=(lo, hi),
                        ftol=1e-4, xtol=1e-4)
    est = to_vals(sol.x)
    per = {}
    for r in sorted(data):
        vm, out = L.model_on(est, data[r])
        e = (vm - data[r]["v"]) * 1e3
        per[f"{r:g}"] = {"role": "fit" if r in a.fit else "prediction", "rms_mV": float(np.sqrt(np.nanmean(e ** 2)))}
    x = sol.x
    v1 = {"dU0_2": L.U0_SCALE * x[2], "dU0_3": L.U0_SCALE * x[3], "b": float(np.exp(x[4])), "k0_1": float(np.exp(x[5])),
          "d": L.U0_SCALE * x[1], "Z_CC_ohm_m2": 0.025 * float(np.exp(x[6])), "w_S": float(np.exp(x[0]))}
    at_bound = [n for n, v, l_, h in zip(NAMES, x, lo, hi) if min(v - l_, h - v) < 1e-3]
    rep = {"dataset": a.dataset, "fit_rates": a.fit, "mode": a.mode, "v1_inputs": v1, "estimates": est,
           "at_bound": at_bound, "rms_fit_mV": float(np.sqrt(np.mean(sol.fun ** 2))),
           "rms_free_fit_mV": free["rms_fit_mV"] if free else None, "per_rate": per, "n_solves": n_eval[0],
           "wall_s": time.time() - t0, "status": int(sol.status)}
    out = L.OUT / f"{a.dataset}_v1form{tag}_{a.mode}.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("v1_inputs", "at_bound", "rms_fit_mV", "rms_free_fit_mV", "per_rate")},
                     indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
