"""Phase 1 (identifiability proposal): nonlinear calibrate-then-predict tests with the finite-volume model.

Calibrate the Li-SPAN model on some discharge rates, predict the others, and compare with the measured
(Simanjuntak et al. 2024, Fig. 4b, digitized points) or synthetic data.  Optionally fix parameters, e.g. the
contact resistance, to show how much the prediction at an unmeasured rate depends on an assumption that the
calibration data cannot test.

Parameters (as in scripts/lispan_fit_experiment.py): multipliers of k0_1..3, b_1..3, Z_CC and the offset U0_1 [V]
on the PINN-benchmark model (reaction 1 reversible, 2-3 irreversible, Z_CC 0.025 Ohm m2, 30 s ramp).  U0_2 and U0_3
stay fixed: for an irreversible step the rate depends only on k0 exp(F U0 / 2RT), at every current.

    # measured data: calibrate on 0.1 C with Z_CC fixed at half the nominal value, predict 0.05, 0.2 and 1 C
    python scripts/lispan_phase1_fit.py --data experiment --fit 0.1 --predict 0.05 0.2 1 --fix Z_CC=0.5 --tag Z0.5
    # synthetic data from the calibrated Simanjuntak cell, 5 mV noise
    python scripts/lispan_phase1_fit.py --data synthetic --fit 0.1 --predict 1 2 --fix Z_CC=2 --tag Z2
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
from dfn_pinn.lispan.fdjac import least_squares_fd  # noqa: E402
from lispan_make_inverse_data import apply_truth  # noqa: E402

ALL = ["k0_1", "k0_2", "k0_3", "b_1", "b_2", "b_3", "U0_1", "Z_CC"]
U0_SCALE = 0.1
OUT = ROOT / "results" / "lispan" / "phase1"
TRUTH_JSON = ROOT / "results" / "lispan" / "fit_experiment" / "fit_0.1+1C_fdabs.json"
BASE = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)


def to_x(vals, names):
    return np.array([vals[n] / U0_SCALE if n.startswith("U0") else np.log(vals[n]) for n in names])


def from_x(x, names):
    return {n: (U0_SCALE * v if n.startswith("U0") else float(np.exp(v))) for n, v in zip(names, x)}


def simulate(vals, crate, N_c, N_s, t_end=None):
    p = apply_truth(BASE, vals)
    prot = LiSPANProtocol.from_crate(crate, ramp_s=30.0)
    _, res = solve(p, prot, N_c=N_c, N_s=N_s, n_out=1500, t_end=t_end)
    return res["t"], res["V"], prot, p


def v_at(vals, crate, t, N_c, N_s):
    tt, VV, prot, _ = simulate(vals, crate, N_c, N_s, t_end=float(t[-1]) * 1.05 + 100.0)
    return np.where(t <= tt[-1], np.interp(t, tt, VV), prot.V_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["experiment", "synthetic"], default="experiment")
    ap.add_argument("--fit", nargs="+", type=float, required=True)
    ap.add_argument("--predict", nargs="*", type=float, default=[])
    ap.add_argument("--fix", nargs="*", default=[], help="NAME=multiplier (U0_1: offset in V), held fixed")
    ap.add_argument("--start", default=None, help="JSON dict of starting values for the free parameters")
    ap.add_argument("--noise-mV", type=float, default=5.0, help="synthetic data: noise on the calibration curves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-obs", type=int, default=100, help="synthetic data: samples per curve")
    ap.add_argument("--grid", nargs=2, type=int, default=[20, 10])
    ap.add_argument("--max-nfev", type=int, default=30)
    ap.add_argument("--fd-step", type=float, default=0.01)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    N_c, N_s = args.grid
    truth = json.loads(TRUTH_JSON.read_text())["estimates"]
    fixed = {}
    for item in args.fix:
        k, v = item.split("=")
        fixed[k] = float(v)
    free = [n for n in ALL if n not in fixed]
    rates = sorted(set(args.fit) | set(args.predict))

    data = {}
    for k, cr in enumerate(rates):
        if args.data == "experiment":
            d = np.load(ROOT / "results" / "lispan" / "inverse_data" / f"V_{cr:g}C_experiment.npz")
            data[cr] = {"t": np.asarray(d["t"], float), "V": np.asarray(d["V"], float), "V_true": np.asarray(d["V"], float)}
        else:
            tt, VV, _, _ = simulate(truth, cr, N_c, N_s)
            t = np.linspace(0.02 * tt[-1], 0.98 * tt[-1], args.n_obs)
            Vt = np.interp(t, tt, VV)
            noise = np.random.default_rng(args.seed + 100 * k + 7).normal(0.0, args.noise_mV * 1e-3, t.shape)
            data[cr] = {"t": t, "V": Vt + (noise if cr in args.fit else 0.0), "V_true": Vt}

    def full(x):
        vals = dict(fixed)
        vals.update(from_x(x, free))
        return vals

    n_eval = [0]
    t0 = time.time()

    def resid(x):
        n_eval[0] += 1
        r = np.concatenate([v_at(full(x), cr, data[cr]["t"], N_c, N_s) - data[cr]["V"] for cr in args.fit]) * 1e3
        print(f"  eval {n_eval[0]:3d} ({time.time() - t0:5.0f} s): rms {np.sqrt(np.mean(r ** 2)):7.2f} mV | "
              + ", ".join(f"{k} {v:.4g}" for k, v in full(x).items()), flush=True)
        return r

    start = {n: (0.0 if n.startswith("U0") else 1.0) for n in free}
    if args.start:
        start.update({k: v for k, v in json.loads(args.start).items() if k in free})
    x0 = to_x(start, free)
    lo, hi = -2.5 * np.ones(len(free)), 2.5 * np.ones(len(free))
    fun, jac = least_squares_fd(resid, np.full(len(free), args.fd_step), lo, hi)
    sol = least_squares(fun, x0, jac=jac, method="trf", x_scale=1.0, max_nfev=args.max_nfev, bounds=(lo, hi),
                        ftol=args.tol, xtol=args.tol)
    est = full(sol.x)

    per_rate, curves = {}, {}
    for cr in rates:
        Vm = v_at(est, cr, data[cr]["t"], N_c, N_s)
        e_data = (Vm - data[cr]["V"]) * 1e3
        e_true = (Vm - data[cr]["V_true"]) * 1e3
        tt, VV, prot, p = simulate(est, cr, N_c, N_s)
        q = prot.current * tt / 3.6 / (1e3 * p.m_S_model)
        per_rate[f"{cr:g}C"] = {"role": "fit" if cr in args.fit else "prediction",
                                "rms_vs_data_mV": float(np.sqrt(np.mean(e_data ** 2))),
                                "rms_vs_truth_mV": float(np.sqrt(np.mean(e_true ** 2))),
                                "mean_vs_truth_mV": float(np.mean(e_true)),
                                "max_abs_vs_truth_mV": float(np.max(np.abs(e_true))),
                                "capacity_mAh_gS": float(q[-1])}
        curves[f"t_{cr:g}"], curves[f"V_{cr:g}"], curves[f"Q_{cr:g}"] = tt, VV, q
        curves[f"td_{cr:g}"], curves[f"Vd_{cr:g}"], curves[f"Vtrue_{cr:g}"] = data[cr]["t"], data[cr]["V"], data[cr]["V_true"]
    name = f"fit_{args.data}_{'+'.join(f'{c:g}' for c in args.fit)}C{('_' + args.tag) if args.tag else ''}"
    rep = {"data": args.data, "fit_rates": args.fit, "predict_rates": args.predict, "fixed": fixed, "free": free,
           "estimates": est, "Z_CC_ohm_m2": 0.025 * est["Z_CC"], "rms_fit_mV": float(np.sqrt(np.mean(sol.fun ** 2))),
           "per_rate": per_rate, "nfev": int(sol.nfev), "n_solves": n_eval[0], "wall_s": time.time() - t0,
           "status": int(sol.status), "noise_mV": args.noise_mV if args.data == "synthetic" else None,
           "truth": truth if args.data == "synthetic" else None, "grid": [N_c, N_s]}
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1))
    np.savez(OUT / f"{name}.npz", **curves)
    print(json.dumps({k: rep[k] for k in ("estimates", "rms_fit_mV", "per_rate", "n_solves", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
