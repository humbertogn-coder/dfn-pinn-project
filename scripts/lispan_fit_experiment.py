"""Classical least-squares fit of the Li-SPAN finite-volume model to measured discharge points (reference answer
for the PINN inverse on the same data).

    python scripts/lispan_fit_experiment.py --fit 0.1 1 --predict 0.05 0.2

Parameters: {k0_1, k0_2, k0_3, b_1, b_2, b_3, U0_1, Z_CC} (multipliers, U0_1 offset in V) on the PINN-benchmark
model (reaction 1 reversible, 2 and 3 irreversible, no double layer, 30 s ramp); data files from
scripts/lispan_make_inverse_data.py --experiment (results/lispan/inverse_data/V_<rate>C_experiment.npz).
scipy least_squares (trust region) with a forward-difference Jacobian of ABSOLUTE step 0.01 (1 % in log, 1 mV for
U0_1; dfn_pinn.lispan.fdjac - scipy's relative steps collapse to 1.5e-8 at x = 0, below the solver tolerance; fits
made before 2026-10-08 used them).

Synthetic mode (reference answer and cost of a classical inverse for the PINN multi-rate test):

    python scripts/lispan_fit_experiment.py --data nominal --noise-mV 1 --grid 40 20 --predict \
        --start '{"k0_1": 2.0, "k0_2": 0.5, "k0_3": 1.5, "b_1": 1.1, "b_2": 0.9, "b_3": 1.1, "U0_1": 0.02, "Z_CC": 1.5}'

uses V_<rate>C_nominal.npz with exactly the noise realization and time window of the PINN multi-rate run
(rate index k in --fit order, numpy default_rng(noise_seed + 100 k + 7), t <= 0.98 t_end of the reference), i.e. the
nonlinear least-squares estimate a perfect forward model would return from the same data.
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
from dfn_pinn.lispan.fdjac import FitCheckpoint, least_squares_fd  # noqa: E402

NAMES = ["k0_1", "k0_2", "k0_3", "b_1", "b_2", "b_3", "U0_1", "Z_CC"]
U0_SCALE = 0.1   # same convention as the PINN (offset = U0_SCALE * x)
GRID = [20, 10]  # finite volumes cathode / separator (set by --grid)


def values(x):
    return {n: (U0_SCALE * v if n.startswith("U0") else float(np.exp(v))) for n, v in zip(NAMES, x)}


def model_voltage(x, crate, t, base, N_c=20, N_s=10):
    N_c, N_s = GRID
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
    ap.add_argument("--data", choices=["experiment", "nominal"], default="experiment")
    ap.add_argument("--noise-mV", type=float, default=0.0, help="synthetic data only")
    ap.add_argument("--noise-seed", type=int, default=0, help="the PINN run's cfg.seed")
    ap.add_argument("--start", default=None, help="JSON dict of starting multipliers (U0_1 as offset in V)")
    ap.add_argument("--grid", nargs=2, type=int, default=[20, 10])
    ap.add_argument("--fd-step", type=float, default=0.01, help="absolute FD step (log units / 0.1 V units)")
    ap.add_argument("--tol", type=float, default=1e-4, help="ftol = xtol of least_squares (the BDF solutions are "
                    "accurate to ~1e-6 relative, so scipy's 1e-8 is never met and the fit would run to max_nfev)")
    ap.add_argument("--resume", action="store_true", help="continue from the checkpoint of an interrupted fit "
                    "(<out>/ckpt_<rates>C<kind><tag>.json, written whenever the misfit improves)")
    args = ap.parse_args()
    GRID[:] = args.grid
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)
    data = {}
    truth = None
    for cr in set(args.fit) | set(args.predict):
        d = np.load(ROOT / "results" / "lispan" / "inverse_data" / f"V_{cr:g}C_{args.data}.npz")
        if args.data == "experiment":
            data[cr] = (np.asarray(d["t"]), np.asarray(d["V"]), np.asarray(d["Q_mAh_gS"]))
            continue
        t_d, V_d = np.asarray(d["t"], float), np.asarray(d["V"], float)
        if args.noise_mV > 0:     # same draw as dfn_pinn.lispan.multirate.load_data
            k = args.fit.index(cr) if cr in args.fit else 50 + len(data)
            V_d = V_d + np.random.default_rng(args.noise_seed + 100 * k + 7).normal(0.0, args.noise_mV * 1e-3, V_d.shape)
        keep = t_d <= 0.98 * float(d["t_end_ref"])
        data[cr] = (t_d[keep], V_d[keep], None)
        truth = {n: (0.0 if n.startswith("U0") else 1.0) for n in NAMES}
    kind = "" if args.data == "experiment" else f"_{args.data}"
    ck = FitCheckpoint(out / f"ckpt_{'+'.join(f'{c:g}' for c in args.fit)}C{kind}{args.tag}.json")
    if not args.resume and ck.state:
        ck.path.rename(ck.path.with_suffix(f".old{int(time.time())}.json"))      # a new fit: keep the old file aside
        ck = FitCheckpoint(ck.path)
    t0 = time.time() - (ck.prior_wall_s if args.resume else 0.0)
    c0 = time.process_time()
    n_eval = [ck.prior_solves if args.resume else 0]

    def resid(x, record=True):
        n_eval[0] += 1
        r = np.concatenate([model_voltage(x, cr, data[cr][0], base) - data[cr][1] for cr in args.fit]) * 1e3
        if record:     # never the nominal/truth evaluation below: a resumed synthetic fit would start at the truth
            ck.record(x, float(np.sqrt(np.mean(r ** 2))), n_eval[0], time.time() - t0)
        print(f"  eval {n_eval[0]:3d} ({time.time() - t0:5.0f} s): rms {np.sqrt(np.mean(r ** 2)):7.2f} mV | "
              + ", ".join(f"{k} {v:.4g}" for k, v in values(x).items()), flush=True)
        return r

    x0 = np.zeros(len(NAMES))
    x_start = x0.copy()
    if args.init_json or args.start:
        est0 = json.loads(Path(args.init_json).read_text())["estimates"] if args.init_json else json.loads(args.start)
        x_start = np.array([est0[n] / U0_SCALE if n.startswith("U0") else np.log(est0[n]) for n in NAMES])
    x_orig = x_start.copy()
    if args.resume and ck.state:
        x_start = ck.x()
        print(f"resuming from {ck.path.name}: rms {ck.best:.2f} mV after {ck.prior_solves} solves", flush=True)
    r0 = resid(x0, record=False)
    lo, hi = -2.0 * np.ones(len(NAMES)), 2.0 * np.ones(len(NAMES))
    fun, jac = least_squares_fd(resid, np.full(len(NAMES), args.fd_step), lo, hi)
    sol = least_squares(fun, x_start, jac=jac, method="trf", x_scale=1.0, max_nfev=args.max_nfev, bounds=(lo, hi),
                        ftol=args.tol, xtol=args.tol)
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
              "per_rate_rms_mV": {}, "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0,
              "n_solves_per_rate": n_eval[0], "grid": GRID, "fd_step": args.fd_step, "status": int(sol.status),
              "resumed": bool(args.resume), "cpu_s_note": "cpu_s covers this process only; wall_s and the solve count "
              "include the interrupted part(s) when resumed", "data": args.data, "noise_mV": args.noise_mV}
    if truth is not None:
        report["errors_pct_or_mV"] = {n: (1e3 * est[n] if n.startswith("U0") else 100.0 * (est[n] - 1.0)) for n in NAMES}
        # Cramer-Rao bound on the actual observation points, sigma = the noise level (J is in mV per unit of x)
        try:
            sd_x = np.sqrt(np.diag(np.linalg.inv(J.T @ J))) * args.noise_mV
            report["crlb_pct_or_mV"] = {n: (1e3 * U0_SCALE * s if n.startswith("U0") else 100.0 * s)
                                        for n, s in zip(NAMES, sd_x)}
        except np.linalg.LinAlgError:
            pass
    for cr in sorted(data):
        r = (model_voltage(sol.x, cr, data[cr][0], base) - data[cr][1]) * 1e3
        r_nom = (model_voltage(x0, cr, data[cr][0], base) - data[cr][1]) * 1e3
        report["per_rate_rms_mV"][f"{cr:g}C"] = {"nominal": float(np.sqrt(np.mean(r_nom ** 2))),
                                                "fitted": float(np.sqrt(np.mean(r ** 2))),
                                                "role": "fit" if cr in args.fit else "prediction"}
    report["start"] = values(x_orig)
    (out / f"fit_{'+'.join(f'{c:g}' for c in args.fit)}C{kind}{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
