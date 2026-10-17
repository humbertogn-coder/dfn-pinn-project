"""Phase 2 pilot: calibrate-then-predict on published multi-rate SPAN discharge curves (effective Li-SPAN model).

Data (scripts/extract_chen2019_fig3.py, scripts/digitize_ma2026_fig6.py -> data/literature_span/):
  chen2019_carbonate / chen2019_ether : Se0.06SPAN, Chen et al. Nat. Commun. 10, 1021 (2019) Fig. 3d / 3b,
      0.4, 1, 2, 4, 6, 10 A/g (per g S+Se; the 0.2 A/g curve is the cell's first cycle and is not used)
  ma2026_SPAN / ma2026_SPAN-A : Ma et al. Small 22, e74169 (2026) Fig. 6c / 6d, carbonate electrolyte,
      0.1, 0.2, 0.3, 0.5, 1, 2 C with 1 C = 1.675 A/g S (the 0.1 C curve is the first cycle of the rate test)
  hao2025_SPAN : plain SPAN, Hao et al. ACS Nano 19, 25385 (2025) SI Fig. S9, 1 M LiPF6 EC/DEC/FEC,
      0.1, 0.2, 0.5, 1, 2 C, capacity per g SPAN, 1 C = 0.71 A/g SPAN (from the GITT pulse size, SI Fig. S11a; the
      paper does not define 1 C; data/literature_span/README.md)
  hao2025_SPAN_cycling : same cell type, SI Fig. S12b, 0.5 C, cycles 2, 10, 100 (the "rates" of this entry are cycle
      numbers; all at 0.5 C)

Model (as for the SPAN500 calibration, scripts/lispan_span500.py): Simanjuntak geometry and electrolyte, PINN-benchmark
form (reaction 1 reversible, 2-3 irreversible, no double layer, 30 s ramp). The papers' capacity basis is mapped to the
model's sulfur by an effective factor w_S:  Q_data = w_S * Q_model[mAh/g model S], and the model current for a specific
current i [A/g] is I = i * m_S_model / w_S [A/m2]. Electrode thickness and loading of the papers' cells are NOT
modelled (not all reported); w_S, the contact resistance and the kinetics absorb them, so all parameters are effective.

Fitted (any subset can be fixed with --fix): w_S, U0_1, U0_2, U0_3 (offsets, V), b_1..3 (multipliers), k0_1
(multiplier; reaction 1 is the only reversible step, so its k0 and U0 are separate parameters), Z_CC (multiplier of
0.025 Ohm m2). k0_2, k0_3 stay at the Simanjuntak effective values: for the irreversible steps only
k0 exp(F U0 / 2RT) enters, and the U0 offsets carry it (docs/PHASE1_IDENTIFIABILITY.md).

Fits use 10 / 5 finite volumes and rtol 1e-5 (0.1 mV from the 20 / 10 grid used for the reported curves).
Residuals: model voltage at the data capacities (80 points per curve, uniform in capacity from Q_MIN to 0.98 of the
measured end capacity; a model that has reached 1.0 V earlier counts as 1.0 V), every rate weighted equally.

    python scripts/lispan_lit_rates.py --dataset chen2019_carbonate --fit 0.4 2 --tag slowfast
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
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

DATA = ROOT / "data" / "literature_span"
OUT = ROOT / "results" / "lispan" / "literature_rates"
DATASETS = {
    "chen2019_carbonate": {"dir": "chen2019", "prefix": "carbonate", "unit": "Ag",
                           "rates": {0.4: 0.4, 1.0: 1.0, 2.0: 2.0, 4.0: 4.0, 6.0: 6.0, 10.0: 10.0}, "q_min": 15.0},
    "chen2019_ether": {"dir": "chen2019", "prefix": "ether", "unit": "Ag",
                       "rates": {0.4: 0.4, 1.0: 1.0, 2.0: 2.0, 4.0: 4.0, 6.0: 6.0, 10.0: 10.0}, "q_min": 15.0},
    "ma2026_SPAN": {"dir": "ma2026", "prefix": "SPAN", "unit": "C",
                    "rates": {r: 1.675 * r for r in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)}, "q_min": 60.0},
    "ma2026_SPAN-A": {"dir": "ma2026", "prefix": "SPAN-A", "unit": "C",
                      "rates": {r: 1.675 * r for r in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)}, "q_min": 60.0},
    "hao2025_SPAN": {"dir": "hao2025", "prefix": "SPAN", "unit": "C",
                     "rates": {r: 0.71 * r for r in (0.1, 0.2, 0.5, 1.0, 2.0)}, "q_min": 40.0},
    "hao2025_SPAN_cycling": {"dir": "hao2025", "prefix": "SPAN_0.5C", "unit": "cyc",
                             "rates": {2.0: 0.355, 10.0: 0.355, 100.0: 0.355}, "q_min": 40.0},
}
NAMES = ["w_S", "U0_1", "U0_2", "U0_3", "b_1", "b_2", "b_3", "k0_1", "Z_CC"]
U0_SCALE = 0.1
BASE = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)
LO = {"w_S": np.log(0.2), "U0": -10.0, "b": -1.5, "k0_1": -6.0, "Z_CC": -7.0, "D_salt": np.log(1e-3),
      "kappa0": np.log(1e-2)}
HI = {"w_S": np.log(3.0), "U0": 10.0, "b": 1.5, "k0_1": 6.0, "Z_CC": 3.0, "D_salt": np.log(10.0), "kappa0": np.log(10.0)}
EXTRA = ["D_salt", "kappa0"]       # optional transport multipliers (--extra)


class SolveTimeout(Exception):
    pass


def _alarm(*_):
    raise SolveTimeout()


def load(ds, label, n=80):
    cfg = DATASETS[ds]
    f = DATA / cfg["dir"] / f"{cfg['prefix']}_{label:g}{cfg['unit']}.csv"
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    q, v = d[:, 0], d[:, 1]
    q_end = float(q.max())
    qq = np.linspace(cfg["q_min"], 0.98 * q_end, n)
    return {"q": qq, "v": np.interp(qq, q, v), "q_end": q_end, "q_raw": q, "v_raw": v, "i_sp": cfg["rates"][label]}


def bounds(names):
    lo = [LO["U0"] if n.startswith("U0") else LO["b"] if n.startswith("b_") else LO[n] for n in names]
    hi = [HI["U0"] if n.startswith("U0") else HI["b"] if n.startswith("b_") else HI[n] for n in names]
    return np.array(lo), np.array(hi)


def to_vals(x, names):
    return {n: (U0_SCALE * v if n.startswith("U0") else float(np.exp(v))) for n, v in zip(names, x)}


def to_x(vals, names):
    return np.array([vals[n] / U0_SCALE if n.startswith("U0") else np.log(vals[n]) for n in names])


def simulate(vals, i_sp, N=(20, 10), rtol0=1e-6, timeout=90):
    """Discharge at i_sp [A/g of the data basis]; returns (Q_data-basis mAh/g, V) or None on solver failure."""
    p = apply_truth(BASE, {k: v for k, v in vals.items() if k != "w_S"})
    w = vals["w_S"]
    current = i_sp * (p.m_S_model * 1e3) / w                              # A/m2
    prot = LiSPANProtocol(current=current, ramp_s=30.0)
    for rtol in (rtol0, 10 * rtol0):
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
        try:
            _, res = solve(p, prot, N_c=N[0], N_s=N[1], n_out=1200, rtol=rtol)
            return res["Q_mAh_gS"] * w, res["V"]
        except (SolveTimeout, RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue       # timeout, or a singular Jacobian when the salt is depleted: retry looser, then give up
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    return None


TIMEOUT = [90]


def model_on(vals, d, N=(20, 10), rtol=1e-6):
    out = simulate(vals, d["i_sp"], N=N, rtol0=rtol, timeout=TIMEOUT[0])
    if out is None:
        return np.full_like(d["v"], np.nan), None
    Q, V = out
    return np.where(d["q"] <= Q[-1], np.interp(d["q"], Q, V), 1.0), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--fit", nargs="+", type=float, required=True)
    ap.add_argument("--fix", nargs="*", default=[], help="NAME=value (multiplier; U0_m offset in V; w_S value)")
    ap.add_argument("--extra", nargs="*", default=[], choices=EXTRA,
                    help="also fit these transport multipliers (salt diffusivity, electrolyte conductivity)")
    ap.add_argument("--start", default=None, help="JSON dict of starting values")
    ap.add_argument("--start-json", default=None, help="start from the estimates of an earlier fit (its JSON report)")
    ap.add_argument("--fit-grid", nargs=2, type=int, default=[10, 5], help="finite volumes used during the fit "
                    "(the reported curves use 20 / 10; the difference is ~0.1 mV)")
    ap.add_argument("--fit-rtol", type=float, default=1e-5)
    ap.add_argument("--timeout", type=int, default=90, help="seconds per solve before it is retried / given up")
    ap.add_argument("--max-nfev", type=int, default=40)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    TIMEOUT[0] = args.timeout
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = DATASETS[args.dataset]
    labels = sorted(cfg["rates"])
    data = {r: load(args.dataset, r) for r in labels}
    fixed = {k: float(v) for k, v in (it.split("=") for it in args.fix)}
    names = NAMES + list(args.extra)
    free = [n for n in names if n not in fixed]
    slow = min(args.fit)
    start = {"w_S": data[slow]["q_end"] / 1250.0, "U0_1": 0.0, "U0_2": 0.0, "U0_3": 0.0, "b_1": 1.0, "b_2": 1.0,
             "b_3": 1.0, "k0_1": 1.0, "Z_CC": 0.3, "D_salt": 1.0, "kappa0": 1.0}
    if args.start_json:
        start.update(json.loads(Path(args.start_json).read_text())["estimates"])
    if args.start:
        start.update(json.loads(args.start))
    x0 = to_x({n: start[n] for n in free}, free)
    n_eval, t0 = [0], time.time()

    def full(x):
        v = dict(fixed)
        v.update(to_vals(x, free))
        return v

    def resid(x):
        n_eval[0] += 1
        rr = []
        for r in args.fit:
            vm, _ = model_on(full(x), data[r], N=tuple(args.fit_grid), rtol=args.fit_rtol)
            rr.append(np.where(np.isfinite(vm), vm - data[r]["v"], 1.0))
        r_all = np.concatenate(rr) * 1e3
        print(f"  eval {n_eval[0]:3d} ({time.time() - t0:5.0f} s): rms {np.sqrt(np.mean(r_all ** 2)):7.2f} mV | "
              + ", ".join(f"{k} {v:.4g}" for k, v in full(x).items()), flush=True)
        return r_all

    lo, hi = bounds(free)
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    fun, jac = least_squares_fd(resid, np.full(len(free), 0.01), lo, hi)
    sol = least_squares(fun, x0, jac=jac, method="trf", x_scale=1.0, max_nfev=args.max_nfev, bounds=(lo, hi),
                        ftol=1e-4, xtol=1e-4)
    est = full(sol.x)
    per, curves = {}, {}
    for r in labels:
        vm, out = model_on(est, data[r])
        e = (vm - data[r]["v"]) * 1e3
        Qm = float(out[0][-1]) if out is not None else None
        per[f"{r:g}"] = {"role": "fit" if r in args.fit else "prediction", "i_sp_A_per_g": data[r]["i_sp"],
                         "rms_mV": float(np.sqrt(np.nanmean(e ** 2))), "mean_mV": float(np.nanmean(e)),
                         "max_abs_mV": float(np.nanmax(np.abs(e))), "q_end_data": data[r]["q_end"], "q_end_model": Qm}
        if out is not None:
            curves[f"Q_{r:g}"], curves[f"V_{r:g}"] = out
        curves[f"qd_{r:g}"], curves[f"vd_{r:g}"] = data[r]["q_raw"], data[r]["v_raw"]
    name = f"{args.dataset}_fit{'+'.join(f'{r:g}' for r in args.fit)}{('_' + args.tag) if args.tag else ''}"
    rep = {"dataset": args.dataset, "fit_rates": args.fit, "fixed": fixed, "free": free, "estimates": est,
           "Z_CC_ohm_m2": 0.025 * est["Z_CC"], "rms_fit_mV": float(np.sqrt(np.mean(sol.fun ** 2))), "per_rate": per,
           "n_solves": n_eval[0], "wall_s": time.time() - t0, "status": int(sol.status), "nfev": int(sol.nfev)}
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1))
    np.savez(OUT / f"{name}.npz", **curves)
    print(json.dumps({k: rep[k] for k in ("estimates", "rms_fit_mV", "per_rate", "n_solves", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
