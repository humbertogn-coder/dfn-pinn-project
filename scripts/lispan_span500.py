"""Real-data case (paper step 5): calibrate the Li-SPAN model to the SPAN500 C/10 discharge of Wang et al.,
Nat. Mater. 25, 791 (2026), Fig. 4a source data (data/span_natmater2026/41563_2026_2484_MOESM7_ESM.xlsx).

What the data are: 2032 coin cells, Li foil, Celgard 2320, ~2.0 mg/cm2 SPAN500 electrode (SPAN:Super-P:alginate
8:1:1), ether electrolyte, room temperature, C/10 with 1 C = 600 mA per g SPAN, 1-3 V; cycles 1-3 only, one rate.
What is not given: sulfur content of SPAN500, electrode thickness/porosity, which of the two ether electrolytes.

Model choices (all stated in the output JSON):
  * Simanjuntak geometry and electrolyte transport kept (at ~1 A/m2 the electrolyte and separator contribute < 1 mV,
    results/paper/assumption_checks.json), PINN benchmark form (reaction 1 reversible, 2-3 irreversible, no double
    layer, 30 s ramp), Z_CC = 2.2e-4 Ohm m2 = high-frequency intercept of the SPAN500 EIS (Fig. 4c, 1.93 Ohm x
    1.131 cm2).
  * Capacity axis: Q_model [mAh/g_S] = Q_data [mAh/g_SPAN] / w_S with the effective sulfur fraction w_S fitted (it
    also sets the model current, I = 0.06 A/g_SPAN / w_S per g of model sulfur).
  * Fitted: w_S, U0_1..3 (offsets, V), b_1..3 (multipliers).  One rate cannot separate kinetics from the OCV
    parameters (results/lispan/identifiability), so the fit is repeated for two kinetic hypotheses:
      "paper": the effective k0 fitted to Simanjuntak Fig. 6a (Tafel regime, ~0.1 V kinetic overpotential at C/10)
      "fast":  k0 x 1000 (near-equilibrium, the regime the ~10 Ohm cm2 SPAN500 EIS arc would suggest)
    and both calibrations are used to predict 1 C.

    python scripts/lispan_span500.py --cycle 3            # fits both hypotheses, figure, PINN data file
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
from lispan_make_inverse_data import apply_truth, time_of_charge  # noqa: E402
from dfn_pinn.lispan.fdjac import FitCheckpoint, least_squares_fd  # noqa: E402

XLSX = ROOT / "data" / "span_natmater2026" / "41563_2026_2484_MOESM7_ESM.xlsx"
OUT = ROOT / "results" / "lispan" / "span500"
Z_CC_EIS = 1.93 * 1.131e-4          # Ohm m2: HF intercept 1.93 Ohm on a 12 mm disc
I_SPAN = 0.060                      # A per g SPAN at C/10
NAMES = ["w_S", "U0_1", "U0_2", "U0_3", "b_1", "b_2", "b_3"]
HYPOTHESES = {"paper": 1.0, "fast": 1000.0}


def load_discharge(cycle):
    import pandas as pd
    df = pd.read_excel(XLSX, sheet_name="Fig.4a", header=None).iloc[3:].astype(float)
    j = 2 * (cycle - 1)
    q, v = df[j].to_numpy(), df[j + 1].to_numpy()
    ok = ~np.isnan(q)
    q, v = q[ok], v[ok]
    k = int(np.argmax(q))                 # discharge = rows up to the maximum capacity (charge follows, from 0)
    return q[: k + 1], v[: k + 1]


def base_params(k0_factor):
    p = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=Z_CC_EIS)
    return replace(p, k0=tuple(k * k0_factor for k in p.k0))


def unpack(x):
    """x = [log w_S, U0_1..3 offsets / 0.1 V, log b multipliers]."""
    return {"w_S": float(np.exp(x[0])), "U0_1": 0.1 * x[1], "U0_2": 0.1 * x[2], "U0_3": 0.1 * x[3],
            "b_1": float(np.exp(x[4])), "b_2": float(np.exp(x[5])), "b_3": float(np.exp(x[6]))}


def simulate(vals, base, crate_span=0.1, n_out=1500, N=(20, 10)):
    """Discharge at crate_span x 600 mA/g_SPAN; returns (Q in mAh/g_SPAN, V, protocol, result)."""
    p = apply_truth(base, {k: v for k, v in vals.items() if k != "w_S"})
    w = vals["w_S"]
    q_theo_per_gS = p.Q_theo / 3600.0 / (p.m_S_model * 1e3)          # Ah per g model sulfur (1.254)
    current = (crate_span * 10.0 * I_SPAN / w) / q_theo_per_gS * p.Q_theo / 3600.0   # A/m2
    prot = LiSPANProtocol(current=current, ramp_s=30.0)
    _, res = solve(p, prot, N_c=N[0], N_s=N[1], n_out=n_out)
    return res["Q_mAh_gS"] * w, res["V"], prot, res


def fit(q_d, v_d, base, x0, max_nfev, log, ck=None):
    n = [ck.prior_solves if ck else 0]
    t0 = time.time() - (ck.prior_wall_s if ck else 0.0)

    def resid(x):
        n[0] += 1
        Q, V, _, _ = simulate(unpack(x), base)
        r = (np.where(q_d <= Q[-1], np.interp(q_d, Q, V), 1.0) - v_d) * 1e3
        if ck is not None:
            ck.record(x, float(np.sqrt(np.mean(r ** 2))), n[0], time.time() - t0)
        log(f"  eval {n[0]:3d} ({time.time() - t0:5.0f} s) rms {np.sqrt(np.mean(r ** 2)):7.2f} mV | "
            + ", ".join(f"{k} {v:.4g}" for k, v in unpack(x).items()))
        return r

    lo = np.array([np.log(0.15), -10, -10, -10, -1.5, -1.5, -1.5])      # U0 offsets within +-1 V
    hi = np.array([np.log(0.8), 10, 10, 10, 1.5, 1.5, 1.5])
    fun, jac = least_squares_fd(resid, np.full(len(x0), 0.01), lo, hi)     # 1 % (log) / 1 mV (U0) absolute steps
    sol = least_squares(fun, x0, jac=jac, method="trf", x_scale=1.0, max_nfev=max_nfev, bounds=(lo, hi),
                        ftol=1e-4, xtol=1e-4)      # BDF solutions are ~1e-6 accurate: scipy's 1e-8 is never met
    dof = max(len(sol.fun) - len(x0), 1)
    s2 = float(np.sum(sol.fun ** 2) / dof)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
        sd, corr = np.sqrt(np.diag(cov)), cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    except np.linalg.LinAlgError:
        sd, corr = np.full(len(x0), np.nan), np.full((len(x0), len(x0)), np.nan)
    return sol, sd, corr, n[0], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=3)
    ap.add_argument("--check-cycle", type=int, default=2, help="second cycle evaluated with the fitted parameters")
    ap.add_argument("--q-min", type=float, default=5.0, help="mAh/g_SPAN; skips the initial relaxation spike")
    ap.add_argument("--n-points", type=int, default=150)
    ap.add_argument("--max-nfev", type=int, default=30)
    ap.add_argument("--hypotheses", nargs="*", default=list(HYPOTHESES))
    ap.add_argument("--resume", action="store_true", help="continue each hypothesis from results/lispan/span500/"
                    "ckpt_cycle<c>_<hyp>.json (written whenever the misfit improves)")
    ap.add_argument("--start", default=None, help='JSON {hypothesis: {w_S, U0_m (V offsets), b_m}} starting points')
    args = ap.parse_args()
    starts = json.loads(args.start) if args.start else {}
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"fit_cycle{args.cycle}.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    q_all, v_all = load_discharge(args.cycle)
    q_d = np.linspace(args.q_min, q_all[-1], args.n_points)
    v_d = np.interp(q_d, q_all, v_all)
    q_c, v_c = load_discharge(args.check_cycle)
    report = {"data": {"source": str(XLSX.relative_to(ROOT)), "cycle": args.cycle, "Q_end_mAh_gSPAN": float(q_all[-1]),
                       "points": len(q_d), "q_min": args.q_min, "check_cycle": args.check_cycle,
                       "check_Q_end_mAh_gSPAN": float(q_c[-1])},
              "model": {"form": "PINN benchmark (TFF, no double layer, 30 s ramp), Simanjuntak geometry/electrolyte",
                        "Z_CC_Ohm_m2": Z_CC_EIS, "grid": [20, 10]},
              "fits": {}}
    jpath = OUT / f"fit_cycle{args.cycle}.json"
    if jpath.exists():                     # keep fits of hypotheses not rerun now
        report["fits"] = json.loads(jpath.read_text()).get("fits", {})
    for hyp in args.hypotheses:
        base = base_params(HYPOTHESES[hyp])
        # start: nominal OCV parameters, w_S from the capacity at the nominal model
        Q0, _, _, _ = simulate({"w_S": 1.0, "U0_1": 0, "U0_2": 0, "U0_3": 0, "b_1": 1, "b_2": 1, "b_3": 1}, base)
        w0 = float(q_all[-1] / Q0[-1])
        x0 = np.array([np.log(w0), 0, 0, 0, 0, 0, 0], float)
        if hyp in starts:
            st = starts[hyp]
            x0 = np.array([np.log(st["w_S"]), *(st[f"U0_{m}"] / 0.1 for m in (1, 2, 3)),
                           *(np.log(st[f"b_{m}"]) for m in (1, 2, 3))], float)
        ck = FitCheckpoint(OUT / f"ckpt_cycle{args.cycle}_{hyp}.json")
        if args.resume and ck.state:
            x0 = ck.x()
            log(f"[{hyp}] resuming from {ck.path.name}: rms {ck.best:.2f} mV after {ck.prior_solves} solves")
        elif ck.state:
            ck.path.rename(ck.path.with_suffix(f".old{int(time.time())}.json"))
            ck = FitCheckpoint(ck.path)
        log(f"[{hyp}] k0 x{HYPOTHESES[hyp]:g}, start {unpack(x0)}")
        sol, sd, corr, nsol, wall = fit(q_d, v_d, base, x0, args.max_nfev, log, ck)
        vals = {k: float(v) for k, v in unpack(sol.x).items()}
        Q, V, prot, res = simulate(vals, base)
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        r_c = (np.where(q_c[q_c >= args.q_min] <= Q[-1], np.interp(q_c[q_c >= args.q_min], Q, V), 1.0)
               - v_c[q_c >= args.q_min]) * 1e3
        Q1, V1, prot1, _ = simulate(vals, base, crate_span=1.0)
        report["fits"][hyp] = {
            "k0_factor": HYPOTHESES[hyp], "k0": list(base.k0), "estimates": vals,
            "sd_internal": dict(zip(NAMES, sd.tolist())), "corr": corr.tolist(), "rms_mV": rms,
            "max_mV": float(np.abs(sol.fun).max()), "check_cycle_rms_mV": float(np.sqrt(np.mean(r_c ** 2))),
            "model_current_A_m2": prot.current, "model_Q_end_mAh_gSPAN": float(Q[-1]), "solves": nsol, "wall_s": wall,
            "U0_fitted_V": [u + d for u, d in zip(base.U0, (vals["U0_1"], vals["U0_2"], vals["U0_3"]))],
            "b_fitted_V": [b * m for b, m in zip(base.b, (vals["b_1"], vals["b_2"], vals["b_3"]))],
            "prediction_1C": {"Q_end_mAh_gSPAN": float(Q1[-1]), "V_at_Q": {f"{q:g}": float(np.interp(q, Q1, V1))
                                                                         for q in (50, 150, 250, 350)}},
        }
        log(f"[{hyp}] rms {rms:.2f} mV, check cycle {report['fits'][hyp]['check_cycle_rms_mV']:.2f} mV, "
            f"estimates {vals}")
        # PINN data file: time axis of the fitted model current (same form as lispan_make_inverse_data output)
        t_d = np.array([time_of_charge(q / vals["w_S"] * 3.6 * base.m_S_model * 1e3, prot.current, prot.ramp_s)
                        for q in q_d])
        np.savez(OUT / f"V_span500_c{args.cycle}_{hyp}.npz", t=t_d, V=v_d, I=np.full_like(t_d, prot.current),
                 Q_mAh_gSPAN=q_d, t_end_ref=float(res["t"][-1]), truth_json=json.dumps({}),
                 model_json=json.dumps({"current": prot.current, "ramp_s": prot.ramp_s, "Z_CC": Z_CC_EIS,
                                        "k0": list(base.k0), "fv_estimates": vals}))
        jpath.write_text(json.dumps(report, indent=1))
    report["data"]["cycle_to_cycle_rms_mV"] = float(1e3 * np.sqrt(np.mean(
        (np.interp(q_d, q_c, v_c) - v_d)[q_d <= q_c[-1]] ** 2)))
    jpath.write_text(json.dumps(report, indent=1))
    curves = {}
    for hyp, f in report["fits"].items():
        base = base_params(f["k0_factor"])
        Q, V, _, _ = simulate(f["estimates"], base)
        Q1, V1, _, _ = simulate(f["estimates"], base, crate_span=1.0)
        curves[hyp] = (Q, V, Q1, V1)
    plot(q_all, v_all, q_c, v_c, curves, report, OUT / f"fit_cycle{args.cycle}.png")


def plot(q_all, v_all, q_c, v_c, curves, report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    cyc, cc = report["data"]["cycle"], report["data"]["check_cycle"]
    axs[0].plot(q_all, v_all, "k", lw=2.5, alpha=0.35, label=f"SPAN500 cycle {cyc} (fitted)")
    axs[0].plot(q_c, v_c, "k:", lw=1, label=f"SPAN500 cycle {cc}")
    styles = {"paper": "C0", "fast": "C3"}
    for hyp, (Q, V, Q1, V1) in curves.items():
        f = report["fits"][hyp]
        axs[0].plot(Q, V, color=styles.get(hyp), lw=1.2,
                    label=f"model, {hyp} kinetics (k0 x{f['k0_factor']:g}): {f['rms_mV']:.1f} mV rms")
        axs[1].plot(Q, V, color=styles.get(hyp), lw=1.0, ls="--")
        axs[1].plot(Q1, V1, color=styles.get(hyp), lw=1.5, label=f"{hyp} kinetics: C/10 (dashed), 1 C prediction")
        axs[0].plot([], [])
    for ax in axs:
        ax.set_xlabel("capacity [mAh / g SPAN]"); ax.set_ylabel("cell voltage [V]"); ax.set_ylim(0.95, 2.6)
        ax.grid(alpha=0.3)
    axs[0].set_title("calibration at C/10")
    axs[1].set_title("same C/10 fit, different 1 C predictions")
    axs[0].legend(fontsize=7); axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    sys.exit(main())
