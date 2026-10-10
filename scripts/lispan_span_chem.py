"""Chemical-information study on the Nat. Mater. 2026 SPAN data (Wang et al., Nat. Mater. 25, 791 (2026)).

Same calibration as scripts/lispan_span500.py ("paper" kinetics, Simanjuntak geometry/transport, Z_CC from the EIS
high-frequency intercept, fitted w_S, U0_1..3 offsets, b_1..3 multipliers), applied to more data sets:

  fig4a:<cycle>              SPAN500, cycles 1-3, C/10 source data (Fig. 4a)               -> first-cycle activation
  si21:<material>:<cycle>    SPAN300 / SPAN320 / SPAN500, cycles 1-2, digitized from Supplementary Fig. 21
                             (scripts/digitize_natmater_si_fig21.py)                         -> synthesis temperature

For each data set ("label") this script
  1. fits the FV model to the whole discharge (fit_<label>.json),
  2. writes the PINN data file for the window Q <= 0.95 x the FV end of discharge (V_<label>_pinnwindow.npz) and
     the PINN configs (stage 1 forward re-adaptation, inverse) with t_end = 0.98 x the FV discharge time,
  3. fits the FV model again on exactly that window with w_S fixed (the like-for-like reference of the PINN,
     fit_<label>_window.json).

    python scripts/lispan_span_chem.py --label s500_c1 --source fig4a:1 --zcc-ohm 1.93
    python scripts/lispan_span_chem.py --label si300_c2 --source si21:SPAN300:2 --zcc-ohm 8.33
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

from dfn_pinn.lispan.fdjac import FitCheckpoint, least_squares_fd  # noqa: E402
from lispan_make_inverse_data import time_of_charge  # noqa: E402
import lispan_span500 as sp5  # noqa: E402

OUT = ROOT / "results" / "lispan" / "span_chem"
DIGI = ROOT / "data" / "span_natmater2026" / "digitized"
AREA_M2 = 1.131e-4          # 12 mm disc
NAMES6 = ["U0_1", "U0_2", "U0_3", "b_1", "b_2", "b_3"]


def load(source):
    kind, *rest = source.split(":")
    if kind == "fig4a":
        q, v = sp5.load_discharge(int(rest[0]))
        return q, v, f"{sp5.XLSX.relative_to(ROOT)} sheet Fig.4a, cycle {rest[0]} (source data)"
    if kind == "si21":
        f = DIGI / f"si_fig21_{rest[0]}_c{rest[1]}.csv"
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        return d[:, 0], d[:, 1], f"{f.relative_to(ROOT)} (digitized from Supplementary Fig. {21}, cycle {rest[1]})"
    raise ValueError(source)


def window_fit(q_d, v_d, w_S, start, base, ck_path, max_nfev):
    ck = FitCheckpoint(ck_path)
    n, t0 = [0], time.time()

    def vals(x):
        return {"w_S": w_S, **{k: (0.1 * v if k.startswith("U0") else float(np.exp(v))) for k, v in zip(NAMES6, x)}}

    def resid(x):
        n[0] += 1
        Q, V, _, _ = sp5.simulate(vals(x), base)
        r = (np.where(q_d <= Q[-1], np.interp(q_d, Q, V), 1.0) - v_d) * 1e3
        ck.record(x, float(np.sqrt(np.mean(r ** 2))), n[0], time.time() - t0)
        return r

    x0 = np.array([start[k] / 0.1 if k.startswith("U0") else np.log(start[k]) for k in NAMES6])
    lo, hi = np.array([-10.0] * 3 + [-1.5] * 3), np.array([10.0] * 3 + [1.5] * 3)
    fun, jac = least_squares_fd(resid, np.full(6, 0.01), lo, hi)
    sol = least_squares(fun, x0, jac=jac, method="trf", max_nfev=max_nfev, bounds=(lo, hi), ftol=1e-4, xtol=1e-4)
    return vals(sol.x), float(np.sqrt(np.mean(sol.fun ** 2))), n[0], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--zcc-ohm", type=float, required=True, help="EIS high-frequency intercept of the 12 mm cell")
    ap.add_argument("--q-min", type=float, default=5.0)
    ap.add_argument("--n-points", type=int, default=150)
    ap.add_argument("--max-nfev", type=int, default=30)
    ap.add_argument("--init-run", default="results/lispan_runs/lispan_final_0.1C_seed0_20261008T005207Z/final.pt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    q_all, v_all, src = load(args.source)
    z_cc = args.zcc_ohm * AREA_M2
    q_d = np.linspace(args.q_min, q_all[-1], args.n_points)
    v_d = np.interp(q_d, q_all, v_all)
    base = replace(sp5.base_params(1.0), Z_CC=z_cc)
    logf = open(OUT / f"fit_{args.label}.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    # 1. full-curve fit
    Q0, _, _, _ = sp5.simulate({"w_S": 1.0, "U0_1": 0, "U0_2": 0, "U0_3": 0, "b_1": 1, "b_2": 1, "b_3": 1}, base)
    x0 = np.array([np.log(q_all[-1] / Q0[-1]), 0, 0, 0, 0, 0, 0], float)
    ck = FitCheckpoint(OUT / f"ckpt_{args.label}.json")
    log(f"[{args.label}] {src}; Z_CC {z_cc:.3e} Ohm m2; Q_end {q_all[-1]:.1f} mAh/g_SPAN; start w_S {np.exp(x0[0]):.3f}")
    sol, sd, corr, nsol, wall = sp5.fit(q_d, v_d, base, x0, args.max_nfev, log, ck)
    est = {k: float(v) for k, v in sp5.unpack(sol.x).items()}
    Q, V, prot, res = sp5.simulate(est, base)
    rep = {"label": args.label, "source": args.source, "data": src, "Q_end_mAh_gSPAN": float(q_all[-1]),
           "points": int(q_d.size), "q_min": args.q_min, "Z_CC_Ohm_m2": z_cc, "zcc_ohm_cell": args.zcc_ohm,
           "kinetics": "Simanjuntak effective k0 ('paper')", "estimates": est,
           "sd_internal": dict(zip(sp5.NAMES, sd.tolist())), "rms_mV": float(np.sqrt(np.mean(sol.fun ** 2))),
           "max_mV": float(np.abs(sol.fun).max()), "solves": nsol, "wall_s": wall,
           "U0_fitted_V": [u + d for u, d in zip(base.U0, (est["U0_1"], est["U0_2"], est["U0_3"]))],
           "b_fitted_V": [b * m for b, m in zip(base.b, (est["b_1"], est["b_2"], est["b_3"]))],
           "model_current_A_m2": prot.current, "model_Q_end_mAh_gSPAN": float(Q[-1]), "t_end_model_s": float(res["t"][-1])}
    np.savez(OUT / f"curve_{args.label}.npz", Q=Q, V=V, q_data=q_all, v_data=v_all)
    log(f"[{args.label}] full curve: rms {rep['rms_mV']:.2f} mV, {est}")

    # 2. PINN window, data file and configs
    keep = q_d <= 0.95 * Q[-1]
    qw, vw = q_d[keep], v_d[keep]
    t_w = np.array([time_of_charge(q / est["w_S"] * 3.6 * base.m_S_model * 1e3, prot.current, prot.ramp_s) for q in qw])
    t_end = 0.98 * float(res["t"][-1])
    data_path = OUT / f"V_{args.label}_pinnwindow.npz"
    np.savez(data_path, t=t_w, V=vw, I=np.full_like(t_w, prot.current), Q_mAh_gSPAN=qw, t_end_ref=float(res["t"][-1]),
             truth_json=json.dumps({}),
             model_json=json.dumps({"current": prot.current, "ramp_s": prot.ramp_s, "Z_CC": z_cc, "k0": list(base.k0),
                                    "fv_estimates": est, "window": "Q <= 0.95 x FV end of discharge"}))
    st1 = json.loads((ROOT / "configs" / "lispan_forward_span500_stage1.json").read_text())
    st1["name"] = f"lispan_chem_{args.label}_stage1"
    st1["description"] = (f"Stage 1 (forward re-adaptation, nominal parameters, no data) for the PINN inverse on {src}; "
                          f"current {prot.current:.4f} A/m2 from the FV fit (w_S {est['w_S']:.4f}), Z_CC from the EIS "
                          f"high-frequency intercept ({args.zcc_ohm} Ohm).")
    st1["params"]["Z_CC"] = z_cc
    st1["protocol"]["current"] = prot.current
    st1["t_end"] = t_end
    (ROOT / "configs" / f"lispan_chem_{args.label}_stage1.json").write_text(json.dumps(st1, indent=2))
    inv = json.loads((ROOT / "configs" / "lispan_inverse_span500_paper_wS.json").read_text())
    inv["name"] = f"lispan_chem_{args.label}_inv"
    inv["description"] = (f"PINN inverse (U0_1..3, b_1..3; trust-region LM, physics-only fields) on {src}; same "
                          f"protocol as lispan_inverse_span500_paper_wS.json (cycle 3), with param_log_bound 8 "
                          f"(U0 offsets within +-0.8 V) and gn_max_step 1.0 (0.1 V per trust-region step on U0). init = stage-1 final.pt (filled by scripts/run_chem_queue.sh).")
    inv["params"]["Z_CC"] = z_cc
    inv["rates"][0].update({"current": prot.current, "data_path": str(data_path.relative_to(ROOT)), "t_end": t_end,
                            "init": "FILL"})
    inv["train"]["param_log_bound"] = 8.0
    inv["train"]["gn_max_step"] = 1.0          # 0.1 V per trust-region step on U0 (see scripts/run_chem_queue.sh)
    (ROOT / "configs" / f"lispan_chem_{args.label}_inv.json").write_text(json.dumps(inv, indent=2))

    # 3. FV fit on the PINN window, w_S fixed
    westim, wrms, wn, wwall = window_fit(qw, vw, est["w_S"], est, base, OUT / f"ckpt_{args.label}_window.json",
                                         args.max_nfev)
    rep["window"] = {"points": int(qw.size), "q_max_mAh_gSPAN": float(qw[-1]), "t_end_s": t_end, "estimates": westim,
                     "rms_mV": wrms, "solves": wn, "wall_s": wwall}
    (OUT / f"fit_{args.label}.json").write_text(json.dumps(rep, indent=1))
    log(f"[{args.label}] window ({qw.size} pts, Q <= {qw[-1]:.0f}): rms {wrms:.2f} mV, {westim}")


if __name__ == "__main__":
    sys.exit(main())
