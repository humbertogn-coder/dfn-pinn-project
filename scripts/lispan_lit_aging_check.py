"""Exploratory check: what can one-rate cycling data say about why a SPAN cell ages?

Data: Hao et al. (2025) SI Fig. S12b, plain SPAN cycled at 0.5 C, discharge curves of cycles 2, 10 and 100
(data/literature_span/hao2025, dataset hao2025_SPAN_cycling in lispan_lit_rates.py). Cycle 10 is the reference
(activated cell; cycle 2 is still activating).

Base parameters: the calibration of the same material on its rate test (SI Fig. S9, all five rates,
results/lispan/literature_rates/hao2025_SPAN_fit0.1+0.2+0.5+1+2.json). Each cycle is then fitted at 0.5 C under two
two-parameter hypotheses that both include a loss of active sulfur (w_S):
  "resistance": w_S and the contact/ohmic resistance Z_CC change;
  "kinetics":   w_S and the kinetics of the irreversible steps 2 and 3 change (a common shift dU of U0_2 and U0_3,
                which for these steps is the same as a change of k0 by exp(-F dU / 2RT); docs/PHASE1_IDENTIFIABILITY.md).
Each fitted hypothesis then predicts the same cycle at 0.1 C and 2 C. If both hypotheses fit the 0.5 C curve equally
well but predict different 2 C curves, one-rate cycling data cannot tell them apart and a periodic check-up at a
second rate can.

    python scripts/lispan_lit_aging_check.py
Output: results/lispan/literature_rates/aging_check_hao2025.json / .npz
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import lispan_lit_rates as L  # noqa: E402
from dfn_pinn.lispan.fdjac import least_squares_fd  # noqa: E402

RES = ROOT / "results" / "lispan" / "literature_rates"
BASE_FITS = ["hao2025_SPAN_fit0.1+0.2+0.5+1+2.json", "hao2025_SPAN_fit0.2+2.json", "hao2025_SPAN_fit0.1+2.json"]
CYCLES = [10.0, 2.0, 100.0]
I_1C = 0.71                        # A/g SPAN (lispan_lit_rates.DATASETS["hao2025_SPAN"])
N, RTOL = (10, 5), 1e-5


def vals_for(base, hyp, x):
    v = dict(base)
    v["w_S"] = float(np.exp(x[0]))
    if hyp == "resistance":
        v["Z_CC"] = base["Z_CC"] * float(np.exp(x[1]))
    else:
        v["U0_2"] = base["U0_2"] + 0.1 * x[1]
        v["U0_3"] = base["U0_3"] + 0.1 * x[1]
    return v


def main():
    base_file = next(RES / f for f in BASE_FITS if (RES / f).exists())
    base = json.loads(base_file.read_text())["estimates"]
    out = {"base_parameters_from": base_file.name, "i_1C_A_per_g": I_1C, "cycles": {}}
    curves = {}
    for cyc in CYCLES:
        d = L.load("hao2025_SPAN_cycling", cyc)
        row = {"q_end_data": d["q_end"]}
        for hyp in ("resistance", "kinetics"):
            t0 = time.time()

            def resid(x):
                vm, _ = L.model_on(vals_for(base, hyp, x), d, N=N, rtol=RTOL)
                return np.where(np.isfinite(vm), vm - d["v"], 1.0) * 1e3

            lo, hi = np.array([np.log(0.2), -5.0]), np.array([np.log(3.0), 5.0])
            x0 = np.array([np.log(base["w_S"] * d["q_end"] / 700.0), 0.0])
            fun, jac = least_squares_fd(resid, np.full(2, 0.01), lo, hi)     # absolute FD steps (fdjac.py)
            sol = least_squares(fun, x0, jac=jac, method="trf", x_scale=1.0, max_nfev=30, ftol=1e-4, xtol=1e-4,
                                bounds=(lo, hi))
            est = vals_for(base, hyp, sol.x)
            r = {"rms_fit_0.5C_mV": float(np.sqrt(np.mean(sol.fun ** 2))), "w_S": est["w_S"],
                 "Z_CC_ohm_m2": 0.025 * est["Z_CC"], "dU_steps23_mV": 1e3 * (est["U0_2"] - base["U0_2"]),
                 "wall_s": time.time() - t0}
            for crate in (0.1, 0.5, 2.0):
                res = L.simulate(est, I_1C * crate, N=(20, 10), rtol0=1e-6)
                if res is None:
                    r[f"{crate:g}C"] = "solver failure"
                    continue
                Q, V = res
                r[f"{crate:g}C"] = {"q_end": round(float(Q[-1]), 1),
                                    "V_at_300": round(float(np.interp(300.0, Q, V)), 4) if Q[-1] > 300 else None}
                curves[f"Q_c{cyc:g}_{hyp}_{crate:g}C"], curves[f"V_c{cyc:g}_{hyp}_{crate:g}C"] = Q, V
            row[hyp] = r
            print(f"cycle {cyc:g} {hyp}: {json.dumps(r)}", flush=True)
        curves[f"qd_c{cyc:g}"], curves[f"vd_c{cyc:g}"] = d["q_raw"], d["v_raw"]
        a, b = row["resistance"], row["kinetics"]
        if isinstance(a.get("2C"), dict) and isinstance(b.get("2C"), dict) and a["2C"]["V_at_300"] and b["2C"]["V_at_300"]:
            row["2C_difference_between_hypotheses"] = {
                "V_at_300_mV": round(1e3 * (a["2C"]["V_at_300"] - b["2C"]["V_at_300"]), 1),
                "q_end_mAh_g": round(a["2C"]["q_end"] - b["2C"]["q_end"], 1)}
        out["cycles"][f"{cyc:g}"] = row
    (RES / "aging_check_hao2025.json").write_text(json.dumps(out, indent=1))
    np.savez(RES / "aging_check_hao2025.npz", **curves)
    print("wrote", RES / "aging_check_hao2025.json")


if __name__ == "__main__":
    main()
