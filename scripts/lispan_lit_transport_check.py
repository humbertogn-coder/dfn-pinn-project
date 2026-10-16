"""Check (Phase 2 pilot, finding 3): can slower electrolyte transport in the Li-SPAN model reproduce the
rate-dependent capacity loss of the Ma et al. (2026) SPAN-aerogel cells?

Takes the parameters of the 0.2 + 1 C calibration (results/lispan/literature_rates/ma2026_SPAN-A_fit0.2+1.json),
multiplies the salt diffusivity and the electrolyte conductivity, and reports the end-of-discharge capacity and the
voltage at 300 mAh/g at 0.2, 1 and 2 C. Solver failures are recorded as such.

    python scripts/lispan_lit_transport_check.py
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import lispan_lit_rates as L  # noqa: E402

FIT = ROOT / "results" / "lispan" / "literature_rates" / "ma2026_SPAN-A_fit0.2+1.json"
CASES = [(1.0, 1.0), (0.1, 1.0), (0.03, 1.0), (1.0, 0.3), (0.1, 0.3)]
DATA_Q_END = {0.2: 1248.0, 1.0: 686.0, 2.0: 454.0}   # digitized end capacities (data/literature_span/ma2026)


def main():
    est = json.loads(FIT.read_text())["estimates"]
    base0 = L.BASE
    out = {"parameters_from": FIT.name, "data_q_end_mAh_g": DATA_Q_END, "cases": []}
    for d_mult, k_mult in CASES:
        L.BASE = replace(base0, D_salt=base0.D_salt * d_mult, kappa0=base0.kappa0 * k_mult)
        row = {"D_salt_x": d_mult, "kappa0_x": k_mult}
        for crate in (0.2, 1.0, 2.0):
            try:
                res = L.simulate(est, 1.675 * crate, N=(10, 5), rtol0=1e-5)
            except Exception as e:  # e.g. singular Jacobian when the salt is depleted
                res = None
                row[f"{crate:g}C_error"] = type(e).__name__
            if res is None:
                row[f"{crate:g}C"] = "solver failure"
                continue
            Q, V = res
            row[f"{crate:g}C"] = {"q_end_mAh_g": round(float(Q[-1]), 1),
                                  "V_at_300_mAh_g": round(float(np.interp(300.0, Q, V)), 4) if Q[-1] > 300 else None}
        out["cases"].append(row)
        print(json.dumps(row), flush=True)
    L.BASE = base0
    path = ROOT / "results" / "lispan" / "literature_rates" / "transport_check_ma2026_SPAN-A.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
