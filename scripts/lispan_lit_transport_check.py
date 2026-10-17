"""Check (Phase 2 pilot, finding 3): can slower electrolyte transport in the Li-SPAN model reproduce the
rate-dependent capacity loss of a published SPAN cell?

Takes the parameters of a two-rate calibration (default: Ma et al. (2026) SPAN aerogel, 0.2 + 1 C,
results/lispan/literature_rates/ma2026_SPAN-A_fit0.2+1.json), multiplies the salt diffusivity and the electrolyte
conductivity, and reports the end-of-discharge capacity and the voltage at 300 mAh/g at the chosen rates. Solver
failures are recorded as such.

    python scripts/lispan_lit_transport_check.py                       # Ma 2026 SPAN-A (original check)
    python scripts/lispan_lit_transport_check.py --fit-json hao2025_SPAN_fit0.2+2.json --i-1c 0.71 \
        --crates 0.2 1 2 --q-end 0.2=694.7 1=623.2 2=592.2 --out transport_check_hao2025_SPAN.json
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import lispan_lit_rates as L  # noqa: E402

RES = ROOT / "results" / "lispan" / "literature_rates"
CASES = [(1.0, 1.0), (0.1, 1.0), (0.03, 1.0), (1.0, 0.3), (0.1, 0.3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-json", default="ma2026_SPAN-A_fit0.2+1.json")
    ap.add_argument("--i-1c", type=float, default=1.675, help="current of 1 C in A per g of the data basis")
    ap.add_argument("--crates", nargs="+", type=float, default=[0.2, 1.0, 2.0])
    ap.add_argument("--q-end", nargs="*", default=["0.2=1248", "1=686", "2=454"],
                    help="digitized end capacities, rate=mAh/g (data/literature_span)")
    ap.add_argument("--cases", nargs="*", default=None, help="D_mult,kappa_mult pairs, e.g. 1,1 0.3,1")
    ap.add_argument("--out", default="transport_check_ma2026_SPAN-A.json")
    args = ap.parse_args()
    cases = CASES if args.cases is None else [tuple(float(v) for v in c.split(",")) for c in args.cases]
    est = json.loads((RES / args.fit_json).read_text())["estimates"]
    base0 = L.BASE
    q_end = {float(k): float(v) for k, v in (s.split("=") for s in args.q_end)}
    out = {"parameters_from": args.fit_json, "i_1C_A_per_g": args.i_1c, "data_q_end_mAh_g": q_end, "cases": []}
    for d_mult, k_mult in cases:
        L.BASE = replace(base0, D_salt=base0.D_salt * d_mult, kappa0=base0.kappa0 * k_mult)
        row = {"D_salt_x": d_mult, "kappa0_x": k_mult}
        for crate in args.crates:
            try:
                res = L.simulate(est, args.i_1c * crate, N=(10, 5), rtol0=1e-5)
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
    path = RES / args.out
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
