"""Tabulate final metrics of several v2 runs (e.g. a seed study).

    python scripts/v2_summarize_runs.py results/v2_runs/seed*_*
"""

import json
import sys
from pathlib import Path

import numpy as np

KEYS = [("V_rmse_mV", "V rmse [mV]"), ("V_max_mV", "V max [mV]"), ("ce_rmse", "c_e rmse"),
        ("ce_max", "c_e max"), ("phie_max_mV", "phi_e max [mV]"), ("j_n_rmse_rel", "j_n rmse/jref"),
        ("j_p_rmse_rel", "j_p rmse/jref"), ("thsurf_n_rmse", "th_s,n rmse"), ("thsurf_p_rmse", "th_s,p rmse"),
        ("thsurf_n_max", "th_s,n max"), ("thsurf_p_max", "th_s,p max")]


def main(runs):
    rows = []
    for r in runs:
        r = Path(r)
        f = r / "final_metrics.json"
        if not f.exists():
            print(f"skip {r.name}: no final_metrics.json")
            continue
        m = json.loads(f.read_text())
        cfg = json.loads((r / "config.json").read_text())["train"]
        rows.append((r.name, cfg.get("seed"), m))
    if not rows:
        return
    print(f"{'run':40s} {'seed':>4s} " + " ".join(f"{lab:>14s}" for _, lab in KEYS))
    for name, seed, m in rows:
        print(f"{name[:40]:40s} {str(seed):>4s} " + " ".join(f"{m[k]:14.4g}" for k, _ in KEYS))
    if len(rows) > 1:
        arr = np.array([[m[k] for k, _ in KEYS] for _, _, m in rows])
        print(f"{'mean':40s} {'':>4s} " + " ".join(f"{v:14.4g}" for v in arr.mean(0)))
        print(f"{'std':40s} {'':>4s} " + " ".join(f"{v:14.4g}" for v in arr.std(0, ddof=1)))


if __name__ == "__main__":
    main(sys.argv[1:])
