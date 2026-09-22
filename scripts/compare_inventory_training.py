"""Compare a projected particle run with its matched unprojected control."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_flux_residual import load_run, resolve_run, physical_quadrature, residual_components, volume_time_rms
from train_constant_flux_particle import assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output", type=Path, help="New comparison directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    projected = resolve_run(root/"results", "inventory", args.projected, "full_radius")
    control = resolve_run(root/"results", "startup", args.control, "full_radius")
    a, b = projected[1], control[1]
    for key in ("seed", "adam_steps", "lbfgs_steps"):
        if a["config"][key] != b["config"][key]:
            raise ValueError(f"Unmatched configuration: {key}; use explicit run paths")
    if a["loss_weights"] != b["loss_weights"] or a["initial_weights_sha256"] != b["initial_weights_sha256"]:
        raise ValueError("Unmatched losses or initialization")
    with np.load(projected[0]/"training_points.npz") as left, np.load(control[0]/"training_points.npz") as right:
        for key in ("interior", "initial", "surface"):
            if not np.array_equal(left[key], right[key]):
                raise ValueError(f"Training points differ: {key}")
    torch.set_num_threads(1)
    entries = {}
    for name, (path, report) in (("control", control), ("projected", projected)):
        print(f"Verifying {name}: {path}", flush=True)
        model, _, _, gap = load_run(path, report, report["config"]["variant"])
        metrics, _ = assess(model)
        for key, value in metrics.items():
            if not np.isclose(value, report["metrics"][key], atol=1e-12, rtol=1e-8):
                raise ValueError(f"Metric failed reproduction: {key}")
        weighted = []
        for nr, nt in ((64, 4), (128, 8)):
            r, w, t, wt = physical_quadrature(nr, nt)
            rr, tt = np.meshgrid(r, t, indexing="ij")
            residual = residual_components(model, rr.ravel(), tt.ravel(), batch_size=64)[2].reshape(rr.shape)
            weighted.append(volume_time_rms(residual, r, w, wt))
        metrics["physical_pde_rms"] = weighted[1]
        entries[name] = {"run": str(path.resolve()), "metrics": metrics,
                         "model_sha256": hashlib.sha256((path/"model.pt").read_bytes()).hexdigest(),
                         "report_sha256": hashlib.sha256((path/"report.json").read_bytes()).hexdigest(),
                         "reload_gap": gap, "physical_pde_coarse": weighted[0],
                         "physical_pde_quadrature_gap": abs(weighted[0]-weighted[1])}
    keys = ("positive_time_max_error", "positive_time_rms_error", "max_surface_flux_residual",
            "max_volume_mean_balance_error", "positive_time_pde_rms", "physical_pde_rms")
    for key in keys:
        print(f"{key}: control={entries['control']['metrics'][key]:.6e}, "
              f"projected={entries['projected']['metrics'][key]:.6e}")
    output = args.output or root/"results"/("inventory_comparison_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    payload = {"runs": entries, "initial_weights_and_original_points_match": True,
               "scope": "One exploratory seed, unchanged optimizer budgets, not equal computational cost. "
                        "All saved metrics reproduced. Mean quadratures 128/256 are independent of projection order 64. "
                        "No continuous-domain bound, positivity guarantee or multi-seed acceptance."}
    (output/"report.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"No training was run. Output directory: {output}")


if __name__ == "__main__":
    main()
