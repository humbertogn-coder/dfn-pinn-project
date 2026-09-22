"""Reproduce four saved particle candidates and report multi-objective tradeoffs."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_flux_residual import (
    load_run, physical_quadrature, residual_components, resolve_run, volume_time_rms,
)
from train_constant_flux_particle import assess


OBJECTIVES = (
    "initial_max_error", "positive_time_max_error", "positive_time_rms_error",
    "max_surface_flux_residual", "max_volume_mean_balance_error", "volume_time_pde_rms",
)


def nondominated(values):
    """All objectives are minimized; ties do not dominate each other."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("Require a nonempty finite objective matrix")
    return [not any(np.all(other <= row) and np.any(other < row) for other in values)
            for row in values]


def require_comparable(reports):
    first = reports[0]
    for report in reports[1:]:
        for key in ("seed", "adam_steps", "lbfgs_steps"):
            if report["config"][key] != first["config"][key]:
                raise ValueError(f"Candidate settings differ: {key}")
        if report["loss_weights"] != first["loss_weights"]:
            raise ValueError("Candidate loss weights differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "legacy", "full_radius", "surface_layer"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidates = {}
    for name in ("baseline", "legacy", "full_radius", "surface_layer"):
        variant = "baseline" if name == "baseline" else "startup"
        sampling = "legacy" if name == "baseline" else name
        candidates[name] = resolve_run(root/"results", variant, getattr(args, name), sampling)
    require_comparable([report for _, report in candidates.values()])
    torch.set_num_threads(1)
    entries = []
    for name, (path, report) in candidates.items():
        print(f"Verifying {name}: {path}", flush=True)
        variant = "baseline" if name == "baseline" else "startup"
        model, _, _, reload_gap = load_run(path, report, variant)
        metrics, _ = assess(model)
        for key, value in metrics.items():
            if not np.isclose(value, report["metrics"][key], rtol=1e-8, atol=1e-12):
                raise ValueError(f"Saved metric was not reproduced: {name}/{key}")
        weighted = []
        for nr, nt in ((64, 4), (128, 8)):
            rho, wr, tau, wt = physical_quadrature(nr, nt)
            rr, tt = np.meshgrid(rho, tau, indexing="ij")
            residual = residual_components(model, rr.ravel(), tt.ravel())[2].reshape(rr.shape)
            weighted.append(volume_time_rms(residual, rho, wr, wt))
        metrics["volume_time_pde_rms"] = weighted[1]
        entries.append({"candidate": name, "run": str(path.resolve()), "config": report["config"],
                        "model_sha256": hashlib.sha256((path/"model.pt").read_bytes()).hexdigest(),
                        "report_sha256": hashlib.sha256((path/"report.json").read_bytes()).hexdigest(),
                        "prediction_reload_max_difference": reload_gap, "metrics": metrics,
                        "quadrature_coarse": weighted[0],
                        "quadrature_absolute_gap": abs(weighted[1]-weighted[0])})
    frontier = nondominated([[entry["metrics"][key] for key in OBJECTIVES] for entry in entries])
    for entry, keep in zip(entries, frontier):
        entry["nondominated"] = keep
        print(f"{entry['candidate']:14s} nondominated={keep} "
              f"max_c={entry['metrics']['positive_time_max_error']:.6e} "
              f"balance={entry['metrics']['max_volume_mean_balance_error']:.6e} "
              f"weighted_PDE={entry['metrics']['volume_time_pde_rms']:.6e}")
    output = root/"results"/("flux_candidate_comparison_"+
                             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    payload = {"objectives": OBJECTIVES, "candidates": entries,
               "scope": "Exploratory single-seed comparison, not acceptance or certification. "
                        "Exact numeric dominance on selected sampled metrics, not statistical significance. "
                        "Weighted PDE integrates [1e-5,0.02]; no claim for earlier positive times. "
                        "Two quadratures are a sensitivity check, not a rigorous error bound. "
                        "No retraining or automatic winner selection."}
    (output/"report.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    with (output/"comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate", "nondominated", *OBJECTIVES])
        writer.writeheader()
        for entry in entries:
            writer.writerow({"candidate": entry["candidate"], "nondominated": entry["nondominated"],
                             **{key: entry["metrics"][key] for key in OBJECTIVES}})
    print(f"No training was run. Output directory: {output}")


if __name__ == "__main__":
    main()
