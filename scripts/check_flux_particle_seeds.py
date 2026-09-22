"""Fixed paired five-seed study of startup legacy and full-radius sampling."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from compare_flux_candidates import OBJECTIVES
from diagnose_flux_residual import load_run, physical_quadrature, residual_components, volume_time_rms


SEEDS = (0, 1, 2, 3, 42)
SAMPLERS = ("legacy", "full_radius")


def describe(values):
    return {"median": statistics.median(values), "minimum": min(values),
            "maximum": max(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def summarize(rows):
    completed = [row for row in rows if row["status"] == "completed"]
    pairs = []
    for seed in SEEDS:
        pair = {row["sampling"]: row for row in completed if row["seed"] == seed}
        if all(name in pair for name in SAMPLERS):
            if pair["legacy"]["initial_weights_sha256"] != pair["full_radius"]["initial_weights_sha256"]:
                raise ValueError("Paired initial weights differ")
            pairs.append({"seed": seed, **{key: pair["full_radius"][key]-pair["legacy"][key]
                                           for key in OBJECTIVES}})
    per_sampler = {}
    for name in SAMPLERS:
        subset = [row for row in completed if row["sampling"] == name]
        per_sampler[name] = {"completed": len(subset), "statistics": {
            key: describe([row[key] for row in subset]) for key in OBJECTIVES} if subset else {}}
    paired_stats = {}
    for key in OBJECTIVES:
        values = [pair[key] for pair in pairs]
        if values:
            paired_stats[key] = {**describe(values),
                                 "full_radius_lower": sum(value < 0 for value in values),
                                 "legacy_lower": sum(value > 0 for value in values),
                                 "ties": sum(value == 0 for value in values)}
    return {"requested": 10, "attempted": len(rows), "completed": len(completed),
            "failed": len(rows)-len(completed), "complete_pairs": len(pairs),
            "per_sampler": per_sampler, "paired_differences_full_minus_legacy": pairs,
            "paired_statistics": paired_stats}


def validate_report(report, seed, sampling):
    for key, expected in {"seed": seed, "variant": "startup", "sampling": sampling,
                          "adam_steps": 2000, "lbfgs_steps": 300}.items():
        if report["config"][key] != expected:
            raise ValueError(f"Unexpected configuration: {key}")
    for key, expected in {"dtype": "float64", "device": "cpu", "threads": 1,
                          "interior_points": 512, "initial_points": 101,
                          "surface_points": 128, "initial_condition": "hard"}.items():
        if report[key] != expected:
            raise ValueError(f"Unexpected setting: {key}")
    if report["loss_weights"] != {"pde": 1, "initial": 100, "surface": 1}:
        raise ValueError("Unexpected loss weights")
    if not all(math.isfinite(value) for value in report["metrics"].values()):
        raise ValueError("Nonfinite metric")


def main():
    root = Path(__file__).resolve().parents[1]
    trainer = root/"scripts/train_constant_flux_particle.py"
    output = root/"results"/("flux_particle_seeds_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((root/"scripts").glob("*.py")) + sorted((root/"src/dfn_pinn").glob("*.py"))
    provenance = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    scope = ("Exploratory five predetermined seeds after seed-42 sampler exploration. "
             "No seed selection, acceptance threshold, significance or DFN accuracy claim. "
             "Seeds vary both initialization and collocation. Statistics exclude failures; "
             "paired statistics require both runs. Negative differences favor full_radius. "
             "Physical PDE RMS covers dimensionless time [1e-5,0.02], not earlier times.")
    rows = []

    def save():
        summary = summarize(rows)
        summary.update(seeds=SEEDS, source_sha256=provenance, runs=rows, scope=scope)
        (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
        return summary

    save()
    torch.set_num_threads(1)
    print(f"Paired study: seeds={SEEDS}; samplers={SAMPLERS}", flush=True)
    print(f"Output directory: {output}", flush=True)
    for seed in SEEDS:
        for sampling in SAMPLERS:
            run = output/f"seed_{seed}_{sampling}"
            log = output/f"seed_{seed}_{sampling}.log"
            row = {"seed": seed, "sampling": sampling, "status": "failed", "error": "", "output": str(run)}
            print(f"Training seed {seed}, {sampling}...", flush=True)
            started = time.perf_counter()
            try:
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run([sys.executable, str(trainer), "--variant", "startup",
                        "--sampling", sampling, "--seed", str(seed), "--adam-steps", "2000",
                        "--lbfgs-steps", "300", "--output", str(run)], cwd=root,
                        stdout=stream, stderr=subprocess.STDOUT, timeout=1800, check=False)
                if result.returncode:
                    raise RuntimeError(f"Trainer exited with {result.returncode}; see {log.name}")
                report = json.loads((run/"report.json").read_text(encoding="utf-8"))
                validate_report(report, seed, sampling)
                model, _, _, gap = load_run(run, report, "startup")
                weighted = []
                for nr, nt in ((64, 4), (128, 8)):
                    rho, wr, tau, wt = physical_quadrature(nr, nt)
                    rr, tt = np.meshgrid(rho, tau, indexing="ij")
                    residual = residual_components(model, rr.ravel(), tt.ravel())[2].reshape(rr.shape)
                    weighted.append(volume_time_rms(residual, rho, wr, wt))
                initial_hash = report["initial_weights_sha256"]
                prior = next((item for item in rows if item["seed"] == seed and item["status"] == "completed"), None)
                if prior and prior["initial_weights_sha256"] != initial_hash:
                    raise ValueError("Paired initial weights differ")
                row.update(report["metrics"])
                row.update(status="completed", volume_time_pde_rms=weighted[1],
                    quadrature_absolute_gap=abs(weighted[1]-weighted[0]),
                    initial_weights_sha256=initial_hash, prediction_reload_max_difference=gap,
                    model_sha256=hashlib.sha256((run/"model.pt").read_bytes()).hexdigest(),
                    training_s=report["training_s"])
            except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                row["error"] = str(error)
            row["process_wall_s"] = time.perf_counter()-started
            rows.append(row)
            save()
            print(f"Seed {seed}, {sampling}: {row['status']}"+
                  (f", max_c={row['positive_time_max_error']:.6e}, weighted_PDE={row['volume_time_pde_rms']:.6e}"
                   if row["status"] == "completed" else f": {row['error']}"), flush=True)
    summary = save()
    columns = ["seed", "sampling", "status", "error", "output", *OBJECTIVES, "process_wall_s"]
    with (output/"runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed: {summary['completed']}/10; complete pairs: {summary['complete_pairs']}/5")
    for key, stats in summary["paired_statistics"].items():
        print(f"{key}: median delta={stats['median']:.6e}, full_radius lower={stats['full_radius_lower']}/"+
              str(summary["complete_pairs"]))
    print(f"Output directory: {output}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
