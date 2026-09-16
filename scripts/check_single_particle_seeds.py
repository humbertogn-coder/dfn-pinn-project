"""Run a fixed five-seed single-particle pilot without selecting the best seed."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


SEEDS = (0, 1, 2, 3, 42)
METRICS = (
    "max_abs_c_hat_error", "rms_c_hat_error", "initial_max_error",
    "surface_max_abs_d_c_hat_d_rho", "center_max_abs_d_c_hat_d_rho",
    "max_mean_error_from_exact_0_5", "max_mean_drift_from_predicted_initial",
    "validation_pde_rms", "minimum_c_hat", "maximum_c_hat",
)


def summarize(rows):
    """Descriptive statistics for completed finite runs; report failures separately."""
    successful = [row for row in rows if row["status"] == "completed"]
    summary = {"requested": len(rows), "completed": len(successful),
               "failed": len(rows)-len(successful), "statistics": {}}
    if successful:
        for key in METRICS + ("reported_elapsed_s", "process_wall_s"):
            values = [row[key] for row in successful]
            summary["statistics"][key] = {
                "mean": statistics.mean(values), "median": statistics.median(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "minimum": min(values), "maximum": max(values),
            }
    return summary


def validate_report(report, seed):
    import math
    config = report["config"]
    if config["seed"] != seed or config["adam_steps"] != 1000 or config["lbfgs_steps"] != 150:
        raise ValueError("Unexpected training configuration")
    for key, expected in {"dtype": "float64", "device": "cpu", "threads": 1,
                          "collocation": 256, "initial_points": 65, "surface_points": 64,
                          "end_time": .1, "diffusion_number": 1.}.items():
        if report[key] != expected:
            raise ValueError(f"Unexpected configuration: {key}")
    if report["loss_weights"] != {"pde": 1, "initial": 10, "surface": 1}:
        raise ValueError("Unexpected loss weights")
    if not all(math.isfinite(report["metrics"][key]) for key in METRICS):
        raise ValueError("Nonfinite evaluation metric")
    if not math.isfinite(report["elapsed_s"]) or report["elapsed_s"] < 0:
        raise ValueError("Invalid elapsed time")


def main():
    root = Path(__file__).resolve().parents[1]
    trainer = root/"scripts/train_single_particle.py"
    output = root/"results"/("single_particle_seeds_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    sources = (trainer, Path(__file__), root/"src/dfn_pinn/spherical_diffusion.py",
               root/"src/dfn_pinn/projection.py")
    provenance = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sources}
    rows = []
    print(f"Five-seed study: {SEEDS}; unchanged training settings.", flush=True)
    for seed in SEEDS:
        run = output/f"seed_{seed}"
        log = output/f"seed_{seed}.log"
        row = {"seed": seed, "status": "failed", "error": "", "output": str(run)}
        started = time.perf_counter()
        print(f"Training seed {seed}...", flush=True)
        try:
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    [sys.executable, str(trainer), "--seed", str(seed), "--adam-steps", "1000",
                     "--lbfgs-steps", "150", "--output", str(run)], cwd=root,
                    stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
            if result.returncode:
                raise RuntimeError(f"Training exited with code {result.returncode}; see {log.name}")
            report = json.loads((run/"report.json").read_text(encoding="utf-8"))
            validate_report(report, seed)
            if not all((run/name).is_file() for name in ("model.pt", "benchmark.png", "evaluation.npz", "loss_history.csv")):
                raise ValueError("Incomplete training artifacts")
            row.update({key: report["metrics"][key] for key in METRICS})
            row.update(status="completed", reported_elapsed_s=report["elapsed_s"])
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
            row["error"] = str(error)
        row["process_wall_s"] = time.perf_counter()-started
        rows.append(row)
        # Preserve progress if a later run is interrupted.
        (output/"progress.json").write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")
        if row["status"] == "completed":
            print(f"Seed {seed}: max_error={row['max_abs_c_hat_error']:.6e}, "
                  f"mean_drift={row['max_mean_drift_from_predicted_initial']:.6e}", flush=True)
        else:
            print(f"Seed {seed}: FAILED: {row['error']}", flush=True)
    summary = summarize(rows)
    summary.update(seeds=list(SEEDS), source_sha256=provenance, runs=rows,
                   scope="Five predetermined seeds, no hyperparameter changes or seed selection. "
                   "Seeds jointly vary network initialization and collocation. Same evaluation grid "
                   "for all runs. Statistics exclude failed runs, whose count is explicit. No "
                   "accuracy pass/fail threshold, confidence interval or DFN generalization claim. "
                   "Reported elapsed time excludes imports and plots; process wall time includes them.")
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    columns = ["seed", "status", "error", "output", *METRICS, "reported_elapsed_s", "process_wall_s"]
    with (output/"runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed: {summary['completed']}/{summary['requested']}")
    for key in ("max_abs_c_hat_error", "rms_c_hat_error", "max_mean_drift_from_predicted_initial", "process_wall_s"):
        if key in summary["statistics"]:
            s = summary["statistics"][key]
            print(f"{key}: median={s['median']:.6e}, min={s['minimum']:.6e}, max={s['maximum']:.6e}")
    print(f"Output directory: {output}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
