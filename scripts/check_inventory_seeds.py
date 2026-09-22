"""Resume a fixed inventory study; reuse seed 42 and train seeds 0,1,2,3."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from assess_flux_pilot import CRITERIA


SEEDS = (42, 0, 1, 2, 3)
ARTIFACTS = ("report.json", "model.pt", "training_points.npz", "evaluation.npz", "loss_history.csv", "benchmark.png")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(run):
    return {name: digest(run/name) for name in ARTIFACTS}


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def study_lock(directory):
    """OS lock is released even if the interpreter exits unexpectedly."""
    with (directory/"study.lock").open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_training(run, seed):
    report = json.loads((run/"report.json").read_text(encoding="utf-8"))
    expected = {"seed": seed, "variant": "inventory", "sampling": "full_radius",
                "adam_steps": 2000, "lbfgs_steps": 300, "mass_weight": 0.0}
    if any(report["config"].get(key) != value for key, value in expected.items()):
        raise ValueError("Projected training configuration differs from the fixed study")
    if report["inventory_projection"]["radial_order"] != 64:
        raise ValueError("Projection order must be 64")
    return fingerprints(run)


def recover_training(attempts):
    """Reuse a fully written run, but never overwrite partial training artifacts."""
    for previous in reversed(attempts):
        candidate = Path(previous["run"])
        if all((candidate/name).is_file() for name in ARTIFACTS):
            return candidate
    return None


def build_summary(state):
    completed = [entry for entry in state["runs"].values() if entry["status"] == "completed"]
    targets = {}
    for key, limit in CRITERIA.items():
        values = [entry["comparison"]["runs"]["projected"]["metrics"][key] for entry in completed]
        targets[key] = {"limit": limit, "passing": sum(value <= limit for value in values),
                        "required": 5, "worst_completed": max(values) if values else None}
    status = "INCOMPLETE" if len(completed) != 5 else (
        "PASS" if all(item["passing"] == 5 for item in targets.values()) else "FAIL")
    paired = []
    for entry in completed:
        runs = entry["comparison"]["runs"]
        keys = (*CRITERIA, "positive_time_rms_error", "positive_time_pde_rms", "physical_pde_rms")
        paired.append({"seed": entry["seed"], "projected_minus_control": {
            key: runs["projected"]["metrics"][key]-runs["control"]["metrics"][key] for key in keys}})
    return {"completed": len(completed), "required": 5, "exploratory_status": status,
            "targets": targets, "paired_differences": paired,
            "scope": "Exploratory previously studied seeds, not independent confirmatory evidence. "
                     "Seed 42 reused explicitly. Missing and failed runs prevent acceptance. "
                     "Same optimizer budgets, unequal computational cost. No continuous-domain or DFN guarantee."}


def run_command(command, log, root):
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=14400, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed with code {result.returncode}; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, help="Persistent study directory")
    parser.add_argument("--controls", type=Path, help="Existing five-seed control study; creation only")
    parser.add_argument("--seed42", type=Path, help="Existing projected seed-42 run; creation only")
    parser.add_argument("--prepare-only", action="store_true", help="Verify reused seed 42 without new training")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.study or root/"results/inventory_seed_study_v1").resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [root/"scripts"/name for name in (
        "check_inventory_seeds.py", "train_constant_flux_particle.py", "train_single_particle.py",
        "check_constant_flux_sphere.py", "compare_inventory_training.py", "diagnose_flux_residual.py",
        "assess_flux_pilot.py")] + sorted((root/"src/dfn_pinn").glob("*.py"))
    source_hashes = {str(path.relative_to(root)): digest(path) for path in sources}
    with study_lock(output):
        state_path = output/"progress.json"
        if state_path.exists():
            if args.controls or args.seed42:
                parser.error("Existing studies already pin inputs; omit --controls and --seed42")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["source_sha256"] != source_hashes:
                raise ValueError("Study source changed; restore matching source or use a new --study directory")
        else:
            controls = args.controls
            if controls is None:
                choices = sorted((root/"results").glob("flux_particle_seeds_*/summary.json"))
                if not choices:
                    raise ValueError("No control study found")
                controls = choices[-1].parent
            control_summary = json.loads((controls/"summary.json").read_text(encoding="utf-8"))
            paths = {}
            for seed in SEEDS:
                matches = [row for row in control_summary["runs"] if row["seed"] == seed
                           and row["sampling"] == "full_radius" and row["status"] == "completed"]
                if len(matches) != 1:
                    raise ValueError(f"Require exactly one completed control for seed {seed}")
                paths[str(seed)] = str(Path(matches[0]["output"]).resolve())
            reused = args.seed42
            if reused is None:
                for path in sorted((root/"results").glob("constant_flux_pinn_*/report.json"), reverse=True):
                    config = json.loads(path.read_text(encoding="utf-8"))["config"]
                    if config.get("variant") == "inventory" and config["seed"] == 42:
                        reused = path.parent
                        break
            if reused is None:
                raise ValueError("A completed projected seed 42 is required")
            validate_training(reused, 42)
            state = {"source_sha256": source_hashes, "controls": paths,
                     "control_sha256": {seed: fingerprints(Path(path)) for seed, path in paths.items()},
                     "seed42": str(reused.resolve()), "runs": {
                         str(seed): {"seed": seed, "status": "pending", "attempts": []} for seed in SEEDS}}
            atomic_json(state_path, state)

        def save():
            atomic_json(state_path, state)
            atomic_json(output/"summary.json", build_summary(state))

        print(f"Persistent study: {output}", flush=True)
        for seed in SEEDS:
            if args.prepare_only and seed != 42:
                continue
            entry = state["runs"][str(seed)]
            control = Path(state["controls"][str(seed)])
            if fingerprints(control) != state["control_sha256"][str(seed)]:
                raise ValueError(f"Control artifacts changed for seed {seed}")
            if entry["status"] == "completed":
                if (fingerprints(Path(entry["run"])) != entry["artifact_sha256"] or
                        digest(Path(entry["comparison_path"])) != entry["comparison_sha256"]):
                    raise ValueError(f"Completed seed {seed} artifacts changed")
                print(f"Seed {seed}: verified and skipped", flush=True)
                continue
            attempt = output/f"seed_{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            attempt.mkdir()
            # Recover a finished training if interruption occurred before its audit.
            run = Path(state["seed42"]) if seed == 42 else None
            if run is None:
                run = recover_training(entry["attempts"])
            needs_training = run is None
            run = run or attempt/"training"
            record = {"run": str(run), "directory": str(attempt), "status": "running"}
            entry["attempts"].append(record)
            entry["status"] = "running"
            save()
            try:
                if needs_training:
                    print(f"Seed {seed}: training; log={attempt/'training.log'}", flush=True)
                    run_command([sys.executable, str(root/"scripts/train_constant_flux_particle.py"),
                                 "--variant", "inventory", "--sampling", "full_radius", "--seed", str(seed),
                                 "--output", str(run)], attempt/"training.log", root)
                print(f"Seed {seed}: auditing saved model", flush=True)
                hashes = validate_training(run, seed)
                comparison = attempt/"comparison"
                run_command([sys.executable, str(root/"scripts/compare_inventory_training.py"),
                             "--projected", str(run), "--control", str(control), "--output", str(comparison)],
                            attempt/"audit.log", root)
                report_path = comparison/"report.json"
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                entry.update(status="completed", run=str(run), artifact_sha256=hashes,
                             comparison=payload, comparison_path=str(report_path), comparison_sha256=digest(report_path))
                record["status"] = "completed"
                print(f"Seed {seed}: completed", flush=True)
            except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                entry["status"] = record["status"] = "failed"
                record["error"] = str(error)
                print(f"Seed {seed}: FAILED: {error}", flush=True)
            finally:
                save()
        save()
        summary = build_summary(state)
        print(f"Completed: {summary['completed']}/5; exploratory status: {summary['exploratory_status']}")
        print(f"Summary: {output/'summary.json'}")
        return int(any(row["status"] == "failed" for row in state["runs"].values()))


if __name__ == "__main__":
    sys.exit(main())
