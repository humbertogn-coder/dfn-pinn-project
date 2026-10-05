"""Pinned 200-step inventory-weight pair; no automatic extension."""

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from dfn_pinn.joint_kinetics import SCHEMA as JOINT, restore_joint
from dfn_pinn.inventory_weighting import SCHEMA, WEIGHTS, weighted_step, restore_weighted
from prepare_dfn_baseline import sha256, make_samples
from probe_potential_200 import assert_equal_tree
from train_joint_kinetics_feasibility import common_measure, domain_checks


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "results/joint_kinetics_feasibility_20260929T200959116945Z"
    hashes = {
        source / "report.json": "ade0345a2aecbe86b79b589f82b290ccb3827557cd391c30abf76267796a9217",
        source / "samples.pt": "bf6f598f76eca6b0cfd7b002019e7a25d1dc2329b9f0ee5aa9d0aab8202ac66d",
        source / "joint_direct_0000.pt": "4fbe5b40bb126083d5da680718c7bc292bbd999014488800ae62bfc5d1d90591",
    }
    for p, h in hashes.items():
        if sha256(p) != h:
            raise ValueError(f"Pinned input mismatch: {p}")
    prior = json.loads((source / "report.json").read_text())
    if prior["status"] != "JOINT_PAIR_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed source required")
    hashes.update({root / k: v for k, v in prior["source_hashes"].items()})
    for milestone in prior["arms"]["joint_direct"]["milestones"].values():
        hashes[source / milestone["checkpoint"]] = milestone["sha256"]
    acceptance = root / "configs/dfn_baseline_v1.json"
    hashes[acceptance] = prior["acceptance_config_sha256"]
    for p, h in hashes.items():
        if sha256(p) != h:
            raise ValueError(f"Historical artifact changed: {p}")
    torch.set_num_threads(1)
    saved_initial = torch.load(source / "joint_direct_0000.pt", weights_only=True)
    samples = torch.load(source / "samples.pt", weights_only=True)
    historical_sets = tuple(samples)
    model = restore_joint(saved_initial)
    sample_config = {"sampling_seed": 20261002, "training": {
        "interior_points_per_region": 128, "boundary_times": 64, "minimum_positive_time_s": 1e-6}}
    samples["weight_check"] = {k: torch.from_numpy(v) for k, v in make_samples(sample_config, model.settings).items()}
    out = root / "results" / ("inventory_weight_pair_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(exist_ok=False)
    torch.save(samples, out / "samples.pt")
    files = [Path(__file__), root / "scripts/train_joint_kinetics_feasibility.py",
             *[root / k for k in prior["source_hashes"]], *sorted((root / "src/dfn_pinn").glob("*.py"))]
    report = {"status": "RUNNING", "schema": SCHEMA, "arms": {},
              "protocol": {"weights": WEIGHTS, "adam_steps": 200, "learning_rate": .001,
                           "wall_cap_s": 1200, "evaluation_seed": 20261002, "fresh_optimizer": True,
                           "all_fields_trainable": True, "dtype": "float64", "threads": 1},
              "input_hashes": {str(p): h for p, h in hashes.items()},
              "source_hashes": {str(p.relative_to(root)): sha256(p) for p in files},
              "samples_sha256": sha256(out / "samples.pt"), "torch_version": str(torch.__version__),
              "scope": "Common sampled diagnostics only; no independent reference/global acceptance audit."}

    def write():
        temp = out / "report.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(out / "report.json")

    started = time.perf_counter()
    write()
    print(f"Inventory weight pair: 200 steps each. Output: {out}", flush=True)
    try:
        for arm, weight in WEIGHTS.items():
            model = restore_joint(saved_initial)
            assert_equal_tree(model.state_dict(), saved_initial["model"])
            if not all(p.requires_grad and p.dtype == torch.float64 for p in model.parameters()):
                raise ValueError("All float64 parameters must be trainable")
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            row = {"status": "RUNNING", "weight": weight, "history": [], "milestones": {}}
            report["arms"][arm] = row
            for step in range(201):
                if time.perf_counter() - started > 1200:
                    raise TimeoutError("Paired wall cap reached")
                if step:
                    history = weighted_step(model, optimizer, samples["training"], weight)
                    if arm == "control":
                        assert_equal_tree({k: v for k, v in history.items() if k != "raw_losses"},
                                          prior["arms"]["joint_direct"]["history"][step-1])
                    row["history"].append(history)
                ranges = domain_checks(model, samples)
                if step not in (0, 20, 50, 100, 200):
                    continue
                metrics = {label: common_measure(model, data) for label, data in samples.items()}
                for data in metrics.values():
                    for key in ("solid_insulating_0", "solid_insulating_1", "collector_positive_solid_current", "collector_negative_solid_potential_gauge"):
                        if data["residuals"][key]["max_abs"] > 1e-10:
                            raise ValueError("Hard boundary identity failed")
                if arm == "control":
                    old = prior["arms"]["joint_direct"]["milestones"][str(step)]
                    old_saved = torch.load(source / old["checkpoint"], weights_only=True)
                    assert_equal_tree(model.state_dict(), old_saved["model"])
                    assert_equal_tree(optimizer.state_dict(), old_saved["optimizer"])
                    for label in historical_sets:
                        assert_equal_tree(metrics[label], old["common_metrics"][label])
                elif step == 0:
                    assert_equal_tree(metrics, report["arms"]["control"]["milestones"]["0"]["common_metrics"])
                path = out / f"{arm}_{step:04d}.pt"
                torch.save({"schema": SCHEMA, "arm": arm, "inventory_weight": weight, "step": step,
                            "optimizer": optimizer.state_dict(), "rng_state": torch.get_rng_state(),
                            "joint": {"schema": JOINT, "settings": model.settings, "mode": model.mode,
                                      "metadata": model.metadata(), "model": model.state_dict()}}, path)
                saved = torch.load(path, weights_only=True)
                replay = restore_weighted(saved)
                replay_optimizer = torch.optim.Adam(replay.parameters(), lr=.001)
                replay_optimizer.load_state_dict(saved["optimizer"])
                assert_equal_tree(optimizer.state_dict(), replay_optimizer.state_dict())
                for label, data in samples.items():
                    assert_equal_tree(common_measure(replay, data), metrics[label])
                row["milestones"][str(step)] = {"common_metrics": metrics, "ranges": ranges,
                    "checkpoint": path.name, "sha256": sha256(path), "replay_verified": True}
                write()
                m = metrics["weight_check"]["residuals"]
                print(f'{arm} {step}: inventory max n={m["inventory_0"]["max_abs"]:.5g}, p={m["inventory_2"]["max_abs"]:.5g}', flush=True)
            disposable = copy.deepcopy(model)
            disposable_optimizer = torch.optim.Adam(disposable.parameters(), lr=.001)
            disposable_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            assert_equal_tree(weighted_step(disposable, disposable_optimizer, samples["training"], weight),
                              weighted_step(replay, replay_optimizer, samples["training"], weight))
            assert_equal_tree(disposable.state_dict(), replay.state_dict())
            assert_equal_tree(disposable_optimizer.state_dict(), replay_optimizer.state_dict())
            row.update(status="COMPLETED_NOT_VALIDATED", continuation_replay_verified=True)
            write()
        for p, h in hashes.items():
            if sha256(p) != h:
                raise ValueError(f"Historical input changed: {p}")
        report.update(status="PAIR_COMPLETE_NOT_VALIDATED", control_trajectory_exact=True, sources_unchanged=True)
    except Exception as exc:
        report.update(status="PAIR_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_s"] = time.perf_counter() - started
        write()
    print(f'{report["status"]}. Report: {out / "report.json"}', flush=True)


if __name__ == "__main__":
    main()
