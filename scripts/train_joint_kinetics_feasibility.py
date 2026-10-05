"""One pinned warm-start joint direct/inverse pair, with common diagnostics."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from dfn_pinn.joint_kinetics import JointKinetics, SCHEMA, restore_joint
from dfn_pinn.dfn_run_contract import adam_step
from prepare_dfn_baseline import sha256, make_samples
from compare_confined_startup import restore as restore_source
from probe_potential_200 import assert_equal_tree
from probe_solid_potential_scaling import measure
from smoke_particle_startup_variant import concentration_checks


def common_measure(model, samples):
    return measure(SimpleNamespace(residuals=model.direct_residuals), samples)


def domain_checks(model, samples):
    ranges = concentration_checks(model)
    if any(v["outside_0_1"] for v in ranges.values()):
        raise ValueError("Sampled particle concentration outside [0,1]; no clipping")
    with torch.no_grad():
        for data in samples.values():
            for i in range(3):
                value = model.ce[i](data[f"region_{i}"])
                if not torch.isfinite(value).all() or not (value > 0).all():
                    raise ValueError("Invalid sampled electrolyte concentration")
    return ranges


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root/"configs/dfn_joint_kinetics_feasibility_v1.json"
    config = json.loads(config_path.read_text())
    train = config["training"]
    if config["arms"] != ["joint_direct", "joint_inverse"] or train["adam_steps_per_arm"] != 200 or train["learning_rate"] != .001 or config["term_count"] != 34:
        raise ValueError("Only the pinned comparison is supported")
    source = root/config["source_run"]
    hashes = {source/config["checkpoint"]: config["checkpoint_sha256"],
              source/"report.json": config["source_report_sha256"], source/"samples.pt": config["samples_sha256"]}
    prior = json.loads((source/"report.json").read_text())
    if prior["status"] != "PAIRED_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed pinned source required")
    hashes.update({root/k: v for k, v in prior["source_hashes"].items()})
    for p, digest in hashes.items():
        if sha256(p) != digest:
            raise ValueError(f"Source mismatch: {p}")
    torch.set_num_threads(1)
    saved_source = torch.load(source/config["checkpoint"], weights_only=True)
    if saved_source["step"] != 200 or saved_source["arm"] != "confined":
        raise ValueError("Wrong source state")
    source_model = restore_source(saved_source)
    samples = torch.load(source/"samples.pt", weights_only=True)
    old_metrics = prior["arms"]["confined"]["milestones"]["200"]["metrics"]
    for label, data in samples.items():
        assert_equal_tree(measure(source_model, data), old_metrics[label])
    sample_config = {"sampling_seed": config["evaluation"]["additional_sampling_seed"],
        "training": {"interior_points_per_region": 128, "boundary_times": 64,
                     "minimum_positive_time_s": config["evaluation"]["minimum_positive_time_s"]}}
    samples["joint_check"] = {k: torch.from_numpy(v) for k, v in make_samples(sample_config, source_model.settings).items()}
    models = {}
    for arm in config["arms"]:
        models[arm] = JointKinetics(source_model.settings, arm)
        models[arm].load_state_dict(saved_source["model"], strict=True)
    assert_equal_tree(models["joint_direct"].state_dict(), models["joint_inverse"].state_dict())
    for model in models.values():
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError("Some joint parameters are frozen")
        for label in old_metrics:
            assert_equal_tree(common_measure(model, samples[label]), old_metrics[label])
    output = root/"results"/("joint_kinetics_feasibility_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    torch.save(samples, output/"samples.pt")
    sources = [Path(__file__), root/"scripts/compare_confined_startup.py", root/"scripts/prepare_dfn_baseline.py",
               root/"scripts/probe_potential_200.py", root/"scripts/probe_solid_potential_scaling.py",
               root/"scripts/smoke_particle_startup_variant.py", *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report = {"status": "RUNNING", "schema": SCHEMA, "config": config,
        "config_sha256": sha256(config_path), "acceptance_config_sha256": sha256(root/config["evaluation"]["acceptance_config"]),
        "input_hashes": {str(p): h for p, h in hashes.items()}, "arms": {},
        "source_hashes": {str(p.relative_to(root)): sha256(p) for p in sources},
        "paired_initial_state_and_common_metrics_verified": True, "all_parameters_trainable": True,
        "samples_sha256": sha256(output/"samples.pt"), "torch_version": str(torch.__version__),
        "scope": "Joint warm-start feasibility. Common current-form metrics, not full physical acceptance."}

    def write():
        temp = output/"report.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(output/"report.json")

    start = time.perf_counter()
    write()
    print(f"Joint direct/inverse: 200 Adam steps per arm. Output: {output}", flush=True)
    for arm, model in models.items():
        row = {"status": "RUNNING", "history": [], "milestones": {}}
        report["arms"][arm] = row
        optimizer = torch.optim.Adam(model.parameters(), lr=train["learning_rate"])
        try:
            for step in range(201):
                if time.perf_counter()-start > train["paired_wall_cap_s"]:
                    raise TimeoutError("Paired wall cap reached")
                if step:
                    row["history"].append(adam_step(model, optimizer, samples["training"]))
                ranges = domain_checks(model, samples)
                if step not in train["milestones"]:
                    continue
                metrics = {label: common_measure(model, data) for label, data in samples.items()}
                for data in metrics.values():
                    for key in ("solid_insulating_0", "solid_insulating_1", "collector_positive_solid_current", "collector_negative_solid_potential_gauge"):
                        if data["residuals"][key]["max_abs"] > 1e-10:
                            raise ValueError("Hard solid boundary identity failed")
                path = output/f"{arm}_{step:04d}.pt"
                torch.save({"schema": SCHEMA, "settings": model.settings, "mode": arm,
                    "metadata": model.metadata(), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config_sha256": report["config_sha256"], "rng_state": torch.get_rng_state()}, path)
                saved = torch.load(path, weights_only=True)
                replay = restore_joint(saved)
                replay_optimizer = torch.optim.Adam(replay.parameters(), lr=train["learning_rate"])
                replay_optimizer.load_state_dict(saved["optimizer"])
                assert_equal_tree(optimizer.state_dict(), replay_optimizer.state_dict())
                for label, data in samples.items():
                    assert_equal_tree(common_measure(replay, data), metrics[label])
                row["milestones"][str(step)] = {"common_metrics": metrics, "ranges": ranges,
                    "checkpoint": path.name, "sha256": sha256(path), "replay_verified": True}
                write()
                m = metrics["joint_check"]["residuals"]
                print(f'{arm} {step}: common kinetics n={m["kinetics_0"]["rms"]:.5g}, p={m["kinetics_2"]["rms"]:.5g}', flush=True)
            control = copy.deepcopy(model)
            control_optimizer = torch.optim.Adam(control.parameters(), lr=train["learning_rate"])
            control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            assert_equal_tree(adam_step(control, control_optimizer, samples["training"]),
                              adam_step(replay, replay_optimizer, samples["training"]))
            assert_equal_tree(control.state_dict(), replay.state_dict())
            assert_equal_tree(control_optimizer.state_dict(), replay_optimizer.state_dict())
            row.update(status="COMPLETED_NOT_VALIDATED", disposable_continuation_verified=True)
        except Exception as exc:
            row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            print(f'{arm}: {row["error"]}', flush=True)
        finally:
            write()
    for p, h in hashes.items():
        if sha256(p) != h:
            raise ValueError("Historical input changed")
    report.update(status="JOINT_PAIR_COMPLETE_NOT_VALIDATED" if all(v["status"] == "COMPLETED_NOT_VALIDATED" for v in report["arms"].values()) else "JOINT_PAIR_INCOMPLETE",
                  source_unchanged=True, wall_s=time.perf_counter()-start)
    write()
    print(f'{report["status"]}. Report: {output/"report.json"}', flush=True)


if __name__ == "__main__":
    main()
