"""One fixed 200-step boundary variant, compared with frozen amplitude controls."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from dfn_pinn.dfn_boundary_variant import DFNBoundaryVariant, SCHEMA, VARIANT, restore_boundary_model
from dfn_pinn.dfn_smoke import DFNSmoke
from dfn_pinn.dfn_run_contract import adam_step
from prepare_dfn_baseline import sha256
from probe_potential_200 import assert_equal_tree
from probe_solid_potential_scaling import measure
from smoke_dfn_boundary_variant import check_boundaries


PROBE_SCHEMA = "dfn_boundary_200_v1"


def restore_probe(saved):
    if saved.get("schema") != PROBE_SCHEMA or saved.get("budget") != 200:
        raise ValueError("Expected the bounded 200-step boundary experiment")
    # Reuse strict architecture restoration without changing the historical loader.
    return restore_boundary_model(dict(saved, schema=SCHEMA))


def main():
    root = Path(__file__).resolve().parents[1]
    prior = root/"results/potential_scale_200_20260928T192014760285Z"
    control_report = json.loads((prior/"report.json").read_text())
    if control_report["status"] != "BOUNDED_200_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed paired controls required")
    if control_report["torch_version"] != str(torch.__version__):
        raise ValueError("Torch version mismatch")
    hashes = {prior/"report.json": sha256(prior/"report.json"),
              prior/"paired_inputs.pt": control_report["paired_inputs_sha256"]}
    hashes.update({root/k: v for k, v in control_report["source_hashes"].items()})
    for arm in control_report["variants"].values():
        for checkpoint in arm["milestones"].values():
            hashes[prior/checkpoint["checkpoint"]] = checkpoint["sha256"]
    for p, digest in hashes.items():
        if sha256(p) != digest:
            raise ValueError(f"Control source/artifact mismatch: {p}")
    bundle = torch.load(prior/"paired_inputs.pt", weights_only=True)
    settings = control_report["settings"]
    if dict(bundle["settings"], adam_steps=200) != settings:
        raise ValueError("Paired physical/training settings mismatch")
    torch.set_num_threads(1)
    # Verify final control metrics from disk rather than trusting report numbers.
    controls = {}
    for name, arm in control_report["variants"].items():
        saved = torch.load(prior/arm["milestones"]["200"]["checkpoint"], weights_only=True)
        model = DFNSmoke(settings)
        if saved["settings"] != settings or saved["amplitudes"] != arm["amplitudes"]:
            raise ValueError("Control checkpoint metadata mismatch")
        for f, amplitude in zip(model.phis, saved["amplitudes"]):
            f.amplitude = amplitude
        model.load_state_dict(saved["model"], strict=True)
        controls[name] = {label: measure(model, bundle[key]) for label, key in
                          (("training", "training_samples"), ("fresh", "fresh_samples"))}
        assert_equal_tree(controls[name], arm["milestones"]["200"]["metrics"])
    model = DFNBoundaryVariant(settings)
    state = model.state_dict()
    for key, value in bundle["initial_state"].items():
        if key.startswith("phis.") and ".net." in key:
            key = key.replace(".net.", ".raw.net.", 1)
        if key not in state or state[key].shape != value.shape:
            raise ValueError(f"Initial state mapping failed: {key}")
        state[key] = value.clone()
    model.load_state_dict(state, strict=True)
    for key, value in bundle["initial_state"].items():
        mapped = key.replace(".net.", ".raw.net.", 1) if key.startswith("phis.") and ".net." in key else key
        assert_equal_tree(model.state_dict()[mapped], value)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    output = root/"results"/("boundary_200_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    sources = [Path(__file__), root/"scripts/probe_potential_200.py", root/"scripts/probe_solid_potential_scaling.py",
               root/"scripts/prepare_dfn_baseline.py", root/"scripts/smoke_dfn_boundary_variant.py",
               *sorted((root/"src/dfn_pinn").glob("*.py"))]
    torch.save(bundle, output/"paired_inputs.pt")
    report = {"status": "RUNNING", "schema": PROBE_SCHEMA, "variant": VARIANT,
              "settings": settings, "controls_final": controls, "history": [], "milestones": {},
              "initial_raw_weights_verified": True, "controls_replay_verified": True,
              "scope": "Fixed C0 hard-solid-boundary probe; not full DFN validation",
              "input_hashes": {str(p): h for p, h in hashes.items()},
              "source_hashes": {str(p.relative_to(root)): sha256(p) for p in sources},
              "paired_inputs_sha256": sha256(output/"paired_inputs.pt"), "torch_version": str(torch.__version__)}

    def write():
        p = output/"report.tmp"
        p.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        p.replace(output/"report.json")

    start = time.perf_counter()
    write()
    print(f"Fixed 200-step boundary probe: {output}", flush=True)
    try:
        for step in range(201):
            if time.perf_counter()-start > 600:
                raise TimeoutError("Ten-minute cap reached at step boundary")
            if step:
                report["history"].append(adam_step(model, optimizer, bundle["training_samples"]))
            if step not in (0, 20, 50, 100, 200):
                continue
            metrics = {label: measure(model, bundle[key]) for label, key in
                       (("training", "training_samples"), ("fresh", "fresh_samples"))}
            boundary = {label: check_boundaries(model, bundle[key]) for label, key in
                        (("training", "training_samples"), ("fresh", "fresh_samples"))}
            path = output/f"boundary_{step:04d}.pt"
            torch.save({"schema": PROBE_SCHEMA, "budget": 200, "variant": VARIANT,
                        "settings": settings, "representation": model.representation_metadata(),
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "rng_state": torch.get_rng_state()}, path)
            saved = torch.load(path, weights_only=True)
            replay = restore_probe(saved)
            replay_optimizer = torch.optim.Adam(replay.parameters(), lr=settings["learning_rate"])
            replay_optimizer.load_state_dict(saved["optimizer"])
            assert_equal_tree(optimizer.state_dict(), replay_optimizer.state_dict())
            for label, key in (("training", "training_samples"), ("fresh", "fresh_samples")):
                assert_equal_tree(measure(replay, bundle[key]), metrics[label])
            report["milestones"][str(step)] = {"metrics": metrics, "boundary_checks": boundary,
                "checkpoint": path.name, "sha256": sha256(path), "replay_verified": True}
            write()
            print(f'Boundary {step}: fresh MSE={metrics["fresh"]["total_mse"]:.6g}', flush=True)
        control = copy.deepcopy(model)
        control_optimizer = torch.optim.Adam(control.parameters(), lr=settings["learning_rate"])
        control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        assert_equal_tree(adam_step(control, control_optimizer, bundle["training_samples"]),
                          adam_step(replay, replay_optimizer, bundle["training_samples"]))
        assert_equal_tree(control.state_dict(), replay.state_dict())
        assert_equal_tree(control_optimizer.state_dict(), replay_optimizer.state_dict())
        for p, digest in hashes.items():
            if sha256(p) != digest:
                raise ValueError("Control artifact changed")
        report.update(status="BOUNDARY_200_COMPLETE_NOT_VALIDATED", controls_unchanged=True,
                      disposable_continuation_verified=True)
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_s"] = time.perf_counter()-start
        write()
    print(f'{report["status"]}. Report: {output/"report.json"}', flush=True)


if __name__ == "__main__":
    main()
