"""Fixed paired 200-step amplitude experiment; no L-BFGS or automatic extension."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from dfn_pinn.dfn_smoke import DFNSmoke
from dfn_pinn.dfn_run_contract import adam_step
from prepare_dfn_baseline import sha256
from probe_solid_potential_scaling import measure


def assert_equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise ValueError("Optimizer/model tensor replay mismatch")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Replay keys mismatch")
        for key in left:
            assert_equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise ValueError("Replay length mismatch")
        for a, b in zip(left, right):
            assert_equal_tree(a, b)
    elif left != right:
        raise ValueError("Replay metadata mismatch")


def main():
    root = Path(__file__).resolve().parents[1]
    prior = root/"results/solid_potential_probe_20260928T190218279084Z"
    prior_report = json.loads((prior/"report.json").read_text())
    if prior_report["status"] != "BOUNDED_PROBE_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed prior probe required")
    if prior_report["torch_version"] != str(torch.__version__):
        raise ValueError("Use the original Torch version for paired replay")
    hashes = {prior/"report.json": sha256(prior/"report.json"),
              prior/"paired_inputs.pt": prior_report["inputs_sha256"]}
    hashes.update({root/k: v for k, v in prior_report["source_hashes"].items()})
    hashes.update({prior/f"{n}.pt": prior_report["variants"][n]["checkpoint_sha256"]
                   for n in ("original", "ohmic")})
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError(f"Prior artifact/source mismatch: {path}")
    bundle = torch.load(prior/"paired_inputs.pt", weights_only=True)
    if bundle["settings"] != prior_report["settings"]:
        raise ValueError("Prior settings mismatch")
    settings = dict(bundle["settings"], adam_steps=200)
    torch.set_num_threads(1)
    output = root/"results"/("potential_scale_200_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    sources = [Path(__file__), root/"scripts/probe_solid_potential_scaling.py",
               root/"scripts/prepare_dfn_baseline.py", *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report = {"schema": "potential_scale_200_v1", "status": "RUNNING", "settings": settings,
              "torch_version": str(torch.__version__), "variants": {}, "milestones": [0, 20, 50, 100, 200],
              "scope": "Matched C0 amplitude probe; fixed 200 Adam steps per arm; no physical acceptance",
              "input_hashes": {str(p): h for p, h in hashes.items()},
              "source_hashes": {str(p.relative_to(root)): sha256(p) for p in sources}}
    torch.save(bundle, output/"paired_inputs.pt")
    report["paired_inputs_sha256"] = sha256(output/"paired_inputs.pt")

    def write_report():
        path = output/"report.tmp"
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        path.replace(output/"report.json")

    def restore(payload):
        model = DFNSmoke(payload["settings"])
        for field, amplitude in zip(model.phis, payload["amplitudes"]):
            field.amplitude = amplitude
        model.load_state_dict(payload["model"], strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
        optimizer.load_state_dict(payload["optimizer"])
        return model, optimizer

    start = time.perf_counter()
    write_report()
    print(f"Fixed 200-step paired amplitude probe: {output}", flush=True)
    try:
        for name in ("original", "ohmic"):
            model = DFNSmoke(settings)
            model.load_state_dict(bundle["initial_state"], strict=True)
            amplitudes = prior_report["variants"][name]["amplitudes"]
            for field, amplitude in zip(model.phis, amplitudes):
                field.amplitude = amplitude
            optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
            row = {"amplitudes": amplitudes, "history": [], "milestones": {}}
            report["variants"][name] = row
            for step in range(201):
                if time.perf_counter()-start > 600:
                    raise TimeoutError("Ten-minute paired probe cap reached at step boundary")
                if step:
                    entry = adam_step(model, optimizer, bundle["training_samples"])
                    row["history"].append(entry)
                    if step <= 20 and entry != prior_report["variants"][name]["history"][step-1]:
                        raise ValueError("Prior 20-step history mismatch")
                if step not in report["milestones"]:
                    continue
                metrics = {label: measure(model, bundle[key]) for label, key in
                           (("training", "training_samples"), ("fresh", "fresh_samples"))}
                if step == 20:
                    saved_prior = torch.load(prior/f"{name}.pt", weights_only=True)
                    assert_equal_tree(model.state_dict(), saved_prior["model"])
                    assert_equal_tree(metrics, prior_report["variants"][name]["final"])
                    row["prior_20_step_replay_verified"] = True
                path = output/f"{name}_{step:04d}.pt"
                torch.save({"schema": report["schema"], "settings": settings,
                            "variant": name, "step": step, "amplitudes": amplitudes,
                            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "rng_state": torch.get_rng_state()}, path)
                payload = torch.load(path, weights_only=True)
                replay, replay_optimizer = restore(payload)
                assert_equal_tree(optimizer.state_dict(), replay_optimizer.state_dict())
                for label, key in (("training", "training_samples"), ("fresh", "fresh_samples")):
                    assert_equal_tree(measure(replay, bundle[key]), metrics[label])
                row["milestones"][str(step)] = {"metrics": metrics,
                    "checkpoint": path.name, "sha256": sha256(path), "replay_verified": True}
                write_report()
                print(f'{name} {step}: fresh MSE={metrics["fresh"]["total_mse"]:.6g}', flush=True)
            # Verify continuation equivalence on disposable copies only.
            control = copy.deepcopy(model)
            control_optimizer = torch.optim.Adam(control.parameters(), lr=settings["learning_rate"])
            control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            a = adam_step(control, control_optimizer, bundle["training_samples"])
            b = adam_step(replay, replay_optimizer, bundle["training_samples"])
            assert_equal_tree(a, b)
            assert_equal_tree(control.state_dict(), replay.state_dict())
            assert_equal_tree(control_optimizer.state_dict(), replay_optimizer.state_dict())
            row["disposable_next_step_replay_verified"] = True
        for path, digest in hashes.items():
            if sha256(path) != digest:
                raise ValueError("Prior artifact changed")
        report.update(status="BOUNDED_200_COMPLETE_NOT_VALIDATED", prior_artifacts_unchanged=True)
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_s"] = time.perf_counter()-start
        write_report()
    print(f'{report["status"]}. Report: {output/"report.json"}', flush=True)


if __name__ == "__main__":
    main()
