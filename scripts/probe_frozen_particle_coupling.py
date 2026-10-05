"""Fixed 200-step particle-only diagnostic from a frozen DFN checkpoint."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from dfn_pinn.dfn_boundary_variant import SCHEMA as ARCH_SCHEMA, restore_boundary_model
from prepare_dfn_baseline import sha256, make_samples
from probe_boundary_200 import restore_probe
from probe_potential_200 import assert_equal_tree
from probe_solid_potential_scaling import measure


SCHEMA = "frozen_particle_coupling_v1"
TERMS = tuple(f"{name}_{i}" for i in (0, 2) for name in ("particle", "particle_flux", "inventory", "center"))


def freeze_except_particles(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("cs."))


def particle_step(model, optimizer, samples):
    optimizer.zero_grad(set_to_none=True)
    residuals, _ = model.residuals(samples)
    losses = {name: residuals[name].square().mean() for name in TERMS}
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise ValueError("Nonfinite particle loss")
    total.backward()
    for parameter in model.cs.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError("Missing or nonfinite particle gradient")
    if any(p.grad is not None for n, p in model.named_parameters() if not n.startswith("cs.")):
        raise ValueError("Frozen branch received an accumulated gradient")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.cs.parameters()):
        raise ValueError("Nonfinite particle parameters")
    return {"loss": float(total.detach()), "terms": {k: float(v.detach()) for k, v in losses.items()}}


def restore(saved):
    if saved.get("schema") != SCHEMA or saved.get("trainable_prefix") != "cs." or saved.get("loss_terms") != list(TERMS):
        raise ValueError("Particle diagnostic identity mismatch")
    model = restore_boundary_model(dict(saved, schema=ARCH_SCHEMA))
    freeze_except_particles(model)
    return model


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/"results/boundary_200_20260928T201841393914Z"
    prior = json.loads((source/"report.json").read_text())
    if prior["status"] != "BOUNDARY_200_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed boundary probe required")
    item = prior["milestones"]["200"]
    hashes = {source/"report.json": sha256(source/"report.json"), source/item["checkpoint"]: item["sha256"],
              source/"paired_inputs.pt": prior["paired_inputs_sha256"]}
    hashes.update({root/k: v for k, v in prior["source_hashes"].items()})
    for p, digest in hashes.items():
        if sha256(p) != digest:
            raise ValueError(f"Source mismatch: {p}")
    torch.set_num_threads(1)
    original = torch.load(source/item["checkpoint"], weights_only=True)
    if original["step"] != 200 or original["settings"] != prior["settings"]:
        raise ValueError("Source identity mismatch")
    model = restore_probe(original)
    bundle = torch.load(source/"paired_inputs.pt", weights_only=True)
    samples = {"training": bundle["training_samples"], "historical_check": bundle["fresh_samples"]}
    for label, old in (("training", "training"), ("historical_check", "fresh")):
        assert_equal_tree(measure(model, samples[label]), item["metrics"][old])
    config = {"sampling_seed": 20260929, "training": {
        "interior_points_per_region": model.settings["interior_points_per_region"],
        "boundary_times": model.settings["boundary_times"], "minimum_positive_time_s": 1e-6}}
    samples["new_check"] = {k: torch.from_numpy(v) for k, v in make_samples(config, model.settings).items()}
    freeze_except_particles(model)
    fixed = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("cs.")}
    initial_cs = copy.deepcopy(model.cs.state_dict())
    optimizer = torch.optim.Adam(model.cs.parameters(), lr=.001)
    output = root/"results"/("frozen_particle_coupling_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    torch.save(samples, output/"samples.pt")
    report = {"status": "RUNNING", "schema": SCHEMA, "source_run": str(source),
        "settings": model.settings, "subproblem_steps": 200, "learning_rate": .001,
        "optimizer_initialization": "fresh Adam; source moments are deliberately not continued",
        "trainable_prefix": "cs.", "loss_terms": list(TERMS), "term_weights": 1.,
        "new_check_sampling_seed": 20260929, "history": [], "milestones": {},
        "scope": "Warm-start prescribed-current particle diagnostic, not coupled DFN training or reference validation",
        "input_hashes": {str(p): h for p, h in hashes.items()}, "samples_sha256": sha256(output/"samples.pt")}

    def write():
        p = output/"report.tmp"
        p.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        p.replace(output/"report.json")

    start = time.perf_counter()
    write()
    print(f"Frozen-current particle subproblem: 200 steps; {output}", flush=True)
    try:
        for step in range(201):
            if time.perf_counter()-start > 600:
                raise TimeoutError("Ten-minute cap reached")
            if step:
                report["history"].append(particle_step(model, optimizer, samples["training"]))
            if step not in (0, 20, 50, 100, 200):
                continue
            assert_equal_tree(fixed, {k: v for k, v in model.state_dict().items() if not k.startswith("cs.")})
            metrics = {label: measure(model, data) for label, data in samples.items()}
            path = output/f"particles_{step:04d}.pt"
            torch.save({"schema": SCHEMA, "variant": original["variant"], "settings": model.settings,
                "representation": model.representation_metadata(), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "budget": 200,
                "trainable_prefix": "cs.", "loss_terms": list(TERMS)}, path)
            saved = torch.load(path, weights_only=True)
            replay = restore(saved)
            replay_optimizer = torch.optim.Adam(replay.cs.parameters(), lr=.001)
            replay_optimizer.load_state_dict(saved["optimizer"])
            assert_equal_tree(optimizer.state_dict(), replay_optimizer.state_dict())
            for label, data in samples.items():
                assert_equal_tree(measure(replay, data), metrics[label])
            report["milestones"][str(step)] = {"metrics": metrics, "checkpoint": path.name,
                "sha256": sha256(path), "replay_verified": True, "frozen_branches_unchanged": True}
            write()
            flux = metrics["new_check"]["residuals"]["particle_flux_2"]["rms"]
            print(f"Step {step}: positive flux RMS on new check={flux:.6g}", flush=True)
        if all(torch.equal(v, model.cs.state_dict()[k]) for k, v in initial_cs.items()):
            raise ValueError("No particle parameter changed")
        control = copy.deepcopy(model)
        control_optimizer = torch.optim.Adam(control.cs.parameters(), lr=.001)
        control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        assert_equal_tree(particle_step(control, control_optimizer, samples["training"]),
                          particle_step(replay, replay_optimizer, samples["training"]))
        assert_equal_tree(control.state_dict(), replay.state_dict())
        assert_equal_tree(control_optimizer.state_dict(), replay_optimizer.state_dict())
        for p, digest in hashes.items():
            if sha256(p) != digest:
                raise ValueError("Source changed")
        report.update(status="PARTICLE_SUBPROBLEM_COMPLETE_NOT_VALIDATED", source_unchanged=True,
                      disposable_continuation_verified=True)
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_s"] = time.perf_counter()-start
        sources = [Path(__file__), root/"scripts/probe_boundary_200.py", root/"scripts/probe_potential_200.py",
            root/"scripts/probe_solid_potential_scaling.py", root/"scripts/prepare_dfn_baseline.py",
            *sorted((root/"src/dfn_pinn").glob("*.py"))]
        report["source_hashes"] = {str(p.relative_to(root)): sha256(p) for p in sources}
        write()
    print(f'{report["status"]}. Report: {output/"report.json"}', flush=True)


if __name__ == "__main__":
    main()
