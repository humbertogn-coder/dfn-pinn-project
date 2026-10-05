"""Three-step startup-particle wiring smoke with frozen source fields."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import torch

from dfn_pinn.particle_startup_variant import (
    SCHEMA, REPRESENTATION, install_startup_particles, particle_metadata, restore_startup,
)
from prepare_dfn_baseline import sha256
from probe_boundary_200 import restore_probe
from probe_frozen_particle_coupling import particle_step, TERMS
from probe_potential_200 import assert_equal_tree
from probe_solid_potential_scaling import measure


def concentration_checks(model):
    result = {}
    for k, region in enumerate((0, 2)):
        rho = torch.linspace(0, 1, 17, dtype=torch.float64)
        x = torch.linspace(*model.bounds[region], 5, dtype=torch.float64)
        time = torch.cat((torch.zeros(1, dtype=torch.float64), torch.logspace(-6, 0, 9, dtype=torch.float64)))/model.settings["time_reference_s"]
        points = torch.stack(torch.meshgrid(rho, x, time, indexing="ij"), dim=-1).reshape(-1, 3)
        with torch.no_grad():
            values = model.cs[k](points)
            initial = values[points[:, 2] == 0]
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite concentration")
        torch.testing.assert_close(initial, model.cs[k].initial.expand_as(initial), rtol=0, atol=0)
        result[str(k)] = {"min": float(values.min()), "max": float(values.max()),
                          "outside_0_1": bool(((values < 0) | (values > 1)).any()), "initial_exact": True}
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/"results/boundary_200_20260928T201841393914Z"
    prior = json.loads((source/"report.json").read_text())
    if prior["status"] != "BOUNDARY_200_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed boundary checkpoint required")
    item = prior["milestones"]["200"]
    hashes = {source/"report.json": sha256(source/"report.json"), source/item["checkpoint"]: item["sha256"],
              source/"paired_inputs.pt": prior["paired_inputs_sha256"]}
    hashes.update({root/k: v for k, v in prior["source_hashes"].items()})
    for p, h in hashes.items():
        if sha256(p) != h:
            raise ValueError(f"Source mismatch: {p}")
    torch.set_num_threads(1)
    saved_source = torch.load(source/item["checkpoint"], weights_only=True)
    if saved_source["step"] != 200 or saved_source["settings"] != prior["settings"]:
        raise ValueError("Source identity mismatch")
    model = restore_probe(saved_source)
    samples = torch.load(source/"paired_inputs.pt", weights_only=True)
    assert_equal_tree(measure(model, samples["training_samples"]), item["metrics"]["training"])
    fixed = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("cs.")}
    install_startup_particles(model, seed=42)
    optimizer = torch.optim.Adam(model.cs.parameters(), lr=.001)
    report = {"status": "STARTUP_SMOKE_ONLY", "history": [], "concentration_checks": [],
              "source_run": str(source), "scope": "Fresh startup particle networks; frozen current and other fields; no physical acceptance",
              "input_hashes": {str(p): h for p, h in hashes.items()}, "loss_terms": list(TERMS),
              "particle_seed": 42, "learning_rate": .001, "smoke_steps": 3}
    report["initial"] = measure(model, samples["training_samples"])
    report["concentration_checks"].append(concentration_checks(model))
    for i in range(3):
        report["history"].append(particle_step(model, optimizer, samples["training_samples"]))
        report["concentration_checks"].append(concentration_checks(model))
        assert_equal_tree(fixed, {k: v for k, v in model.state_dict().items() if not k.startswith("cs.")})
        print(f'Adam {i+1}: particle loss={report["history"][-1]["loss"]:.6g}', flush=True)
    report["final"] = {label: measure(model, samples[key]) for label, key in
                       (("training", "training_samples"), ("historical_check", "fresh_samples"))}
    output = root/"results"/("particle_startup_smoke_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    path = output/"checkpoint.pt"
    torch.save({"schema": SCHEMA, "variant": saved_source["variant"], "settings": model.settings,
        "particle_seed": 42, "particle_representation": REPRESENTATION, "trainable_prefix": "cs.",
        "representation": model.representation_metadata(), "particle_metadata": particle_metadata(model),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "samples": samples,
        "loss_terms": list(TERMS), "step": 3}, path)
    saved = torch.load(path, weights_only=True)
    replay = restore_startup(saved)
    replay_optimizer = torch.optim.Adam(replay.cs.parameters(), lr=.001)
    replay_optimizer.load_state_dict(saved["optimizer"])
    assert_equal_tree(optimizer.state_dict(), replay_optimizer.state_dict())
    for label, key in (("training", "training_samples"), ("historical_check", "fresh_samples")):
        assert_equal_tree(measure(replay, saved["samples"][key]), report["final"][label])
    control = copy.deepcopy(model)
    control_optimizer = torch.optim.Adam(control.cs.parameters(), lr=.001)
    control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert_equal_tree(particle_step(control, control_optimizer, samples["training_samples"]),
                      particle_step(replay, replay_optimizer, samples["training_samples"]))
    assert_equal_tree(control.state_dict(), replay.state_dict())
    assert_equal_tree(control_optimizer.state_dict(), replay_optimizer.state_dict())
    for p, h in hashes.items():
        if sha256(p) != h:
            raise ValueError("Source artifact changed")
    report.update(replay_verified=True, disposable_continuation_verified=True,
                  frozen_branches_unchanged=True, checkpoint_sha256=sha256(path))
    sources = [Path(__file__), root/"scripts/probe_frozen_particle_coupling.py", root/"scripts/probe_boundary_200.py",
               root/"scripts/probe_potential_200.py", root/"scripts/probe_solid_potential_scaling.py",
               root/"scripts/prepare_dfn_baseline.py", *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report["source_hashes"] = {str(p.relative_to(root)): sha256(p) for p in sources}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"SMOKE ONLY. Report: {output/'report.json'}", flush=True)


if __name__ == "__main__":
    main()
