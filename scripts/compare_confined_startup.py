"""Fixed paired startup representation experiment with frozen DFN current."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
from torch import nn

from dfn_pinn.dfn_boundary_variant import DFNBoundaryVariant, VARIANT
from dfn_pinn.particle_startup_variant import install_startup_particles, particle_metadata
from dfn_pinn.particle_confined_startup import ParticleConfinedStartup
from prepare_dfn_baseline import sha256, make_samples
from probe_boundary_200 import restore_probe
from probe_frozen_particle_coupling import particle_step, TERMS
from probe_potential_200 import assert_equal_tree
from probe_solid_potential_scaling import measure
from smoke_particle_startup_variant import concentration_checks


SCHEMA = "paired_confined_startup_v1"
ARMS = {"unconfined": "beta_sqrtFo_raw", "confined": "beta_Fo_bulk_plus_beta_sqrtFo_layer_raw"}


def configure(model, arm):
    if arm not in ARMS:
        raise ValueError("Unknown representation")
    install_startup_particles(model, 42)
    if arm == "confined":
        model.cs = nn.ModuleList([ParticleConfinedStartup(f.raw, f.scales, float(f.initial),
                                   f.current_scale, f.duration_s) for f in model.cs])
    return model


def restore(saved):
    if saved.get("schema") != SCHEMA or saved.get("variant") != VARIANT or saved.get("budget") != 200:
        raise ValueError("Invalid paired experiment identity")
    arm = saved.get("arm")
    if arm not in ARMS or saved.get("formula") != ARMS[arm] or saved.get("loss_terms") != list(TERMS) or saved.get("particle_seed") != 42:
        raise ValueError("Invalid particle representation identity")
    model = configure(DFNBoundaryVariant(saved["settings"]), arm)
    if saved["representation"] != model.representation_metadata() or saved["particle_metadata"] != particle_metadata(model):
        raise ValueError("Scale metadata mismatch")
    for name, value in model.named_buffers():
        if not torch.equal(value, saved["model"][name]):
            raise ValueError(f"Fixed buffer mismatch: {name}")
    model.load_state_dict(saved["model"], strict=True)
    return model


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/"results/boundary_200_20260928T201841393914Z"
    prior = json.loads((source/"report.json").read_text())
    if prior["status"] != "BOUNDARY_200_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed boundary source required")
    item = prior["milestones"]["200"]
    hashes = {source/"report.json": sha256(source/"report.json"), source/item["checkpoint"]: item["sha256"],
              source/"paired_inputs.pt": prior["paired_inputs_sha256"]}
    hashes.update({root/k: v for k, v in prior["source_hashes"].items()})
    for p, h in hashes.items():
        if sha256(p) != h:
            raise ValueError(f"Source mismatch: {p}")
    torch.set_num_threads(1)
    original = torch.load(source/item["checkpoint"], weights_only=True)
    if original["step"] != 200 or original["settings"] != prior["settings"]:
        raise ValueError("Source identity mismatch")
    base = restore_probe(original)
    bundle = torch.load(source/"paired_inputs.pt", weights_only=True)
    samples = {"training": bundle["training_samples"], "historical_check": bundle["fresh_samples"]}
    for name, label in (("training", "training"), ("historical_check", "fresh")):
        assert_equal_tree(measure(base, samples[name]), item["metrics"][label])
    config = {"sampling_seed": 20260930, "training": {
        "interior_points_per_region": 128, "boundary_times": 64, "minimum_positive_time_s": 1e-6}}
    samples["new_check"] = {k: torch.from_numpy(v) for k, v in make_samples(config, base.settings).items()}
    models = {arm: configure(copy.deepcopy(base), arm) for arm in ARMS}
    assert_equal_tree(models["unconfined"].state_dict(), models["confined"].state_dict())
    fixed = {k: v.clone() for k, v in base.state_dict().items() if not k.startswith("cs.")}
    output = root/"results"/("paired_confined_startup_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    torch.save(samples, output/"samples.pt")
    sources = [Path(__file__), root/"scripts/smoke_particle_startup_variant.py", root/"scripts/probe_frozen_particle_coupling.py",
               root/"scripts/probe_boundary_200.py", root/"scripts/probe_potential_200.py",
               root/"scripts/probe_solid_potential_scaling.py", root/"scripts/prepare_dfn_baseline.py",
               *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report = {"status": "RUNNING", "schema": SCHEMA, "arms": {}, "budget_per_arm": 200,
        "learning_rate": .001, "particle_seed": 42, "new_check_seed": 20260930,
        "raw_initial_weights_equal": True, "loss_terms": list(TERMS), "term_weights": 1.,
        "scope": "Fresh paired particle representations; frozen current and fields. No kinetics in objective, no full DFN acceptance.",
        "input_hashes": {str(p): h for p, h in hashes.items()},
        "source_hashes": {str(p.relative_to(root)): sha256(p) for p in sources},
        "samples_sha256": sha256(output/"samples.pt"), "torch_version": str(torch.__version__)}

    def write():
        p = output/"report.tmp"
        p.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        p.replace(output/"report.json")

    start = time.perf_counter()
    write()
    print(f"Paired fixed 200-step startup experiment: {output}", flush=True)
    for arm, model in models.items():
        row = {"status": "RUNNING", "history": [], "milestones": {}}
        report["arms"][arm] = row
        optimizer = torch.optim.Adam(model.cs.parameters(), lr=.001)
        try:
            for step in range(201):
                if time.perf_counter()-start > 600:
                    raise TimeoutError("Ten-minute paired cap reached")
                if step:
                    row["history"].append(particle_step(model, optimizer, samples["training"]))
                ranges = concentration_checks(model)
                if any(v["outside_0_1"] for v in ranges.values()):
                    row["invalid_concentration"] = {"step": step, "ranges": ranges}
                    raise ValueError("Concentration outside [0,1] on guard grid; no clipping")
                assert_equal_tree(fixed, {k: v for k, v in model.state_dict().items() if not k.startswith("cs.")})
                if step not in (0, 20, 50, 100, 200):
                    continue
                metrics = {label: measure(model, data) for label, data in samples.items()}
                path = output/f"{arm}_{step:04d}.pt"
                torch.save({"schema": SCHEMA, "variant": original["variant"], "budget": 200, "arm": arm,
                    "formula": ARMS[arm], "settings": model.settings, "particle_seed": 42, "loss_terms": list(TERMS),
                    "representation": model.representation_metadata(), "particle_metadata": particle_metadata(model),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)
                saved = torch.load(path, weights_only=True)
                replay = restore(saved)
                replay_optimizer = torch.optim.Adam(replay.cs.parameters(), lr=.001)
                replay_optimizer.load_state_dict(saved["optimizer"])
                assert_equal_tree(optimizer.state_dict(), replay_optimizer.state_dict())
                for label, data in samples.items():
                    assert_equal_tree(measure(replay, data), metrics[label])
                row["milestones"][str(step)] = {"metrics": metrics, "ranges": ranges, "checkpoint": path.name,
                    "sha256": sha256(path), "replay_verified": True, "frozen_branches_unchanged": True}
                write()
                m = metrics["new_check"]["residuals"]
                print(f'{arm} {step}: positive PDE={m["particle_2"]["rms"]:.6g}, flux={m["particle_flux_2"]["rms"]:.6g}', flush=True)
            control = copy.deepcopy(model)
            control_optimizer = torch.optim.Adam(control.cs.parameters(), lr=.001)
            control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            assert_equal_tree(particle_step(control, control_optimizer, samples["training"]),
                              particle_step(replay, replay_optimizer, samples["training"]))
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
            raise ValueError("Source changed")
    report.update(status="PAIRED_COMPLETE_NOT_VALIDATED" if all(v["status"] == "COMPLETED_NOT_VALIDATED" for v in report["arms"].values()) else "PAIRED_INCOMPLETE",
                  source_unchanged=True, wall_s=time.perf_counter()-start)
    write()
    print(f'{report["status"]}. Report: {output/"report.json"}', flush=True)


if __name__ == "__main__":
    main()
