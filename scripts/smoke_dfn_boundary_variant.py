"""Three-step neural boundary smoke and disposable continuation replay."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import torch

from dfn_pinn.dfn_boundary_variant import DFNBoundaryVariant, SCHEMA, VARIANT, restore_boundary_model
from dfn_pinn.dfn_run_contract import adam_step
from prepare_dfn_baseline import sha256
from probe_solid_potential_scaling import measure
from probe_potential_200 import assert_equal_tree


def check_boundaries(model, samples):
    residuals, _ = model.residuals(samples)
    keys = ("solid_insulating_0", "solid_insulating_1", "collector_positive_solid_current",
            "collector_negative_solid_potential_gauge")
    result = {k: float(residuals[k].detach().abs().max()) for k in keys}
    if any(value > 1e-12 for value in result.values()):
        raise ValueError("Hard boundary identity failed")
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = DFNBoundaryVariant()
    samples = model.sample()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.settings["learning_rate"])
    report = {"status": "SMOKE_DIAGNOSTIC_ONLY", "variant": VARIANT,
              "initial": measure(model, samples), "history": [], "boundary_checks": []}
    report["boundary_checks"].append(check_boundaries(model, samples))
    for step in range(3):
        report["history"].append(adam_step(model, optimizer, samples))
        report["boundary_checks"].append(check_boundaries(model, samples))
        print(f'Adam {step+1}: loss={report["history"][-1]["total_loss"]:.6g}', flush=True)
    report["final"] = measure(model, samples)
    output = root/"results"/("dfn_boundary_smoke_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    path = output/"checkpoint.pt"
    torch.save({"schema": SCHEMA, "variant": VARIANT, "settings": model.settings,
                "representation": model.representation_metadata(), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "samples": samples, "adam_completed": 3,
                "rng_state": torch.get_rng_state()}, path)
    saved = torch.load(path, weights_only=True)
    replay = restore_boundary_model(saved)
    assert_equal_tree(measure(replay, saved["samples"]), report["final"])
    replay_optimizer = torch.optim.Adam(replay.parameters(), lr=model.settings["learning_rate"])
    replay_optimizer.load_state_dict(saved["optimizer"])
    assert_equal_tree(optimizer.state_dict(), replay_optimizer.state_dict())
    control = copy.deepcopy(model)
    control_optimizer = torch.optim.Adam(control.parameters(), lr=model.settings["learning_rate"])
    control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert_equal_tree(adam_step(control, control_optimizer, samples), adam_step(replay, replay_optimizer, samples))
    assert_equal_tree(control.state_dict(), replay.state_dict())
    assert_equal_tree(control_optimizer.state_dict(), replay_optimizer.state_dict())
    report.update(metric_replay_verified=True, disposable_continuation_verified=True,
                  checkpoint_sha256=sha256(path), settings=model.settings,
                  representation=model.representation_metadata(), torch_version=str(torch.__version__))
    sources = [Path(__file__), root/"scripts/prepare_dfn_baseline.py", root/"scripts/probe_potential_200.py",
               root/"scripts/probe_solid_potential_scaling.py", *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report["source_hashes"] = {str(p.relative_to(root)): sha256(p) for p in sources}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"SMOKE ONLY; no physical acceptance. Report: {output/'report.json'}", flush=True)


if __name__ == "__main__":
    main()
