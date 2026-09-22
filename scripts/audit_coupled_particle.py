"""Audit an explicitly selected coupled-particle checkpoint without training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch

from dfn_pinn.coupled_training import CoupledParticle
from dfn_pinn.coupled_audit import audit, assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    report_path, checkpoint_path = run/"report.json", run/"checkpoint.pt"
    original = json.loads(report_path.read_text())
    for name, expected in original["source_hashes"].items():
        source = (root/name).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Training source changed: {name}")
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if checkpoint["config"] != original["config"]:
        raise ValueError("Checkpoint and report configurations differ")
    torch.set_num_threads(1)
    model = CoupledParticle(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    replay = model.losses(checkpoint["points"])
    for key, value in replay.items():
        torch.testing.assert_close(value.detach(), value.new_tensor(original["final_training_losses"][key]), rtol=1e-9, atol=1e-12)
    print("Sources and saved training losses verified. Auditing frozen checkpoint...", flush=True)
    c = checkpoint["config"]
    metrics, sampling = audit(model, nr=c["validation_radial_points"], nt=c["validation_time_points"])
    status = original["status"]
    if status not in ("smoke_complete_not_scientific_acceptance", "trained_pending_independent_audit"):
        raise ValueError("Unrecognized training status")
    assessment = assess(metrics, c["acceptance"], status.startswith("smoke"))
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}", flush=True)
    result = {**assessment, "metrics": metrics, "sampling": sampling, "limits": c["acceptance"],
              "run": str(run), "training_losses_reproduced": True,
              "artifact_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (report_path, checkpoint_path)},
              "auditor_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), root/"src/dfn_pinn/coupled_audit.py")}}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_root or run)/f"audit_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output/"audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Status: {assessment['status']}; no training was run.")
    print(f"Report: {output/'audit.json'}")


if __name__ == "__main__":
    main()
