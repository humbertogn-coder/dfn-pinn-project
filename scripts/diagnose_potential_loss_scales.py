"""Inspect loss and parameter-gradient scales without optimization."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch

from dfn_pinn.potential_particle import PotentialParticle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = json.loads((args.run/"report.json").read_text())
    for name, expected in report["source_hashes"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed: {name}")
    saved = torch.load(args.run/"checkpoint.pt", weights_only=True, map_location="cpu")
    c = saved["config"]
    if c != report["config"]:
        raise ValueError("Configuration mismatch")
    torch.set_num_threads(1)
    model = PotentialParticle(c)
    model.load_state_dict(saved["model"])
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    losses = model.losses(saved["points"])
    groups = {"concentration": list(model.concentration.parameters()), "current": list(model.current.parameters())}
    parameters = groups["concentration"]+groups["current"]
    split = len(groups["concentration"])
    thresholds = {"pde": c["acceptance"]["physical_pde_rms"],
                  "surface_flux": c["acceptance"]["max_surface_flux_residual_A_m2"]/c["current_scale_A_m2"],
                  "kinetics": c["acceptance"]["max_kinetics_residual_A_m2"]/c["current_scale_A_m2"],
                  "inventory": c["acceptance"]["max_inventory_error"]}
    records = {}
    for key, loss in losses.items():
        torch.testing.assert_close(loss.detach(), loss.new_tensor(report["final_training_losses"][key]), rtol=1e-9, atol=1e-12)
        weight = c["loss_weights"][key]
        gradients = torch.autograd.grad(weight*loss, parameters, retain_graph=True, allow_unused=True)
        def norm(values):
            return float(torch.sqrt(sum((g.detach().square().sum() for g in values if g is not None), torch.zeros((), dtype=torch.float64))))
        records[key] = {"mse": float(loss.detach()), "weight": weight,
                        "weighted_loss": float(weight*loss.detach()),
                        "weighted_gradient_norm_concentration": norm(gradients[:split]),
                        "weighted_gradient_norm_current": norm(gradients[split:]),
                        "mse_divided_by_acceptance_scale_squared": float(loss.detach())/thresholds[key]**2}
    if any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()):
        raise AssertionError("Diagnostic modified model state")
    result = {"run": str(args.run.resolve()), "losses": records, "model_unchanged": True,
              "checkpoint_sha256": hashlib.sha256((args.run/"checkpoint.pt").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Final-checkpoint training-point gradients only. Threshold-scaled MSE is illustrative, not an acceptance test or prescribed training weight. No optimization."}
    output = args.run/("loss_scale_diagnosis_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(records, indent=2, allow_nan=False))
    print(f"No optimization. Report: {output/'report.json'}")


if __name__ == "__main__":
    main()
