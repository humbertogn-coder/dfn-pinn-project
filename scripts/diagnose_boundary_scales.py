"""Frozen residual scales and branch gradients; no optimizer steps."""

from datetime import datetime, timezone
import json
from pathlib import Path
import torch

from dfn_pinn.constitutive import FARADAY_CONSTANT
from dfn_pinn.dfn_gradient_diagnosis import loss_gradients
from prepare_dfn_baseline import sha256
from probe_boundary_200 import restore_probe
from probe_potential_200 import assert_equal_tree
from probe_solid_potential_scaling import measure


def main():
    root = Path(__file__).resolve().parents[1]
    run = root/"results/boundary_200_20260928T201841393914Z"
    prior = json.loads((run/"report.json").read_text())
    if prior["status"] != "BOUNDARY_200_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed boundary probe required")
    item = prior["milestones"]["200"]
    hashes = {run/"report.json": sha256(run/"report.json"), run/item["checkpoint"]: item["sha256"],
              run/"paired_inputs.pt": prior["paired_inputs_sha256"]}
    hashes.update({root/k: v for k, v in prior["source_hashes"].items()})
    for p, digest in hashes.items():
        if sha256(p) != digest:
            raise ValueError(f"Input mismatch: {p}")
    torch.set_num_threads(1)
    saved = torch.load(run/item["checkpoint"], weights_only=True)
    if saved["step"] != 200 or saved["settings"] != prior["settings"]:
        raise ValueError("Checkpoint identity mismatch")
    model = restore_probe(saved)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    bundle = torch.load(run/"paired_inputs.pt", weights_only=True)
    c, s = model.settings, model.charge_scales
    scales = {"charge_residual_divisor_A_m3": s.current_A_m2/s.length_m,
              "electrolyte_mass_divisor_mol_m3_s": c["initial_ce_mol_m3"]/c["time_reference_s"],
              "particle_scales": {}}
    for k, electrode in enumerate(("n", "p")):
        p = model.reactions[k].scales
        scales["particle_scales"][electrode] = {
            "pde_divisor_mol_m3_s": p.concentration_mol_m3/p.time_s,
            "flux_and_kinetics_divisor_A_m2_active": model.jref[k],
            "molar_flux_divisor_mol_m2_s": model.jref[k]/FARADAY_CONSTANT,
            "normalized_flux_radial_derivative_coefficient": FARADAY_CONSTANT*p.diffusivity_m2_s*p.concentration_mol_m3/(p.radius_m*model.jref[k]),
            "diffusion_number": p.diffusion_number,
            "particle_output_amplitude": model.cs[k].amplitude,
            "hard_initial_time_multiplier": "physical time / duration; vanishes at zero",
            "reaction_output_amplitude_A_m2_active": model.reactions[k].integral.current_model.amplitude}
    report = {"status": "FROZEN_SCALE_GRADIENT_DIAGNOSTIC_ONLY", "scales": scales,
              "samples": {}, "input_hashes": {str(p): h for p, h in hashes.items()},
              "scope": "Euclidean MSE parameter gradients, not Adam updates; normalized terms have different physical divisors. No physical acceptance."}
    for label, key in (("training", "training_samples"), ("fresh", "fresh_samples")):
        assert_equal_tree(measure(model, bundle[key]), item["metrics"][label])
        report["samples"][label] = loss_gradients(model, bundle[key])
        print(f"{label}: 34 term gradients and total-gradient sum verified", flush=True)
    assert_equal_tree(frozen, model.state_dict())
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Parameter gradient buffers were changed")
    for p, digest in hashes.items():
        if sha256(p) != digest:
            raise ValueError("Input changed")
    report.update(metric_replay_verified=True, model_and_artifacts_unchanged=True)
    sources = [Path(__file__), root/"scripts/probe_boundary_200.py", root/"scripts/probe_potential_200.py",
               root/"scripts/probe_solid_potential_scaling.py", root/"scripts/prepare_dfn_baseline.py",
               *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report["source_hashes"] = {str(p.relative_to(root)): sha256(p) for p in sources}
    output = root/"results"/("boundary_scale_gradients_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"No training. Report: {output/'report.json'}", flush=True)


if __name__ == "__main__":
    main()
