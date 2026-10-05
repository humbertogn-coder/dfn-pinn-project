"""Frozen amplitude-probe gradients, without optimization or acceptance claims."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from dfn_pinn.dfn_smoke import DFNSmoke
from dfn_pinn.dfn_gradient_diagnosis import (
    collector_sensitivity, cosine, loss_gradients, vector_gradient,
)
from dfn_pinn.spherical_diffusion import _gradient
from prepare_dfn_baseline import sha256
from probe_solid_potential_scaling import measure


def value_slope_sensitivity(model, times):
    field = model.phis[1]
    points = torch.stack((torch.ones_like(times), times/model.settings["time_reference_s"]), dim=1).requires_grad_()
    value = field(points)
    slope = _gradient(value, points)[:, :1]
    parameters = list(field.parameters())
    values = torch.stack([vector_gradient(v, parameters) for v in value[:, 0]])
    slopes = torch.stack([vector_gradient(v, parameters) for v in slope[:, 0]])
    return {"times_s": times.tolist(),
            "value_jacobian_row_norm_rms": float(values.square().sum(1).mean().sqrt()),
            "slope_jacobian_row_norm_rms": float(slopes.square().sum(1).mean().sqrt()),
            "row_cosines": [cosine(a, b) for a, b in zip(values, slopes)],
            "scope": "Normalized potential value versus d(phi_hat)/dX at the positive collector; not loss gradients."}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/solid_potential_probe_20260928T190218279084Z"))
    args = parser.parse_args()
    run = root/args.run
    source_report = run/"report.json"
    source = json.loads(source_report.read_text())
    if source["status"] != "BOUNDED_PROBE_COMPLETE_NOT_VALIDATED":
        raise ValueError("Expected a completed bounded probe")
    for name, digest in source["source_hashes"].items():
        if sha256(root/name) != digest:
            raise ValueError(f"Source mismatch: {name}")
    frozen_paths = {source_report: sha256(source_report), run/"paired_inputs.pt": source["inputs_sha256"]}
    for name in ("original", "ohmic"):
        frozen_paths[run/f"{name}.pt"] = source["variants"][name]["checkpoint_sha256"]
    for path, digest in frozen_paths.items():
        if sha256(path) != digest:
            raise ValueError(f"Artifact mismatch: {path}")
    torch.set_num_threads(1)
    bundle = torch.load(run/"paired_inputs.pt", weights_only=True)
    times = torch.unique(torch.cat((torch.linspace(1e-6, 1., 17, dtype=torch.float64),
                                   torch.logspace(-6, 0, 17, dtype=torch.float64))))
    report = {"status": "FROZEN_DIAGNOSTIC_ONLY", "input_run": str(run),
              "input_hashes": {str(p): h for p, h in frozen_paths.items()}, "variants": {},
              "scope": "Euclidean gradients, not Adam directions, causal proof or DFN validation."}
    for name in ("original", "ohmic"):
        saved = torch.load(run/f"{name}.pt", weights_only=True)
        if saved["schema"] != "solid_potential_probe_v1" or saved["settings"] != source["settings"]:
            raise ValueError("Probe identity mismatch")
        if saved["amplitudes"] != source["variants"][name]["amplitudes"]:
            raise ValueError("Amplitude metadata mismatch")
        model = DFNSmoke(saved["settings"])
        for field, amplitude in zip(model.phis, saved["amplitudes"]):
            field.amplitude = amplitude
        model.load_state_dict(saved["model"], strict=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        row = {}
        for label, key in (("training", "training_samples"), ("fresh", "fresh_samples")):
            samples = bundle[key]
            if measure(model, samples) != source["variants"][name]["final"][label]:
                raise ValueError("Saved metric replay mismatch")
            row[label] = loss_gradients(model, samples)
        row["collectors"] = collector_sensitivity(model, times)
        row["positive_value_slope"] = value_slope_sensitivity(model, times)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        assert all(p.grad is None for p in model.parameters())
        report["variants"][name] = row
        print(f"{name}: replay and frozen gradients verified", flush=True)
    for path, digest in frozen_paths.items():
        if sha256(path) != digest:
            raise ValueError("Frozen artifact changed")
    report["artifacts_unchanged"] = True
    sources = [Path(__file__), root/"scripts/probe_solid_potential_scaling.py",
               root/"scripts/prepare_dfn_baseline.py", *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report["source_hashes"] = {str(p.relative_to(root)): sha256(p) for p in sources}
    output = root/"results"/("potential_probe_gradients_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"No training. Report: {output/'report.json'}", flush=True)


if __name__ == "__main__":
    main()
