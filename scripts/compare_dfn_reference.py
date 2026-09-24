"""Compare a frozen wiring checkpoint with its matched native-grid reference."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from dfn_pinn.dfn_smoke import DFNSmoke


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(root, report):
    for name, expected in report["source_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Source provenance mismatch: {name}")


def metrics(predicted, expected, scale, limit):
    error = np.abs(predicted - expected)
    if predicted.shape != expected.shape or not np.isfinite(error).all():
        raise ValueError("Invalid comparison arrays")
    maximum = float(error.max())
    return {"max_abs_error": maximum, "rms_error": float(np.sqrt(np.mean(error**2))),
            "max_scaled_error": maximum / scale, "scale": scale,
            "limit": limit, "pass": maximum <= limit}


def evaluate(function, coordinates, factor=1.):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(coordinates), 8192):
            p = torch.as_tensor(coordinates[start:start+8192], dtype=torch.float64)
            chunks.append(function(p).numpy().reshape(-1) * factor)
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/dfn_smoke_20260924T054745712381Z"))
    parser.add_argument("--reference", type=Path, default=Path("results/dfn_smoke_reference_20260924T195608554436Z"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, reference = root / args.run, root / args.reference
    training = json.loads((run / "report.json").read_text())
    ref_report = json.loads((reference / "report.json").read_text())
    verify_sources(root, training)
    verify_sources(root, ref_report)
    if digest(reference / "reference.h5") != ref_report["reference_sha256"]:
        raise ValueError("Reference data hash mismatch")
    saved = torch.load(run / "checkpoint.pt", weights_only=True, map_location="cpu")
    if saved["settings"] != training["settings"] or saved["settings"] != ref_report["settings"]:
        raise ValueError("Checkpoint/reference settings mismatch")
    torch.set_num_threads(1)
    model = DFNSmoke(saved["settings"])
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.no_grad():
        for key, value in model.snapshot(saved["samples"]).items():
            torch.testing.assert_close(value, saved["predictions"][key], rtol=0, atol=0)
    c = model.settings
    length, tref = sum(c["lengths_m"]), c["time_reference_s"]
    phi_scale = model.charge_scales.potential_V
    results = {}
    with h5py.File(reference / "reference.h5", "r") as handle:
        group = handle["x80_r320"]
        times = group["time_s"][:]
        tau = times / tref
        p0 = np.column_stack((np.zeros_like(tau), tau))
        p1 = np.column_stack((np.ones_like(tau), tau))
        voltage = evaluate(model.phis[1], p1, phi_scale) - evaluate(model.phis[0], p0, phi_scale)
        results["voltage_V"] = metrics(voltage, group["voltage_V"][:], 1., .005)
        for key, branches, factor, scale, limit in (
            ("c_e", model.ce, c["initial_ce_mol_m3"], c["initial_ce_mol_m3"], 10.),
            ("phi_e", model.phie, phi_scale, 1., .005),
        ):
            x = group[key]["x_m"][:] / length
            expected = group[key]["values"][:]
            predicted = np.empty_like(expected)
            # Reference concentration/potential arrays use cell centers, never interfaces.
            for i, (left, right) in enumerate(model.bounds):
                mask = (x > left) & (x < right)
                xx, tt = np.meshgrid(x[mask], tau, indexing="ij")
                predicted[mask] = evaluate(branches[i], np.column_stack((xx.ravel(), tt.ravel())), factor).reshape(xx.shape)
            if any(np.isclose(x, b, rtol=0, atol=1e-14).any() for b in (0., model.bounds[0][1], model.bounds[1][1], 1.)):
                raise ValueError("Expected strictly region-interior reference nodes")
            results[key] = metrics(predicted, expected, scale, limit)
        for k, suffix in enumerate(("n", "p")):
            for key, function, factor, scale, limit in (
                (f"surface_{suffix}", None, c["cmax_mol_m3"][k], c["cmax_mol_m3"][k], .001*c["cmax_mol_m3"][k]),
                (f"phi_s_{suffix}", model.phis[k], phi_scale, 1., .005),
                (f"j_{suffix}", model.reactions[k].integral.current_model, 1., model.jref[k], .01*model.jref[k]),
                (f"c_s_{suffix}", model.cs[k], c["cmax_mol_m3"][k], c["cmax_mol_m3"][k], .001*c["cmax_mol_m3"][k]),
            ):
                field = group[key]
                x = field["x_m"][:] / length
                if key.startswith("c_s_"):
                    rr, xx, tt = np.meshgrid(field["r_m"][:] / c["radii_m"][k], x, tau, indexing="ij")
                    points = np.column_stack((rr.ravel(), xx.ravel(), tt.ravel()))
                else:
                    xx, tt = np.meshgrid(x, tau, indexing="ij")
                    points = np.column_stack((xx.ravel(), tt.ravel()))
                    if key.startswith("surface_"):
                        points = np.column_stack((np.ones(len(points)), points))
                        function = model.cs[k]
                expected = field["values"][:]
                predicted = evaluate(function, points, factor).reshape(expected.shape)
                results[key] = metrics(predicted, expected, scale, limit)
    output = run / ("reference_comparison_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir()
    report = {"status": "SMOKE_DIAGNOSTIC_ONLY", "metrics": results,
              "criteria": "Exploratory field-error screening, not full physical acceptance. Fixed before future training; not preregistered for this existing smoke.",
              "limitations": "Sampled native-grid comparison only. No independent PDE, flux, interface or global inventory acceptance audit. Fine mesh is provisional; no rigorous error bounds.",
              "checkpoint_sha256": digest(run / "checkpoint.pt"),
              "reference_sha256": digest(reference / "reference.h5"),
              "source_sha256": digest(Path(__file__)), "time_samples": len(times),
              "run": str(run), "reference": str(reference), "replay_exact": True}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for key, value in results.items():
        print(f"{key}: max={value['max_abs_error']:.6e}, limit={value['limit']:.6e}, pass={value['pass']}")
    print(f"SMOKE_DIAGNOSTIC_ONLY; no training. Report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
