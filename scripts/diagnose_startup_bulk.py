"""Frozen startup time/radial decomposition; no optimizer or representation change."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch

from dfn_pinn.particle_startup_variant import restore_startup
from dfn_pinn.spherical_diffusion import _gradient, diffusion_residual
from prepare_dfn_baseline import sha256
from probe_potential_200 import assert_equal_tree
from probe_solid_potential_scaling import measure


def components(field, points, scales):
    value = field(points)
    grad = _gradient(value, points)
    dr = grad[:, :1]
    drr = _gradient(dr, points)[:, :1]
    rho = points[:, :1]
    safe = torch.where(rho == 0, torch.ones_like(rho), rho)
    laplacian = torch.where(rho == 0, 3*drr, drr+2*dr/safe)
    dt, radial = grad[:, 2:3], scales.diffusion_number*laplacian
    return dt, radial, dt-radial


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/"results/particle_startup_smoke_20260929T073928375126Z"
    prior = json.loads((source/"report.json").read_text())
    if prior["status"] != "STARTUP_SMOKE_ONLY":
        raise ValueError("Expected startup smoke")
    hashes = {source/"report.json": sha256(source/"report.json"), source/"checkpoint.pt": prior["checkpoint_sha256"]}
    hashes.update({root/k: h for k, h in prior["source_hashes"].items()})
    for p, h in hashes.items():
        if sha256(p) != h:
            raise ValueError(f"Source mismatch: {p}")
    torch.set_num_threads(1)
    saved = torch.load(source/"checkpoint.pt", weights_only=True)
    if saved["step"] != 3:
        raise ValueError("Expected three-step snapshot")
    model = restore_startup(saved)
    for label, key in (("training", "training_samples"), ("historical_check", "fresh_samples")):
        assert_equal_tree(measure(model, saved["samples"][key]), prior["final"][label])
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    radii = torch.tensor(np.unique(np.r_[np.linspace(0, 1, 81), 1-np.geomspace(1e-8, .1, 33)]), dtype=torch.float64)
    times = [1e-6, 1e-5, 1e-4, .001, .01, .1, 1.]
    rows, summaries = [], []
    for k, region in enumerate((0, 2)):
        field = model.cs[k]
        xs = torch.linspace(*model.bounds[region], 3, dtype=torch.float64)

        def bulk(p):
            root_time = (field.scales.diffusion_number*p[:, 2:3]).sqrt()
            features = torch.cat((p[:, :1].square(), p[:, 1:2],
                                  root_time/np.sqrt(field.fo_end), torch.zeros_like(root_time)), dim=1)
            return field.initial+field.beta*root_time*field.raw(features)

        def layer_part(p):
            return field(p)-bulk(p)

        for seconds in times:
            rr, xx = torch.meshgrid(radii, xs, indexing="ij")
            p = torch.stack((rr.ravel(), xx.ravel(), torch.full_like(rr.ravel(), seconds/field.scales.time_s)), dim=1).requires_grad_()
            full = components(field, p, field.scales)
            interior = components(bulk, p, field.scales)
            layer = components(layer_part, p, field.scales)
            torch.testing.assert_close(full[2], diffusion_residual(field, p, field.scales), rtol=1e-10, atol=1e-8)
            for a, b, c in zip(full, interior, layer):
                torch.testing.assert_close(a, b+c, rtol=1e-9, atol=1e-8)
            data = {}
            for name, values in (("full", full), ("bulk", interior), ("layer", layer)):
                for term, value in zip(("time", "radial", "residual"), values):
                    data[name+"_"+term] = value.detach().numpy().ravel()
            if not all(np.isfinite(a).all() for a in data.values()):
                raise ValueError("Nonfinite decomposition")
            coords = p.detach().numpy()
            masks = {"all": np.ones(len(p), dtype=bool), "bulk_rho_le_0_9": coords[:, 0] <= .9,
                     "surface_rho_gt_0_9": coords[:, 0] > .9}
            summary = {"electrode": ("n", "p")[k], "time_s": seconds, "regions": {}}
            for name, mask in masks.items():
                indices = np.flatnonzero(mask)
                peak = indices[np.argmax(np.abs(data["full_residual"][mask]))]
                summary["regions"][name] = {"peak_rho": float(coords[peak, 0]),
                    "peak_global_X": float(coords[peak, 1]),
                    "at_full_peak": {key: float(a[peak]) for key, a in data.items()},
                    "max_abs": {key: float(np.abs(a[mask]).max()) for key, a in data.items()},
                    "peak_residual_mol_m3_s": float(data["full_residual"][peak]*field.scales.concentration_mol_m3/field.scales.time_s)}
            summaries.append(summary)
            for i in range(len(p)):
                rows.append({"electrode": ("n", "p")[k], "time_s": seconds,
                             "rho": float(coords[i, 0]), "global_X": float(coords[i, 1]),
                             **{key: float(a[i]) for key, a in data.items()}})
    assert_equal_tree(frozen, model.state_dict())
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("Frozen input changed")
    output = root/"results"/("startup_bulk_diagnosis_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    with (output/"profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = [Path(__file__), root/"scripts/probe_potential_200.py", root/"scripts/probe_solid_potential_scaling.py",
               root/"scripts/prepare_dfn_baseline.py", *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report = {"status": "FROZEN_STARTUP_DECOMPOSITION_ONLY", "summaries": summaries,
        "times_s": times, "radial_points": len(radii), "x_points_per_electrode": 3,
        "scope": "Bulk is the same raw network evaluated with layer feature set to zero; layer is full minus bulk. Algebraic decomposition, not a separate PDE solution or physical domain split. Normalized derivatives; sampled maxima only.",
        "input_hashes": {str(p): h for p, h in hashes.items()},
        "source_hashes": {str(p.relative_to(root)): sha256(p) for p in sources},
        "metric_replay_verified": True, "operator_and_decomposition_verified": True,
        "model_and_artifacts_unchanged": True}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Frozen temporal/radial decomposition verified. No training. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
