"""Localize frozen-model diffusion and inventory errors without training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from dfn_pinn.potential_particle import PotentialParticle
from dfn_pinn.potential_audit import reference_identity
from dfn_pinn.coupled_audit import grid, pde_values, mean_values, target_values
from dfn_pinn.projection import gauss_legendre
from dfn_pinn.spherical_diffusion import _derivatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = json.loads((args.run/"report.json").read_text())
    for name, expected in report["source_hashes"].items():
        source = (root/name).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed: {name}")
    saved = torch.load(args.run/"checkpoint.pt", weights_only=True, map_location="cpu")
    c = saved["config"]
    if c != report["config"]:
        raise ValueError("Configuration mismatch")
    reference_dir = root/c["reference_run"]
    if reference_identity(reference_dir, c) != report["reference_hashes"]:
        raise ValueError("Reference changed")
    torch.set_num_threads(1)
    model = PotentialParticle(c)
    model.load_state_dict(saved["model"])
    model.eval()
    end = c["duration_s"]/c["time_reference_s"]
    r = torch.linspace(0, 1, 201, dtype=torch.float64)
    t = torch.linspace(0, end, 401, dtype=torch.float64)
    values = pde_values(model, grid(r, t)).reshape(len(r), len(t))
    ri, ti = divmod(int(values.abs().argmax()), len(t))
    local_r = torch.linspace(float(r[max(0, ri-1)]), float(r[min(len(r)-1, ri+1)]), 41, dtype=torch.float64)
    local_t = torch.linspace(float(t[max(0, ti-1)]), float(t[min(len(t)-1, ti+1)]), 81, dtype=torch.float64)
    local_points = grid(local_r, local_t)
    local_values = pde_values(model, local_points).flatten()
    idx = int(local_values.abs().argmax())
    peak = local_points[idx:idx+1].clone().requires_grad_()
    dr, drr, dt = _derivatives(model.concentration, peak)
    laplacian = 3*drr if float(peak[0, 0]) == 0 else drr+2*dr/peak[:, :1]
    diffusion = model.current.scales.diffusion_number*laplacian
    result = {"pde_peak": {"signed_residual": float(local_values[idx]), "rho": float(peak[0, 0].detach()),
                           "time_s": float(peak[0, 2].detach()*c["time_reference_s"]),
                           "time_derivative": float(dt.detach()), "diffusion_term": float(diffusion.detach())}}
    qr, wr = gauss_legendre(128, 0., 1.)
    qt, wt = gauss_legendre(128, 0., end)
    residual = pde_values(model, grid(qr, qt)).reshape(128, 128)
    energy = residual.square()*(3*qr.square()*wr)[:, None]*(wt/end)[None, :]
    total = energy.sum()
    result["radial_squared_residual_shares"] = {f"{a:g}_to_{b:g}": float(energy[(qr >= a) & (qr < b)].sum()/total) if total > 0 else 0.
                                                 for a, b in ((0., .8), (.8, .95), (.95, 1.))}
    with np.load(reference_dir/"reference.npz", allow_pickle=False) as data:
        seconds = torch.tensor(data["times_s"], dtype=torch.float64)
        reference_mean = torch.tensor(data["r320_tight_mean"], dtype=torch.float64)
    times = seconds/c["time_reference_s"]
    means = mean_values(model, times, 256)
    targets = target_values(model, times, 128)
    error = means-targets
    idx = int(error.abs().argmax())
    result["inventory_peak"] = {"time_s": float(seconds[idx]), "signed_mean_minus_target": float(error[idx]),
                                "mean_minus_reference_mean": float(means[idx]-reference_mean[idx]),
                                "current_target_minus_reference_mean": float(targets[idx]-reference_mean[idx])}
    result["training_losses"] = report["final_training_losses"]
    result["run"] = str(args.run.resolve())
    result["checkpoint_sha256"] = hashlib.sha256((args.run/"checkpoint.pt").read_bytes()).hexdigest()
    result["diagnostic_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["scope"] = "Sampled localization and quadrature attribution, not proof of optimization cause. No training."
    output = args.run/("diffusion_diagnosis_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"No training was run. Report: {output/'report.json'}")


if __name__ == "__main__":
    main()
