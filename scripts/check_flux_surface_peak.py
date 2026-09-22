"""Refine early surface residual diagnostics for saved startup checkpoints."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diagnose_flux_residual import resolve_run, load_run, residual_components


def refined_grid():
    """Nested grids use strides 4,2,1; all stay within the trained time domain."""
    return 1-.1*np.linspace(1, 0, 321)**2, np.geomspace(1e-5, 1e-3, 321)


def peak_record(residual, dt, diffusion, rho, tau):
    index = np.unravel_index(np.abs(residual).argmax(), residual.shape)
    return {"max_abs": float(abs(residual[index])), "rho": float(rho[index[0]]),
            "tau": float(tau[index[1]]), "signed_residual": float(residual[index]),
            "time_derivative": float(dt[index]), "diffusion_term": float(diffusion[index])}


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    sources = {s: resolve_run(root/"results", "startup", sampling=s) for s in ("legacy", "full_radius")}
    left, right = (item[1] for item in sources.values())
    for key in ("seed", "adam_steps", "lbfgs_steps"):
        if left["config"][key] != right["config"][key]:
            raise ValueError(f"Incompatible runs: {key}")
    rho, tau = refined_grid()
    rr, tt = np.meshgrid(rho, tau, indexing="ij")
    arrays, reports = {"rho": rho, "tau": tau}, {}
    for sampling, (path, report) in sources.items():
        print(f"Reading {sampling}: {path}", flush=True)
        model, train_r, train_t, gap = load_run(path, report, "startup")
        dt, diffusion, residual = [a.reshape(rr.shape) for a in residual_components(model, rr.ravel(), tt.ravel())]
        refinement = []
        for stride in (4, 2, 1):
            record = peak_record(residual[::stride, ::stride], dt[::stride, ::stride],
                                 diffusion[::stride, ::stride], rho[::stride], tau[::stride])
            record["grid_shape"] = list(residual[::stride, ::stride].shape)
            refinement.append(record)
            print(f"{sampling} {record['grid_shape']}: max={record['max_abs']:.6e}, "
                  f"rho={record['rho']:.9f}, tau={record['tau']:.6e}", flush=True)
        interior = peak_record(residual[:-1], dt[:-1], diffusion[:-1], rho[:-1], tau)
        # Probe fixed interior distances; refinement alone could keep a boundary maximum.
        distances = np.array([0., 1e-6, 1e-5, 1e-4, 1e-3, .003, .01, .03, .1])
        probe_t = np.array([1e-5, 2e-5, 5e-5, 1e-4, 1e-3])
        pr, pt = np.meshgrid(1-distances, probe_t, indexing="ij")
        pd, pf, pres = [a.reshape(pr.shape) for a in residual_components(model, pr.ravel(), pt.ravel())]
        probes = [{"distance_from_surface": float(d), "tau": float(t),
                   "time_derivative": float(pd[i, j]), "diffusion_term": float(pf[i, j]),
                   "residual": float(pres[i, j])}
                  for i, d in enumerate(distances) for j, t in enumerate(probe_t)]
        bands = []
        for distance in (.001, .01, .1):
            mask = (rr >= 1-distance)&(tt <= 1e-4)
            train_mask = (train_r >= 1-distance)&(train_t <= 1e-4)
            bands.append({"radial_width": distance, "time_end": 1e-4,
                          "training_points": int(train_mask.sum()),
                          "sampled_max_abs": float(abs(residual[mask]).max())})
        reports[sampling] = {"source": str(path), "prediction_reload_max_difference": gap,
                             "model_sha256": hashlib.sha256((path/"model.pt").read_bytes()).hexdigest(),
                             "refinement": refinement, "strict_interior_peak": interior,
                             "probes": probes, "training_coverage": bands}
        arrays.update({sampling+"_residual": residual, sampling+"_time_derivative": dt,
                       sampling+"_diffusion": diffusion, sampling+"_train_rho": train_r,
                       sampling+"_train_tau": train_t})
        print(f"  Strict interior max={interior['max_abs']:.6e} at rho={interior['rho']:.9f}")
        print(f"  At peak: c_tau={refinement[-1]['time_derivative']:.6e}, "
              f"diffusion={refinement[-1]['diffusion_term']:.6e}")
    output = root/"results"/("flux_surface_peak_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    (output/"report.json").write_text(json.dumps({"runs": reports,
        "scope": "No retraining. Nested sampled grids on rho=[0.9,1], t=[1e-5,1e-3]. "
        "Maxima on the boundary are diagnostic PDE limits, distinct from the flux BC. "
        "Strict interior and fixed-distance probes test localization. Stable sampled "
        "maxima do not certify a continuous bound or behavior before t=1e-5. "
        "This audits the trained functions, not finite-difference mesh convergence."}, indent=2)+"\n", encoding="utf-8")
    np.savez_compressed(output/"samples.npz", **arrays)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    bound = max(np.abs(arrays[s+"_residual"]).max() for s in reports)
    for column, sampling in enumerate(reports):
        residual = arrays[sampling+"_residual"]
        color = axes[0, column].pcolormesh(tau, 1-rho, residual, shading="auto",
                                         cmap="RdBu_r", vmin=-bound, vmax=bound)
        axes[0, column].set(xscale="log", xlabel="Dimensionless time", ylabel="1 - r/R",
                            title=f"{sampling}: signed residual", ylim=(.03, 0))
        axes[1, column].plot(1-rho, arrays[sampling+"_time_derivative"][:, 0], label="Time derivative")
        axes[1, column].plot(1-rho, arrays[sampling+"_diffusion"][:, 0], label="Diffusion")
        axes[1, column].plot(1-rho, residual[:, 0], "--", label="Residual")
        axes[1, column].set(xlabel="1 - r/R", ylabel="Dimensionless PDE terms",
                            xlim=(0, .03), title="At t = 1e-5")
        axes[1, column].legend()
    fig.colorbar(color, ax=axes[0, :], label="c_tau - diffusion (shared scale)")
    fig.savefig(output/"surface_peak.png", dpi=170)
    plt.close(fig)
    print(f"No training was run. Output directory: {output}")


if __name__ == "__main__":
    main()
