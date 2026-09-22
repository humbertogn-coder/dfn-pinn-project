"""Localize saved constant-flux PINN residuals without updating any weights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from train_constant_flux_particle import build_model, make_points, predict, sample_interior, MIN_TIME, END_TIME, SCALES
from dfn_pinn.spherical_diffusion import _derivatives


def residual_components(model, rho, tau, batch_size=512):
    """Evaluate c_tau and spherical diffusion separately on paired flat points."""
    rho, tau = np.asarray(rho), np.asarray(tau)
    if rho.ndim != 1 or tau.shape != rho.shape or not len(rho) or np.any(tau <= 0):
        raise ValueError("Require nonempty paired coordinates at positive times")
    parts = []
    for start in range(0, len(rho), batch_size):
        p = make_points(torch.tensor(rho[start:start+batch_size], dtype=torch.float64),
                        torch.tensor(tau[start:start+batch_size], dtype=torch.float64))
        dr, drr, dt = _derivatives(model, p)
        r = p[:, :1]
        safe_r = torch.where(r == 0, torch.ones_like(r), r)
        diffusion = SCALES.diffusion_number*torch.where(r == 0, 3*drr, drr+2*dr/safe_r)
        part = torch.cat((dt, diffusion), dim=1).detach().numpy()
        if not np.isfinite(part).all():
            raise ValueError("Nonfinite residual derivatives")
        parts.append(part)
    result = np.concatenate(parts)
    return result[:, 0], result[:, 1], result[:, 0]-result[:, 1]


def volume_time_rms(residual, rho, radial_weights, time_weights):
    """Normalized integral of R^2 with spherical volume and physical dt weights."""
    measure = 3*np.asarray(rho)**2*np.asarray(radial_weights)
    weights = measure[:, None]*np.asarray(time_weights)[None, :]
    if np.shape(residual) != weights.shape or not np.isfinite(residual).all():
        raise ValueError("Invalid weighted residual array")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid quadrature measure")
    return float(np.sqrt(np.sum(weights*np.asarray(residual)**2)/weights.sum()))


def physical_quadrature(radial_order, time_order):
    nodes, weights = np.polynomial.legendre.leggauss(radial_order)
    rho, wr = (nodes+1)/2, weights/2
    nodes, weights = np.polynomial.legendre.leggauss(time_order)
    edges = np.geomspace(MIN_TIME, END_TIME, 17)
    # Logarithmic panel placement, but weights integrate physical dt, not d(log t).
    times = np.concatenate([(a+b)/2+(b-a)*nodes/2 for a, b in zip(edges[:-1], edges[1:])])
    wt = np.concatenate([(b-a)*weights/2 for a, b in zip(edges[:-1], edges[1:])])
    return rho, wr, times, wt


def resolve_run(results, variant, explicit=None, sampling="legacy"):
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        candidates = sorted(results.glob("constant_flux_pinn_*"), reverse=True)
    for path in candidates:
        report_path = path/"report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report["config"].get("variant", "baseline") == variant
                and report["config"].get("sampling", "legacy") == sampling
                and report["config"].get("mass_weight", 0) == 0):
            return path, report
    raise ValueError(f"No completed {variant} run found")


def load_run(path, report, variant):
    expected_architecture = [2 if variant == "baseline" else 3, 32, 32, 1]
    if (report["architecture"] != expected_architecture or report["dtype"] != "float64"
            or report["torch_version"] != torch.__version__ or report["interior_points"] != 512
            or report["minimum_positive_time"] != MIN_TIME or report["time_interval"] != [0, END_TIME]
            or report["diffusion_number"] != 1):
        raise ValueError("Unsupported run settings or PyTorch version")
    # The old trainer generated these points after initializing the baseline MLP.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(report["config"]["seed"])
        model = build_model(variant)
        initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        sample = torch.rand(512, 2, dtype=torch.float64)
        r, t = sample_interior(sample, report["config"].get("sampling", "legacy"))
    rho, tau = r.numpy(), t.numpy()
    if "initial_weights_sha256" in report and report["initial_weights_sha256"] != initial_hash:
        raise ValueError("Reconstructed initial weights differ")
    if "training_points_sha256" in report:
        artifact = path/"training_points.npz"
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != report["training_points_sha256"]:
            raise ValueError("Training point checksum mismatch")
        with np.load(artifact, allow_pickle=False) as saved:
            points = saved["interior"]
            if points.shape != (512, 3) or not np.isfinite(points).all():
                raise ValueError("Invalid saved training points")
            if not np.array_equal(points[:, 0], rho) or not np.array_equal(points[:, 2], tau):
                raise ValueError("Saved/reconstructed collocation mismatch")
            rho, tau = points[:, 0], points[:, 2]
    elif report["config"].get("sampling", "legacy") != "legacy":
        raise ValueError("New samplers must archive training points")
    model.load_state_dict(torch.load(path/"model.pt", map_location="cpu", weights_only=True), strict=True)
    model.eval()
    with np.load(path/"evaluation.npz", allow_pickle=False) as saved:
        rr, tt = np.meshgrid(saved["rho"], saved["tau"], indexing="ij")
        with torch.no_grad():
            actual = predict(model, make_points(torch.tensor(rr.ravel()), torch.tensor(tt.ravel()))).numpy().reshape(rr.shape)
        discrepancy = float(np.abs(actual-saved["prediction"]).max())
        if not np.allclose(actual, saved["prediction"], atol=1e-12, rtol=1e-12):
            raise ValueError("Loaded model does not reproduce archived predictions")
    return model, rho, tau, discrepancy


def stats(values):
    return {"count": int(np.size(values)), "rms": float(np.sqrt(np.mean(values**2))),
            "max_abs": float(np.max(np.abs(values)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--startup", type=Path)
    parser.add_argument("--sampling-comparison", action="store_true",
                        help="Compare startup legacy vs full-radius sampling, not architectures")
    parser.add_argument("--surface-layer-comparison", action="store_true",
                        help="Compare startup full-radius vs targeted surface-layer sampling")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    if args.sampling_comparison and args.surface_layer_comparison:
        parser.error("Choose one sampling comparison")
    if args.surface_layer_comparison:
        args.sampling_comparison = True
    if args.sampling_comparison:
        if args.baseline or args.startup:
            parser.error("Explicit architecture paths cannot be combined with --sampling-comparison")
        pair = ("full_radius", "surface_layer") if args.surface_layer_comparison else ("legacy", "full_radius")
        paths = {s: resolve_run(root/"results", "startup", sampling=s) for s in pair}
    else:
        paths = {v: resolve_run(root/"results", v, getattr(args, v)) for v in ("baseline", "startup")}
    left, right = (item[1] for item in paths.values())
    for key in ("seed", "adam_steps", "lbfgs_steps"):
        if left["config"][key] != right["config"][key]:
            raise ValueError(f"Comparison settings differ: {key}")
    if left["loss_weights"] != right["loss_weights"]:
        raise ValueError("Comparison loss weights differ")
    # Reproduce the previous grid and use a denser, surface-enriched map separately.
    rho = np.unique(np.r_[np.linspace(0, 1, 201), 1-np.geomspace(1e-5, .2, 80)])
    tau = np.geomspace(MIN_TIME, END_TIME, 161)
    rr, tt = np.meshgrid(rho, tau, indexing="ij")
    reports, arrays, training_coordinates = {}, {"rho": rho, "tau": tau}, []
    for variant, (path, report) in paths.items():
        print(f"Reading {variant}: {path}", flush=True)
        model, train_r, train_t, gap = load_run(path, report, report["config"].get("variant", "baseline"))
        training_coordinates.append(np.stack((train_r, train_t)))
        train_residual = residual_components(model, train_r, train_t)[2]
        dt, diffusion, residual = [v.reshape(rr.shape) for v in residual_components(model, rr.ravel(), tt.ravel())]
        peak = np.unravel_index(np.abs(residual).argmax(), residual.shape)
        old_r, old_t = np.meshgrid(np.linspace(0, 1, 101), np.geomspace(MIN_TIME, END_TIME, 120), indexing="ij")
        old_rms = stats(residual_components(model, old_r.ravel(), old_t.ravel())[2])["rms"]
        if not np.isclose(old_rms, report["metrics"]["positive_time_pde_rms"], rtol=1e-9, atol=1e-12):
            raise ValueError("Archived PDE RMS was not reproduced")
        weighted = {}
        for nr, nt in ((64, 4), (128, 8)):
            qr, wr, qt, wt = physical_quadrature(nr, nt)
            qrr, qtt = np.meshgrid(qr, qt, indexing="ij")
            qres = residual_components(model, qrr.ravel(), qtt.ravel())[2].reshape(qrr.shape)
            weighted[f"r{nr}_t{nt}x16"] = volume_time_rms(qres, qr, wr, wt)
        regions = {}
        for time_name, time_mask in (("early", tt <= 1e-3), ("later", tt > 1e-3)):
            for radius_name, radius_mask in (("inner", rr < .9), ("outer", rr >= .9)):
                name = f"{time_name}_{radius_name}"
                mask = time_mask & radius_mask
                training_mask = ((train_t <= 1e-3) if time_name == "early" else (train_t > 1e-3))
                training_mask &= ((train_r < .9) if radius_name == "inner" else (train_r >= .9))
                regions[name] = stats(residual[mask]) | {"training_count": int(training_mask.sum())}
        entry = {"run": str(path), "model_sha256": hashlib.sha256((path/"model.pt").read_bytes()).hexdigest(),
                 "prediction_reload_max_difference": gap, "training": stats(train_residual),
                 "original_grid_rms": old_rms, "dense_map": stats(residual),
                 "peak": {"rho": float(rho[peak[0]]), "tau": float(tau[peak[1]]),
                          "residual": float(residual[peak]), "time_derivative": float(dt[peak]),
                          "diffusion_term": float(diffusion[peak])},
                 "volume_time_rms": weighted, "regions": regions}
        entry["early_inner_training_count"] = int(((train_r < .7)&(train_t <= 1e-4)).sum())
        reports[variant] = entry
        arrays.update({f"{variant}_residual": residual, f"{variant}_time_derivative": dt,
                       f"{variant}_diffusion": diffusion, f"{variant}_train_rho": train_r,
                       f"{variant}_train_tau": train_t, f"{variant}_train_residual": train_residual})
        print(f"{variant}: training RMS={entry['training']['rms']:.6e}, original grid RMS={old_rms:.6e}")
        print(f"  Dense peak: |R|={abs(residual[peak]):.6e}, rho={rho[peak[0]]:.6f}, tau={tau[peak[1]]:.6e}")
        print(f"  Volume/time RMS: coarse={weighted['r64_t4x16']:.6e}, fine={weighted['r128_t8x16']:.6e}", flush=True)
    identical = np.array_equal(*training_coordinates)
    if not args.sampling_comparison and not identical:
        raise ValueError("Reconstructed training points differ between variants")
    if args.sampling_comparison and identical:
        raise ValueError("Sampling experiment unexpectedly used identical points")
    output = root/"results"/("flux_residual_diagnosis_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/"residual_maps.npz", **arrays)
    payload = {"variants": reports, "training_points_identical": identical,
               "comparison": "sampling" if args.sampling_comparison else "representation",
               "scope": "No retraining. Same-seed checkpoints and existing evaluation reproduced. "
                        "Training points reconstructed from current source/RNG, not archived points. "
                        "Map includes boundaries as diagnostic limits; not all map points are interior PDE points. "
                        "Physical RMS uses normalized 3*rho^2 drho dt on [0,1] x [1e-5,0.02]. "
                        "Two quadratures test sensitivity, not an exact integral bound. Regional/map RMS "
                        "depends on sampling density. No claims for 0<t<1e-5 or other seeds."}
    (output/"report.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    upper = max(abs(arrays[f"{v}_residual"]).max() for v in reports)
    for ax, variant in zip(axes, reports):
        color = ax.pcolormesh(tau, rho, np.maximum(abs(arrays[f"{variant}_residual"]), 1e-6),
                              norm=LogNorm(vmin=1e-6, vmax=max(upper, 1e-5)), shading="auto")
        ax.scatter(arrays[f"{variant}_train_tau"], arrays[f"{variant}_train_rho"], s=2, c="white", alpha=.35)
        ax.set(xscale="log", xlabel="Dimensionless time", ylabel="r/R", title=f"{variant}: absolute PDE residual")
    fig.colorbar(color, ax=axes, label="|c_tau - spherical diffusion| (shared scale)")
    fig.savefig(output/"residual_maps.png", dpi=180)
    plt.close(fig)
    print(f"No training was run. Output directory: {output}")


if __name__ == "__main__":
    main()
