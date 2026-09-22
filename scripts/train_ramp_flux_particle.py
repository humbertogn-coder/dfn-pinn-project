"""Train the synthetic linear-ramp particle PINN with hard discrete inventory."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from check_ramp_flux_sphere import ramp_solution, eigenvalues, SLOPE, END_TIME
from train_constant_flux_particle import build_model, sample_interior, make_points, predict, MIN_TIME, SCALES
from dfn_pinn.particle_inventory import UnitFluxInventoryProjection
from dfn_pinn.projection import gauss_legendre
from dfn_pinn.spherical_diffusion import diffusion_residual, center_residual
from diagnose_flux_residual import physical_quadrature, residual_components, volume_time_rms


def target_mean(t):
    return 1-1.5*SLOPE*t**2


def flux(t):
    return SLOPE*t


class RampInventoryProjection(UnitFluxInventoryProjection):
    """Reuse the audited discrete correction, replacing only its target mean."""

    def forward(self, points):
        t = points[:, 2:3]
        # Cancel the inherited unit-step target; retain the full time graph.
        return super().forward(points)+3*t-1.5*SLOPE*t**2


def build_ramp_model():
    return RampInventoryProjection(build_model("startup"), 64)


def backward_terms(model, interior, initial, surface, batch_size=64):
    losses = []
    for kind, points, weight in (("pde", interior, 1), ("initial", initial, 100), ("surface", surface, 1)):
        total = points.new_zeros(())
        for part in points.detach().split(batch_size):
            part = part.clone().requires_grad_()
            if kind == "pde":
                residual = diffusion_residual(model, part, SCALES)
            elif kind == "initial":
                residual = model(part)-1
            else:
                dr = torch.autograd.grad(model(part).sum(), part, create_graph=True)[0][:, :1]
                residual = dr+flux(part[:, 2:3])
            contribution = residual.square().sum()/len(points)
            if not torch.isfinite(contribution):
                raise FloatingPointError("Nonfinite ramp training loss")
            (weight*contribution).backward()
            total = total+contribution.detach()
        losses.append(total)
    return tuple(losses)


def assess_ramp(model):
    r = np.linspace(0, 1, 101)
    t = np.r_[0., np.geomspace(MIN_TIME, END_TIME, 120)]
    rr, tt = np.meshgrid(r, t, indexing="ij")
    with torch.no_grad():
        prediction = predict(model, make_points(torch.tensor(rr.ravel()), torch.tensor(tt.ravel()))).reshape(rr.shape).numpy()
    roots = eigenvalues(4096)
    reference = ramp_solution(r, t, roots)
    gap = float(np.abs(reference-ramp_solution(r, t, roots[:2048])).max())
    if gap > 1e-9:
        raise ValueError("Ramp reference truncation check failed")
    means = {}
    for order in (128, 256):
        qr, w = gauss_legendre(order, 0., 1.)
        qrr, qtt = torch.meshgrid(qr, torch.tensor(t), indexing="ij")
        with torch.no_grad():
            c = predict(model, make_points(qrr.ravel(), qtt.ravel())).reshape(qrr.shape)
            means[order] = (3*qr[:, None]**2*w[:, None]*c).sum(0).numpy()
    positive_t = torch.tensor(t[1:])
    boundary = make_points(torch.ones_like(positive_t), positive_t)
    dr = torch.autograd.grad(model(boundary).sum(), boundary)[0][:, 0].detach().numpy()
    center = center_residual(model, make_points(torch.zeros_like(positive_t), positive_t)).detach().numpy()
    residual = residual_components(model, rr[:, 1:].ravel(), tt[:, 1:].ravel(), batch_size=64)[2]
    weighted = []
    for nr, nt in ((64, 4), (128, 8)):
        qr, w, qt, wt = physical_quadrature(nr, nt)
        qrr, qtt = np.meshgrid(qr, qt, indexing="ij")
        res = residual_components(model, qrr.ravel(), qtt.ravel(), batch_size=64)[2].reshape(qrr.shape)
        weighted.append(volume_time_rms(res, qr, w, wt))
    error = np.abs(prediction-reference)
    peak = np.unravel_index(error[:, 1:].argmax(), error[:, 1:].shape)
    metrics = {"initial_max_error": float(error[:, 0].max()),
               "positive_time_max_error": float(error[:, 1:].max()),
               "positive_time_rms_error": float(np.sqrt(np.mean(error[:, 1:]**2))),
               "surface_max_error": float(error[-1, 1:].max()),
               "peak_error_rho": float(r[peak[0]]), "peak_error_tau": float(t[peak[1]+1]),
               "max_surface_flux_residual": float(np.abs(dr+flux(t[1:])).max()),
               "center_max_gradient": float(np.abs(center).max()),
               "max_volume_mean_balance_error": float(np.abs(means[128]-target_mean(t)).max()),
               "mass_check_256_max_error": float(np.abs(means[256]-target_mean(t)).max()),
               "mass_check_128_256_max_difference": float(np.abs(means[128]-means[256]).max()),
               "positive_time_pde_rms": float(np.sqrt(np.mean(residual**2))),
               "physical_pde_rms": weighted[1], "physical_pde_quadrature_gap": abs(weighted[1]-weighted[0]),
               "series_2048_4096_max_difference": gap,
               "minimum_concentration": float(prediction.min()), "maximum_concentration": float(prediction.max())}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite validation metric")
    return metrics, {"rho": r, "tau": t, "prediction": prediction, "reference": reference,
                     "volume_mean_128": means[128], "volume_mean_256": means[256],
                     "exact_mean": target_mean(t), "surface_gradient": dr, "flux": flux(t)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adam-steps", type=int, default=2000)
    parser.add_argument("--lbfgs-steps", type=int, default=300)
    args = parser.parse_args()
    if args.adam_steps < 0 or args.lbfgs_steps < 0:
        parser.error("Iteration budgets must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    output = root/"results"/("ramp_flux_pinn_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = build_ramp_model()
    initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    r, t = sample_interior(torch.rand(512, 2, dtype=torch.float64), "full_radius")
    interior = make_points(r, t)
    initial = make_points(torch.linspace(0, 1, 101, dtype=torch.float64), torch.zeros(101, dtype=torch.float64))
    bt = torch.cat((torch.linspace(MIN_TIME, END_TIME, 64, dtype=torch.float64),
                    torch.logspace(np.log10(MIN_TIME), np.log10(END_TIME), 64, dtype=torch.float64)))
    surface = make_points(torch.ones_like(bt), bt)
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        losses = backward_terms(model, interior, initial, surface)
        loss = losses[0]+100*losses[1]+losses[2]
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite ramp loss or parameter gradient")
        history.append([float(loss), *[float(value) for value in losses]])
        return loss

    print("Training linear-ramp sphere: q(t)=50*t, mean=1-75*t^2, float64 CPU.", flush=True)
    print(f"Seed {args.seed}; full_radius; projection order 64; budgets {args.adam_steps}/{args.lbfgs_steps}.", flush=True)
    started = time.perf_counter()
    for step in range(args.adam_steps):
        optimizer.step(closure)
        if (step+1) % 100 == 0:
            print(f"Adam {step+1}: loss={history[-1][0]:.6e}", flush=True)
    adam_evaluations = len(history)
    if args.lbfgs_steps:
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=args.lbfgs_steps,
                                     line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-12)
        optimizer.step(closure)
    elapsed = time.perf_counter()-started
    print("Evaluating against ramp series and independent mass quadratures...", flush=True)
    metrics, arrays = assess_ramp(model)
    torch.save(model.state_dict(), output/"model.pt")
    reloaded = build_ramp_model()
    reloaded.load_state_dict(torch.load(output/"model.pt", weights_only=True, map_location="cpu"), strict=True)
    rr, tt = np.meshgrid(arrays["rho"], arrays["tau"], indexing="ij")
    with torch.no_grad():
        replay = predict(reloaded, make_points(torch.tensor(rr.ravel()), torch.tensor(tt.ravel()))).numpy().reshape(rr.shape)
    if not np.allclose(replay, arrays["prediction"], atol=1e-12, rtol=1e-12):
        raise ValueError("Checkpoint reload failed")
    np.savez_compressed(output/"evaluation.npz", **arrays)
    np.savez_compressed(output/"training_points.npz", interior=interior.detach().numpy(),
                        initial=initial.detach().numpy(), surface=surface.detach().numpy())
    np.savetxt(output/"loss_history.csv", np.asarray(history).reshape(-1, 4), delimiter=",",
               header="total,pde,initial,surface", comments="")
    report = {"config": vars(args), "protocol": "linear_ramp", "slope": SLOPE, "end_time": END_TIME,
              "minimum_positive_time": MIN_TIME, "dtype": "float64", "device": "cpu", "threads": 1,
              "torch_version": torch.__version__, "architecture": [3, 32, 32, 1], "sampling": "full_radius",
              "interior_points": 512, "initial_points": 101, "surface_points": 128,
              "projection_order": 64, "independent_mass_orders": [128, 256],
              "loss_weights": {"pde": 1, "initial": 100, "surface": 1},
              "initial_weights_sha256": initial_hash, "training_s": elapsed,
              "adam_evaluations": adam_evaluations, "lbfgs_evaluations": len(history)-adam_evaluations,
              "metrics": metrics, "checkpoint_reload_max_difference": float(np.abs(replay-arrays["prediction"]).max()),
              "model_sha256": hashlib.sha256((output/"model.pt").read_bytes()).hexdigest(),
              "training_points_sha256": hashlib.sha256((output/"training_points.npz").read_bytes()).hexdigest(),
              "scope": "Synthetic linear ramp, one seed. No interior labels or soft mass loss. "
                       "Flux errors scaled by peak flux 1, not instantaneous flux near zero. "
                       "Final extracted mean is 0.03, not the step benchmark's 0.06. "
                       "Sampled errors, no automatic acceptance or DFN claim."}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    ax[0].plot(arrays["tau"], arrays["reference"][-1], label="Ramp series")
    ax[0].plot(arrays["tau"], arrays["prediction"][-1], "--", label="PINN")
    ax[0].set(title="Surface concentration", xlabel="Dimensionless time")
    ax[0].legend()
    ax[1].plot(arrays["tau"], arrays["exact_mean"], label="1 - 75 t^2")
    ax[1].plot(arrays["tau"], arrays["volume_mean_256"], "--", label="PINN, 256 nodes")
    ax[1].set(title="Volume mean", xlabel="Dimensionless time")
    ax[1].legend()
    ax[2].semilogx(arrays["tau"][1:], np.max(np.abs(arrays["prediction"]-arrays["reference"]), axis=0)[1:])
    ax[2].set(title="Maximum radial error", xlabel="Dimensionless time")
    fig.savefig(output/"benchmark.png", dpi=160)
    plt.close(fig)
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
