"""Physics-only PINN pilot for a sphere with suddenly applied unit outward flux."""

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from check_constant_flux_sphere import eigenvalues, series_solution
from train_single_particle import make_points
from dfn_pinn.projection import gauss_legendre
from dfn_pinn.particle_inventory import UnitFluxInventoryProjection
from dfn_pinn.spherical_diffusion import ParticleScales, diffusion_residual, center_residual


END_TIME = .02
MIN_TIME = 1e-5
SCALES = ParticleScales(1., 1., 1., 1.)


class FluxNet(nn.Module):
    """Center symmetry is hard; initial concentration and surface flux are soft."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32),
                                    nn.Tanh(), nn.Linear(32, 1)).double()

    def forward(self, p):
        features = torch.stack((2*p[:, 0]**2-1, 2*p[:, 2]/END_TIME-1), dim=-1)
        return 1+.2*self.layers(features)


class StartupFluxNet(nn.Module):
    """Hard initial value, square-root time and a surface-layer input.

    The exact-time-zero branch defines values only, not the right time
    derivative at the incompatible corner. Physics is evaluated for t>0.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32),
                                    nn.Tanh(), nn.Linear(32, 1)).double()

    def forward(self, p):
        tau = p[:, 2:3]
        positive = tau > 0
        root = torch.sqrt(torch.where(positive, tau, torch.ones_like(tau)))
        rho2 = p[:, :1]**2
        layer = torch.exp(-(1-rho2)/(4*root))
        features = torch.cat((2*rho2-1, 2*root/np.sqrt(END_TIME)-1, 2*layer-1), dim=1)
        return torch.where(positive, 1+root*self.layers(features), 1+p[:, :1]*0)


def build_model(variant):
    """Preserve baseline random sampling despite different parameter counts."""
    baseline = FluxNet()
    if variant == "baseline":
        return baseline
    if variant not in ("startup", "inventory"):
        raise ValueError("Unknown variant")
    sampling_state = torch.get_rng_state()
    model = StartupFluxNet()
    torch.set_rng_state(sampling_state)
    return UnitFluxInventoryProjection(model, 64) if variant == "inventory" else model


def terms(model, interior, initial, surface):
    pde = diffusion_residual(model, interior, SCALES).square().mean()
    ic = (model(initial)-1).square().mean()
    dr = torch.autograd.grad(model(surface).sum(), surface, create_graph=True)[0][:, 0]
    flux = (dr+1).square().mean()
    return pde, ic, flux


def mass_loss(model, times, radial_order=64):
    """Mean inventory error scaled by final extracted inventory, not local time."""
    r, w = gauss_legendre(radial_order, 0., 1.)
    rr, tt = torch.meshgrid(r, times, indexing="ij")
    points = torch.stack((rr.ravel(), torch.zeros_like(rr).ravel(), tt.ravel()), dim=1)
    values = model(points).reshape(rr.shape)
    means = (3*r[:, None]**2*w[:, None]*values).sum(0)
    return ((means-(1-3*times))/(3*END_TIME)).square().mean()


def predict(model, points):
    """Bound projected inference memory without coupling separate query rows."""
    if not isinstance(model, UnitFluxInventoryProjection):
        return model(points)
    return torch.cat([model(part) for part in points.split(128)])


def projected_backward(model, interior, initial, surface, batch_size=64):
    """Accumulate the full objective gradient in chunks, not minibatch steps."""
    losses = []
    for kind, points, weight in (("pde", interior, 1), ("initial", initial, 100),
                                  ("surface", surface, 1)):
        total = points.new_zeros(())
        for part in points.detach().split(batch_size):
            part = part.clone().requires_grad_()
            if kind == "pde":
                residual = diffusion_residual(model, part, SCALES)
            elif kind == "initial":
                residual = model(part)-1
            else:
                residual = torch.autograd.grad(model(part).sum(), part, create_graph=True)[0][:, :1]+1
            contribution = residual.square().sum()/len(points)
            if not torch.isfinite(contribution):
                raise FloatingPointError("Nonfinite projected training loss")
            (weight*contribution).backward()
            total = total+contribution.detach()
        losses.append(total)
    return tuple(losses)


def sample_interior(sample, sampling="legacy"):
    """Map the same 512 uniform pairs to a fixed collocation design."""
    if sample.shape != (512, 2) or sampling not in ("legacy", "full_radius", "surface_layer"):
        raise ValueError("Expected 512 pairs and a known sampling design")
    rho = .001+.999*sample[:, 0]
    rho[256:] = 1-.3*sample[256:, 0]
    tau = MIN_TIME+(END_TIME-MIN_TIME)*sample[:, 1]
    tau[256:] = MIN_TIME*(END_TIME/MIN_TIME)**sample[256:, 1]
    if sampling in ("full_radius", "surface_layer"):
        index = torch.arange(128, device=sample.device)
        rho[384:] = .001+.999*((index % 16+sample[384:, 0])/16)
        tau[384:] = MIN_TIME*(END_TIME/MIN_TIME)**((index // 16+sample[384:, 1])/8)
    if sampling == "surface_layer":
        index = torch.arange(64, device=sample.device)
        distance = 1e-6*(1e-2/1e-6)**((index % 8+sample[320:384, 0])/8)
        rho[320:384] = 1-distance
        tau[320:384] = MIN_TIME*(1e-3/MIN_TIME)**((index // 8+sample[320:384, 1])/8)
    return rho, tau


def assess(model):
    radii = np.linspace(0, 1, 101)
    times = np.r_[0., np.geomspace(MIN_TIME, END_TIME, 120)]
    rr, tt = np.meshgrid(radii, times, indexing="ij")
    p = make_points(torch.tensor(rr.ravel()), torch.tensor(tt.ravel()))
    with torch.no_grad():
        prediction = predict(model, p).reshape(rr.shape).numpy()
    roots = eigenvalues(1024)
    exact = series_solution(radii, times, roots)
    series_gap = np.abs(exact-series_solution(radii, times, roots[:512])).max()
    if series_gap > 1e-9:
        raise ValueError("Analytic series truncation audit failed")
    r, w = gauss_legendre(128, 0., 1.)
    r2, t2 = torch.meshgrid(r, torch.tensor(times), indexing="ij")
    with torch.no_grad():
        values = predict(model, make_points(r2.ravel(), t2.ravel())).reshape(r2.shape)
        means = (3*values*r[:, None]**2*w[:, None]).sum(0).numpy()
    tau = torch.tensor(times[1:])
    boundary = make_points(torch.ones_like(tau), tau)
    dr = torch.autograd.grad(model(boundary).sum(), boundary)[0][:, 0].detach().numpy()
    center = center_residual(model, make_points(torch.zeros_like(tau), tau)).detach()
    error = prediction-exact
    positive = np.abs(error[:, 1:])
    peak = np.unravel_index(positive.argmax(), positive.shape)
    positive_points = make_points(torch.tensor(rr[:, 1:].ravel()), torch.tensor(tt[:, 1:].ravel()))
    if isinstance(model, UnitFluxInventoryProjection):
        residual = torch.cat([diffusion_residual(model, part.detach().requires_grad_(), SCALES).detach()
                              for part in positive_points.split(64)]).numpy()
    else:
        residual = diffusion_residual(model, positive_points, SCALES).detach().numpy()
    metrics = {
        "initial_max_error": float(np.abs(error[:, 0]).max()),
        "positive_time_max_error": float(positive.max()),
        "positive_time_rms_error": float(np.sqrt(np.mean(error[:, 1:]**2))),
        "peak_error_rho": float(radii[peak[0]]), "peak_error_tau": float(times[peak[1]+1]),
        "surface_max_error": float(positive[-1].max()),
        "startup_max_error_tau_le_1e_3": float(np.abs(error[:, (times>0)&(times<=1e-3)]).max()),
        "later_max_error_tau_gt_1e_3": float(np.abs(error[:, times>1e-3]).max()),
        "max_surface_flux_residual": float(np.abs(dr+1).max()),
        "center_max_gradient": float(center.abs().max()),
        "max_volume_mean_balance_error": float(np.abs(means-(1-3*times)).max()),
        "max_volume_mean_change_error": float(np.abs(means-means[0]+3*times).max()),
        "positive_time_pde_rms": float(np.sqrt(np.mean(residual**2))),
        "series_512_1024_max_difference": float(series_gap),
    }
    if isinstance(model, UnitFluxInventoryProjection):
        qr, qw = gauss_legendre(256, 0., 1.)
        qrr, qtt = torch.meshgrid(qr, torch.tensor(times), indexing="ij")
        with torch.no_grad():
            qvalues = predict(model, make_points(qrr.ravel(), qtt.ravel())).reshape(qrr.shape)
            fine_means = (3*qvalues*qr[:, None]**2*qw[:, None]).sum(0).numpy()
        metrics["mass_check_256_max_error"] = float(np.abs(fine_means-(1-3*times)).max())
        metrics["mass_check_128_256_max_difference"] = float(np.abs(fine_means-means).max())
    return metrics, dict(rho=radii, tau=times, prediction=prediction, reference=exact,
                         volume_mean=means, exact_volume_mean=1-3*times, surface_gradient=dr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=("baseline", "startup", "inventory"), default="baseline")
    parser.add_argument("--sampling", choices=("legacy", "full_radius", "surface_layer"), default="legacy")
    parser.add_argument("--adam-steps", type=int, default=2000)
    parser.add_argument("--lbfgs-steps", type=int, default=300)
    parser.add_argument("--output", help="New output directory; existing directories are rejected")
    parser.add_argument("--mass-weight", type=float, default=0., help="Weight of normalized mean-inventory loss")
    args = parser.parse_args()
    if args.adam_steps < 0 or args.lbfgs_steps < 0:
        parser.error("Iteration counts must be nonnegative")
    if not np.isfinite(args.mass_weight) or args.mass_weight < 0:
        parser.error("Mass weight must be finite and nonnegative")
    if args.variant == "inventory" and args.mass_weight:
        parser.error("Inventory projection must be tested without a mass penalty")
    output = Path(args.output).resolve() if args.output else Path(__file__).resolve().parents[1]/"results"/("constant_flux_pinn_"+
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = build_model(args.variant)
    initial_weights_sha256 = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    sample = torch.rand(512, 2, dtype=torch.float64)
    rho, tau = sample_interior(sample, args.sampling)
    interior = make_points(rho, tau)
    initial = make_points(torch.linspace(0, 1, 101, dtype=torch.float64), torch.zeros(101, dtype=torch.float64))
    bt = torch.cat((torch.linspace(MIN_TIME, END_TIME, 64, dtype=torch.float64),
                    torch.logspace(np.log10(MIN_TIME), np.log10(END_TIME), 64, dtype=torch.float64)))
    surface = make_points(torch.ones_like(bt), bt)
    mass_times = torch.unique(torch.cat((
        torch.linspace(MIN_TIME, END_TIME, 32, dtype=torch.float64),
        torch.logspace(np.log10(MIN_TIME), np.log10(END_TIME), 32, dtype=torch.float64))))
    history = []
    started = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        for p in (interior, initial, surface):
            p.grad = None
        losses = (projected_backward(model, interior, initial, surface) if args.variant == "inventory"
                  else terms(model, interior, initial, surface))
        loss = losses[0]+100*losses[1]+losses[2]
        if args.mass_weight:
            balance = mass_loss(model, mass_times)
            loss = loss+args.mass_weight*balance
            losses = (*losses, balance)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        if args.variant != "inventory":
            loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite gradient")
        history.append([float(loss.detach()), *[float(v.detach()) for v in losses]])
        return loss

    print("Training constant-flux sphere: uniform initial state, outward flux=1.", flush=True)
    print(f"Variant: {args.variant}", flush=True)
    print(f"Mass-loss weight: {args.mass_weight}", flush=True)
    print(f"Sampling: {args.sampling}; early inner points: {int(((rho<.7)&(tau<=1e-4)).sum())}", flush=True)
    for step in range(args.adam_steps):
        optimizer.step(closure)
        if (step+1) % 500 == 0:
            print(f"Adam {step+1}: loss={history[-1][0]:.6e}", flush=True)
    adam_evaluations = len(history)
    if args.lbfgs_steps:
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=args.lbfgs_steps,
                                     line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                                     tolerance_change=1e-12)
        optimizer.step(closure)
    elapsed = time.perf_counter()-started
    metrics, arrays = assess(model)
    report = {"config": vars(args), "torch_version": torch.__version__, "dtype": "float64",
              "initial_weights_sha256": initial_weights_sha256,
              "early_inner_points": int(((rho<.7)&(tau<=1e-4)).sum()),
              "device": "cpu", "threads": 1, "training_s": elapsed, "metrics": metrics,
              "time_interval": [0, END_TIME], "minimum_positive_time": MIN_TIME,
              "diffusion_number": 1, "architecture": [2 if args.variant == "baseline" else 3, 32, 32, 1],
              "initial_condition": "soft" if args.variant == "baseline" else "hard",
              "interior_points": 512, "initial_points": 101, "surface_points": 128,
              "loss_weights": {"pde": 1, "initial": 100, "surface": 1},
              "adam_evaluations": adam_evaluations, "lbfgs_evaluations": len(history)-adam_evaluations,
              "scope": "Synthetic constant unit flux, one seed. No interior training labels, "
                       "optional integrated conservation loss. Flux is soft; initial enforcement depends on variant. Exact corner "
                       "excluded from flux loss. Grid RMS uses logarithmic time sampling, not "
                       "physical-time weighting. Sampled diagnostics, not acceptance certification."}
    if args.mass_weight:
        report["loss_weights"]["mass"] = args.mass_weight
        report["mass_quadrature"] = {"radial_order": 64, "times": mass_times.tolist(),
                                     "normalization": 3*END_TIME, "target": "1 - 3*t"}
    if args.variant == "inventory":
        report["inventory_projection"] = {"radial_order": 64, "target": "1 - 3*t",
                                          "training_chunk": 64, "prediction_chunk": 128,
                                          "independent_mass_orders": [128, 256]}
    np.savez_compressed(output/"training_points.npz", interior=interior.detach().numpy(),
                        initial=initial.detach().numpy(), surface=surface.detach().numpy())
    report["training_points_sha256"] = hashlib.sha256((output/"training_points.npz").read_bytes()).hexdigest()
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    torch.save(model.state_dict(), output/"model.pt")
    np.savez_compressed(output/"evaluation.npz", **arrays)
    np.savetxt(output/"loss_history.csv", np.asarray(history).reshape(-1, 5 if args.mass_weight else 4), delimiter=",",
               header="total,pde,initial,surface"+(",mass" if args.mass_weight else ""), comments="")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    t = arrays["tau"]
    ax[0].semilogx(t[1:], arrays["reference"][-1, 1:], label="Analytic")
    ax[0].semilogx(t[1:], arrays["prediction"][-1, 1:], "--", label="PINN")
    ax[0].set(xlabel="Dimensionless time", ylabel="Surface concentration", title="Surface response")
    ax[0].legend()
    ax[1].semilogx(t[1:], arrays["volume_mean"][1:], label="PINN")
    ax[1].semilogx(t[1:], 1-3*t[1:], "--", label="1 - 3 t")
    ax[1].set(xlabel="Dimensionless time", ylabel="Volume mean", title="Lithium balance")
    ax[1].legend()
    ax[2].semilogx(t[1:], np.max(np.abs(arrays["prediction"][:, 1:]-arrays["reference"][:, 1:]), axis=0))
    ax[2].set(xlabel="Dimensionless time", ylabel="Max radial error", title="Sampled error")
    fig.savefig(output/"benchmark.png", dpi=160)
    plt.close(fig)
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
