"""Small physics-only PINN benchmark: closed sphere, analytic initial profile."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
import torch
from torch import nn

from dfn_pinn.projection import gauss_legendre
from dfn_pinn.spherical_diffusion import ParticleScales, diffusion_residual, center_residual


ALPHA = brentq(lambda a: np.tan(a)-a, np.pi+1e-6, 1.5*np.pi-1e-6)
END_TIME = .1
# Synthetic dimensionless benchmark: D*t_ref/R^2=1, not Chen2020 physical units.
SCALES = ParticleScales(1., 1., 1., 1.)


def exact(rho, tau):
    return .5+.1*torch.sinc(ALPHA*rho/torch.pi)*torch.exp(-ALPHA**2*tau)


def make_points(rho, tau):
    return torch.stack((rho, torch.zeros_like(rho), tau), dim=-1).requires_grad_()


class ParticleNet(nn.Module):
    """Pointwise tanh MLP; rho squared enforces center symmetry only."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 24),
                                    nn.Tanh(), nn.Linear(24, 1))
        self.double()

    def forward(self, p):
        inputs = torch.stack((2*p[:, 0]**2-1, 2*p[:, 2]/END_TIME-1), dim=-1)
        return .5+.1*self.layers(inputs)


def loss_terms(model, interior, initial, boundary):
    # Divide all residuals by the known initial perturbation amplitude.
    pde = (diffusion_residual(model, interior, SCALES)/.1).square().mean()
    ic = ((model(initial)[:, 0]-exact(initial[:, 0], initial[:, 2]))/.1).square().mean()
    prediction = model(boundary)
    derivative = torch.autograd.grad(prediction.sum(), boundary, create_graph=True)[0][:, 0]
    bc = (derivative/.1).square().mean()
    return pde, ic, bc


def evaluate(model):
    rho = torch.linspace(0, 1, 101, dtype=torch.float64)
    tau = torch.linspace(0, END_TIME, 81, dtype=torch.float64)
    tt, rr = torch.meshgrid(tau, rho, indexing="ij")
    p = make_points(rr.flatten(), tt.flatten())
    with torch.no_grad():
        prediction = model(p).reshape(tt.shape)
        reference = exact(rr, tt)
    error = prediction-reference
    # Independent radial quadrature; not a term in the training objective.
    nodes, weights = gauss_legendre(64, 0., 1.)
    times, radii = torch.meshgrid(tau, nodes, indexing="ij")
    with torch.no_grad():
        c = model(make_points(radii.flatten(), times.flatten())).reshape(times.shape)
        means = 3*(c*nodes**2*weights).sum(-1)
    boundary = make_points(torch.ones_like(tau[1:]), tau[1:])
    gradient = torch.autograd.grad(model(boundary).sum(), boundary)[0][:, 0]
    center = center_residual(model, make_points(torch.zeros_like(tau), tau))
    validation = make_points(rr.flatten(), tt.flatten())
    pde = diffusion_residual(model, validation, SCALES).detach()
    metrics = {"max_abs_c_hat_error": error.abs().max().item(),
               "rms_c_hat_error": error.square().mean().sqrt().item(),
               "initial_max_error": error[0].abs().max().item(),
               "surface_max_abs_d_c_hat_d_rho": gradient.abs().max().item(),
               "center_max_abs_d_c_hat_d_rho": center.abs().max().item(),
               "max_mean_error_from_exact_0_5": (means-.5).abs().max().item(),
               "max_mean_drift_from_predicted_initial": (means-means[0]).abs().max().item(),
               "validation_pde_rms": pde.square().mean().sqrt().item(),
               "minimum_c_hat": prediction.min().item(), "maximum_c_hat": prediction.max().item()}
    return metrics, rho.numpy(), tau.numpy(), prediction.numpy(), reference.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adam-steps", type=int, default=1000)
    parser.add_argument("--lbfgs-steps", type=int, default=150)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.adam_steps < 0 or args.lbfgs_steps < 0:
        parser.error("Optimizer iteration counts must be nonnegative")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = ParticleNet()
    sample = torch.rand(256, 2, dtype=torch.float64)
    interior = make_points(.001+.999*sample[:, 0], END_TIME*sample[:, 1])
    initial = make_points(torch.linspace(0, 1, 65, dtype=torch.float64), torch.zeros(65, dtype=torch.float64))
    bt = torch.linspace(END_TIME/64, END_TIME, 64, dtype=torch.float64)
    boundary = make_points(torch.ones_like(bt), bt)
    history = []
    start = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def objective():
        optimizer.zero_grad(set_to_none=True)
        for p in (interior, initial, boundary):
            p.grad = None
        terms = loss_terms(model, interior, initial, boundary)
        loss = terms[0]+10*terms[1]+terms[2]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite parameter gradient")
        history.append([float(loss.detach()), *[float(v.detach()) for v in terms]])
        return loss

    print("Training synthetic no-flux sphere PINN on CPU in float64.", flush=True)
    for step in range(args.adam_steps):
        optimizer.step(objective)
        if (step+1) % 250 == 0:
            print(f"Adam {step+1}: loss={history[-1][0]:.6e}", flush=True)
    adam_evaluations = len(history)
    if args.lbfgs_steps:
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=args.lbfgs_steps,
                                      line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                                      tolerance_change=1e-12)
        optimizer.step(objective)
    metrics, rho, tau, prediction, reference = evaluate(model)
    output = args.output or Path(__file__).resolve().parents[1]/"results"/(
        "single_particle_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), output/"model.pt")
    np.savez_compressed(output/"evaluation.npz", rho=rho, tau=tau, prediction=prediction, reference=reference)
    np.savetxt(output/"loss_history.csv", np.asarray(history).reshape(-1, 4), delimiter=",",
               header="total,pde,initial,surface", comments="")
    report = {"config": vars(args) | {"output": str(output)}, "metrics": metrics,
              "torch_version": torch.__version__, "python_version": platform.python_version(),
              "alpha": ALPHA, "end_time": END_TIME, "diffusion_number": 1.,
              "dtype": "float64", "device": "cpu", "threads": 1,
              "collocation": 256, "initial_points": 65, "surface_points": 64,
              "loss_weights": {"pde": 1, "initial": 10, "surface": 1},
              "adam_closure_evaluations": adam_evaluations,
              "lbfgs_closure_evaluations": len(history)-adam_evaluations,
              "elapsed_s": time.perf_counter()-start,
              "scope": "One synthetic no-flux sphere and one seed. No interior labels, no DFN, "
                       "no acceptance certificate. Analytic initial data used for training; "
                       "later analytic solution used only for evaluation. CPU determinism "
                       "does not guarantee bitwise equality across platforms or versions."}
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    for index in (0, 20, 80):
        axes[0].plot(rho, reference[index], label=f"Exact t={tau[index]:.3f}")
        axes[0].plot(rho, prediction[index], "--", label=f"PINN t={tau[index]:.3f}")
    axes[0].set(xlabel="r/R", ylabel="c/c_ref", title="Concentration profiles")
    axes[0].legend(fontsize=7)
    heat = axes[1].imshow(np.abs(prediction-reference), origin="lower", aspect="auto",
                          extent=(0, 1, 0, END_TIME))
    axes[1].set(xlabel="r/R", ylabel="t/t_ref", title="Absolute concentration error")
    fig.colorbar(heat, ax=axes[1])
    if history:
        axes[2].semilogy(np.maximum(np.asarray(history)[:, 0], 1e-30))
    axes[2].set(xlabel="Objective evaluation", ylabel="Weighted loss", title="Adam then L-BFGS")
    fig.savefig(output/"benchmark.png", dpi=160)
    plt.close(fig)
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
