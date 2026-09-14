"""Separate interpolation and numerical errors using an exact spherical mode."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from check_concentration_convergence import sample_spatial
import matplotlib.pyplot as plt
import numpy as np
import pybamm
from scipy.optimize import brentq


ALPHA = brentq(lambda a: a * np.cos(a) - np.sin(a), np.pi + 0.1, 1.5 * np.pi)


def exact_solution(radius, time):
    """c=1+0.1*sin(alpha*r)/(alpha*r)*exp(-alpha^2*t), regular at r=0."""
    return 1 + 0.1 * np.sinc(ALPHA * np.asarray(radius)[:, None] / np.pi) * np.exp(
        -ALPHA**2 * np.asarray(time)[None, :]
    )


def solve_sphere(points, time):
    model = pybamm.BaseModel()
    r = pybamm.SpatialVariable("r", domain=["sphere"], coord_sys="spherical polar")
    c = pybamm.Variable("Concentration", domain=["sphere"])
    model.rhs = {c: pybamm.div(pybamm.grad(c))}
    model.initial_conditions = {c: 1 + 0.1 * pybamm.sin(ALPHA * r) / (ALPHA * r)}
    model.boundary_conditions = {c: {"left": (pybamm.Scalar(0), "Neumann"),
                                      "right": (pybamm.Scalar(0), "Neumann")}}
    model.variables = {"Concentration": c}
    geometry = {"sphere": {r: {"min": 0, "max": 1}}}
    mesh = pybamm.Mesh(geometry, {"sphere": pybamm.Uniform1DSubMesh}, {r: points})
    disc = pybamm.Discretisation(mesh, {"sphere": pybamm.FiniteVolume()})
    disc.process_model(model)
    solution = pybamm.IDAKLUSolver(rtol=1e-11, atol=1e-13).solve(model, time, t_interp=time)
    values = np.asarray(solution["Concentration"].entries)
    nodes = np.asarray(mesh["sphere"].nodes)
    if values.shape != (points, len(time)) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid numerical solution")
    if not np.allclose(solution.t, time, rtol=0, atol=1e-12):
        raise ValueError("Unexpected output times")
    return nodes, values


def main():
    time = np.linspace(0, 0.2, 201)
    target = (np.arange(80) + 0.5) / 80
    truth = exact_solution(target, time)
    rows, arrays = [], {}
    for points in (80, 160, 320):
        print(f"Solving analytic sphere benchmark: r{points}...", flush=True)
        nodes, numerical = solve_sphere(points, time)
        native_truth = exact_solution(nodes, time)
        interpolated_truth = sample_spatial(native_truth, [nodes], [target])
        interpolated_numerical = sample_spatial(numerical, [nodes], [target])
        errors = {"native_solver": numerical - native_truth,
                  "interpolation_only": interpolated_truth - truth,
                  "combined": interpolated_numerical - truth}
        for kind, error in errors.items():
            rows.append({"radial_points": points, "error_type": kind,
                         "max_abs_error": float(np.max(np.abs(error))),
                         "rms_error": float(np.sqrt(np.mean(error**2)))})
        arrays[f"exact_interpolated_r{points}"] = interpolated_truth
        arrays[f"numerical_interpolated_r{points}"] = interpolated_numerical
        arrays[f"nodes_r{points}"] = nodes
        arrays[f"numerical_native_r{points}"] = numerical
    pairwise = []
    for kind in ("exact", "numerical"):
        for coarse, fine in ((80, 160), (160, 320)):
            difference = arrays[f"{kind}_interpolated_r{coarse}"] - arrays[f"{kind}_interpolated_r{fine}"]
            pairwise.append({"source": kind, "pair": f"{coarse}-{fine}",
                             "max_abs_difference": float(np.max(np.abs(difference)))})
    now = datetime.now(timezone.utc)
    output = Path(__file__).resolve().parents[1] / "results" / (
        "spherical_diffusion_" + now.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    report = {"created_utc": now.isoformat(), "pybamm_version": pybamm.__version__,
              "equation": "dc/dtau = (1/rho^2)*d/drho(rho^2*dc/drho)",
              "exact_solution": "1+0.1*sinc(alpha*rho/pi)*exp(-alpha^2*tau)",
              "alpha": ALPHA, "root_residual": float(ALPHA*np.cos(ALPHA)-np.sin(ALPHA)),
              "boundary_conditions": "Zero radial derivative at rho=0 and rho=1",
              "units": "Dimensionless concentration, rho=r/R, tau=D*t/R^2",
              "solver": "IDAKLUSolver", "rtol": 1e-11, "atol": 1e-13,
              "metrics": rows, "pairwise": pairwise,
              "scope": "Smooth analytic eigenmode, not the DFN step-current boundary transient. "
                       "Native solver error compares FV nodal outputs with analytic point values. "
                       "Interpolation-only error samples the exact solution on each mesh. "
                       "All shared-grid comparisons use r80 centers without extrapolation. "
                       "An r80 interpolation error of zero is by construction. RMS is unweighted. "
                       "This benchmark cannot certify DFN accuracy or attribute all DFN discrepancies."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "samples.npz", time=time, target_radius=target, exact_target=truth, **arrays)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for points in (80, 160, 320):
        for ax, key, title in zip(axes, ("exact", "numerical"),
                                  ("Interpolation only", "Numerical solution plus interpolation")):
            error = np.max(np.abs(arrays[f"{key}_interpolated_r{points}"] - truth), axis=0)
            ax.plot(time, error, label=f"r{points}")
            ax.set(title=title, xlabel="Dimensionless time", ylabel="Max absolute concentration error")
            ax.grid(alpha=0.2)
            ax.legend()
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle("Spherical diffusion | Exact eigenmode reference")
    fig.savefig(output / "spherical_diffusion.png", dpi=180)
    plt.close(fig)
    print("Mesh  Error type           Max error       RMS error")
    for row in rows:
        print(f"{row['radial_points']:4d}  {row['error_type']:20s} "
              f"{row['max_abs_error']:.6e}  {row['rms_error']:.6e}")
    for row in pairwise:
        print(f"Pairwise {row['source']} {row['pair']}: {row['max_abs_difference']:.6e}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
