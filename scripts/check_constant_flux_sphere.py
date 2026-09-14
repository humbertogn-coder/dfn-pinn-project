"""Validate a switched-on outward flux against an independent sphere series."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybamm
from scipy.optimize import brentq


def eigenvalues(count):
    return np.array([brentq(lambda a: a * np.cos(a) - np.sin(a),
                           k * np.pi, (k + 0.5) * np.pi) for k in range(1, count + 1)])


def series_solution(radius, time, roots):
    """Unit diffusivity/radius/outward flux; initial concentration is one."""
    radius, time = np.asarray(radius), np.asarray(time)
    modes = np.sinc(radius[:, None] * roots[None, :] / np.pi)
    coefficients = 2 / (roots**2 * np.cos(roots))
    result = (1 - 3 * time[None, :] + 0.3 - radius[:, None]**2 / 2
              + (modes * coefficients) @ np.exp(-roots[:, None]**2 * time[None, :]))
    # At exactly zero, impose the initial condition instead of a truncated series.
    result[:, time == 0] = 1
    return result


def solve_sphere(points, time):
    model = pybamm.BaseModel()
    r = pybamm.SpatialVariable("r", domain=["sphere"], coord_sys="spherical polar")
    c = pybamm.Variable("Concentration", domain=["sphere"])
    model.rhs = {c: pybamm.div(pybamm.grad(c))}
    model.initial_conditions = {c: pybamm.Scalar(1)}
    model.boundary_conditions = {c: {"left": (pybamm.Scalar(0), "Neumann"),
                                      "right": (pybamm.Scalar(-1), "Neumann")}}
    model.variables = {"Concentration": c, "Surface": pybamm.boundary_value(c, "right")}
    mesh = pybamm.Mesh({"sphere": {r: {"min": 0, "max": 1}}},
                      {"sphere": pybamm.Uniform1DSubMesh}, {r: points})
    disc = pybamm.Discretisation(mesh, {"sphere": pybamm.FiniteVolume()})
    disc.process_model(model)
    solution = pybamm.IDAKLUSolver(rtol=1e-11, atol=1e-13).solve(model, time, t_interp=time)
    values = np.asarray(solution["Concentration"].entries)
    surface = np.asarray(solution["Surface"].entries).reshape(-1)
    if (values.shape != (points, len(time)) or surface.shape != time.shape
            or not np.all(np.isfinite(values)) or not np.all(np.isfinite(surface))
            or not np.allclose(solution.t, time, rtol=0, atol=1e-12)):
        raise ValueError("Invalid numerical output")
    return np.asarray(mesh["sphere"].nodes), np.asarray(mesh["sphere"].edges), values, surface


def main():
    time = np.r_[0., np.geomspace(1e-5, 0.02, 180)]
    roots = eigenvalues(1024)
    reference_surface = series_solution([1.], time, roots)[0]
    rows, arrays, series_checks = [], {}, []
    for points in (80, 160, 320):
        print(f"Running constant-flux sphere: r{points}...", flush=True)
        nodes, edges, values, surface = solve_sphere(points, time)
        exact = series_solution(nodes, time, roots)
        check_r = np.r_[nodes, 1.]
        discrepancy = float(np.max(np.abs(series_solution(check_r, time, roots)
                                            - series_solution(check_r, time, roots[:512]))))
        series_checks.append(discrepancy)
        if discrepancy > 1e-9:
            raise ValueError("Series truncation check failed; increase the mode count")
        # Spherical shell weights integrate a piecewise-constant finite-volume field.
        mean = (edges[1:]**3 - edges[:-1]**3) @ values
        mass_error = float(np.max(np.abs(mean - (1 - 3 * time))))
        for kind, difference in (("native_particle", values - exact),
                                 ("surface", surface - reference_surface)):
            positive = np.abs(difference[..., 1:])
            index = np.unravel_index(np.argmax(positive), positive.shape)
            rows.append({"radial_points": points, "field": kind,
                         "max_abs_error_positive_time": float(positive.max()),
                         "rms_error_positive_time": float(np.sqrt(np.mean(positive**2))),
                         "peak_time": float(time[index[-1] + 1]),
                         "initial_max_abs_error": float(np.max(np.abs(difference[..., 0]))),
                         "volume_mean_mass_balance_error": mass_error})
        arrays[f"nodes_r{points}"] = nodes
        arrays[f"numerical_r{points}"] = values
        arrays[f"exact_r{points}"] = exact
        arrays[f"surface_r{points}"] = surface
    output = Path(__file__).resolve().parents[1] / "results" / (
        "constant_flux_sphere_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    report = {"pybamm_version": pybamm.__version__, "solver": "IDAKLUSolver",
              "rtol": 1e-11, "atol": 1e-13, "series_modes": 1024,
              "max_512_vs_1024_difference": max(series_checks), "time": time.tolist(),
              "equation": "dc/dtau = rho^-2*d/drho(rho^2*dc/drho)",
              "conditions": "c(rho,0)=1; dc/drho(0,tau)=0; dc/drho(1,tau)=-1 for tau>0",
              "series": "c=1-3*tau+3/10-rho^2/2 + sum[2*sinc(alpha*rho/pi)*exp(-alpha^2*tau)/(alpha^2*cos(alpha))]; tan(alpha)=alpha",
              "mass_balance": "3*integral(c*rho^2 drho)=1-3*tau",
              "metrics": rows,
              "scope": "Dimensionless constant outward flux, not a complete DFN cell. "
                       "Uniform initial data and switched flux are incompatible at the corner "
                       "rho=1,tau=0. Initial and positive-time errors are reported separately. "
                       "Surface uses PyBaMM boundary reconstruction, no cross-mesh interpolation. "
                       "Native point errors include FV representation error. RMS is unweighted "
                       "on logarithmic time samples. Peaks before 1e-5 or between samples may be missed. "
                       "Series truncation is checked at all comparison nodes and the surface."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "samples.npz", time=time, exact_surface=reference_surface, **arrays)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].semilogx(time[1:], reference_surface[1:], color="black", label="Analytic series")
    for points in (80, 160, 320):
        surface = arrays[f"surface_r{points}"]
        axes[0].semilogx(time[1:], surface[1:], label=f"r{points}")
        axes[1].loglog(time[1:], np.abs(surface[1:] - reference_surface[1:]), label=f"r{points}")
    axes[0].set(ylabel="Surface concentration", title="Constant outward flux")
    axes[1].set(ylabel="Absolute surface error", title="Error against analytic series")
    for ax in axes:
        ax.set_xlabel("Dimensionless time")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(output / "constant_flux_sphere.png", dpi=180)
    plt.close(fig)
    print("Mesh Field             Max error(t>0)  Peak time     Initial error")
    for row in rows:
        print(f"{row['radial_points']:4d} {row['field']:17s} "
              f"{row['max_abs_error_positive_time']:.6e}  {row['peak_time']:.6e}  "
              f"{row['initial_max_abs_error']:.6e}")
    print(f"Series check (512 vs 1024): {max(series_checks):.3e}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
