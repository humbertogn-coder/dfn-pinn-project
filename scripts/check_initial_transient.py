"""Separate radial and through-cell refinement in the first 120 seconds."""

from datetime import datetime, timezone
import csv
import json
from pathlib import Path

from check_concentration_convergence import sample_spatial
from export_pybamm_reference import run_discharge
import matplotlib.pyplot as plt
import numpy as np
import pybamm


def main():
    cases = [(40, 40), (80, 80), (80, 40), (80, 160), (40, 80), (160, 80)]
    runs, settings = {}, {}
    common_t = np.arange(0., 120.01, 0.5)
    targets = {}
    for x_points, r_points in cases:
        key = f"x{x_points}_r{r_points}"
        print(f"Running initial transient: {key}...", flush=True)
        solution, parameters, protocol, mesh, elapsed = run_discharge(
            x_points, "0.5 seconds", radial_points=r_points, duration_s=120,
            rtol=1e-8, atol=1e-10
        )
        t = np.asarray(solution["Time [s]"].entries).reshape(-1)
        if t.shape != common_t.shape or not np.allclose(t, common_t, rtol=0, atol=1e-8):
            raise ValueError("Expected the complete shared half-second output grid")
        if not targets:
            radius = float(parameters["Positive particle radius [m]"])
            length = float(parameters["Positive electrode thickness [m]"])
            start = float(parameters["Negative electrode thickness [m]"] + parameters["Separator thickness [m]"])
            # Original 20-cell centers and a second grid adding their midpoints.
            for label, count in (("coarse_centers", 20), ("augmented", 39)):
                fraction = np.linspace(0.025, 0.975, count)
                targets[label] = [fraction * radius, start + fraction * length]
        particle = solution["Positive particle concentration [mol.m-3]"]
        surface = solution["Positive particle surface concentration [mol.m-3]"]
        coordinates = [np.asarray(particle.mesh.nodes),
                       np.asarray(particle.base_variables[0].secondary_mesh.nodes)]
        runs[key] = {}
        for grid, target in targets.items():
            runs[key][grid] = {
                "particle": sample_spatial(np.asarray(particle.entries), coordinates, target),
                "surface": sample_spatial(np.asarray(surface.entries), [np.asarray(surface.mesh.nodes)], [target[1]]),
            }
        settings[key] = {"mesh": mesh, "solve_wall_time_s": elapsed, "protocol": protocol,
                         "termination": str(solution.termination)}
        del particle, surface, solution
    pairs = [
        ("joint", "x40_r40", "x80_r80"),
        ("radial_40_80", "x80_r40", "x80_r80"),
        ("radial_80_160", "x80_r80", "x80_r160"),
        ("spatial_40_80", "x40_r80", "x80_r80"),
        ("spatial_80_160", "x80_r80", "x160_r80"),
    ]
    rows, curves = [], {}
    for grid, target in targets.items():
        for label, coarse, fine in pairs:
            for field in ("particle", "surface"):
                delta = runs[coarse][grid][field] - runs[fine][grid][field]
                absolute = np.abs(delta)
                index = np.unravel_index(np.argmax(absolute), absolute.shape)
                row = {"grid": grid, "comparison": label, "field": field,
                       "coarse_case": coarse, "fine_case": fine,
                       "max_abs_difference_mol_m3": float(absolute.max()),
                       "rms_difference_mol_m3": float(np.sqrt(np.mean(delta**2))),
                       "worst_time_s": float(common_t[index[-1]]),
                       "worst_x_m": float(target[1][index[-2]]),
                       "worst_r_m": float(target[0][index[0]]) if field == "particle" else radius}
                rows.append(row)
                curves[f"{grid}_{label}_{field}"] = np.max(absolute, axis=tuple(range(absolute.ndim - 1)))
    output = Path(__file__).resolve().parents[1] / "results" / (
        "initial_transient_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    report = {"pybamm_version": pybamm.__version__, "parameter_set": "Chen2020",
              "model": "isothermal DFN", "initial_soc": 1.0, "solver": "IDAKLUSolver",
              "rtol": 1e-8, "atol": 1e-10, "cases": settings, "time_s": common_t.tolist(),
              "targets_m": {key: [a.tolist() for a in axes] for key, axes in targets.items()},
              "metrics": rows,
              "scope": "Positive concentrations only, first 120 seconds. Radial refinement changes "
                       "both particle meshes; spatial refinement changes all three through-cell domains. "
                       "Linear spatial interpolation with bounds checking; no time interpolation. "
                       "Two interior query grids test sampling sensitivity, not exact interpolation error. "
                       "RMS is unweighted. Native radial endpoints are excluded; surface is a separate "
                       "PyBaMM variable. No exact solution or convergence certification is implied."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "samples.npz", time_s=common_t,
                        **{f"{case}_{grid}_{field}": values
                           for case, grids in runs.items() for grid, fields in grids.items()
                           for field, values in fields.items()})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for column, grid in enumerate(targets):
        for row, field in enumerate(("particle", "surface")):
            ax = axes[row, column]
            for label, _, _ in pairs:
                ax.plot(common_t, curves[f"{grid}_{label}_{field}"], label=label)
            ax.set(title=f"{field} | {grid}", xlabel="Time [s]", ylabel="Max difference [mol/m3]")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Positive concentration | Initial transient refinement")
    fig.savefig(output / "initial_transient.png", dpi=180)
    plt.close(fig)
    print("Grid             Comparison        Field       Max [mol/m3]   Time [s]   x [um]   r [um]")
    for row in rows:
        print(f"{row['grid']:17s} {row['comparison']:17s} {row['field']:10s} "
              f"{row['max_abs_difference_mol_m3']:13.4f} {row['worst_time_s']:10.2f} "
              f"{row['worst_x_m']*1e6:8.3f} {row['worst_r_m']*1e6:8.3f}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
