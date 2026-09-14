"""Compare radial refinement in both electrodes over the first five seconds."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from check_concentration_convergence import sample_spatial
from export_pybamm_reference import run_discharge
import matplotlib.pyplot as plt
import numpy as np
import pybamm


def main():
    resolutions = (80, 160, 320)
    time = np.linspace(0, 5, 101)
    runs, targets, settings, radii = {}, {}, {}, {}
    for radial in resolutions:
        print(f"Running fast transient: x80_r{radial}...", flush=True)
        solution, parameters, protocol, mesh, elapsed = run_discharge(
            80, "0.05 seconds", radial_points=radial, duration_s=5,
            rtol=1e-8, atol=1e-10,
        )
        actual_time = np.asarray(solution["Time [s]"].entries).reshape(-1)
        if actual_time.shape != time.shape or not np.allclose(actual_time, time, rtol=0, atol=1e-9):
            raise ValueError("Expected shared output times from 0 to 5 seconds")
        runs[radial] = {}
        for domain, suffix in (("Negative", "n"), ("Positive", "p")):
            particle = solution[f"{domain} particle concentration [mol.m-3]"]
            surface = solution[f"{domain} particle surface concentration [mol.m-3]"]
            coordinates = [np.asarray(particle.mesh.nodes),
                           np.asarray(particle.base_variables[0].secondary_mesh.nodes)]
            if suffix not in targets:
                targets[suffix] = coordinates
                radii[suffix] = float(parameters[f"{domain} particle radius [m]"])
            if not np.array_equal(coordinates[1], targets[suffix][1]):
                raise ValueError("Through-cell coordinates changed")
            runs[radial][f"particle_{suffix}"] = sample_spatial(
                np.asarray(particle.entries), coordinates, targets[suffix]
            )
            surface_values = np.asarray(surface.entries)
            if (not np.array_equal(surface.mesh.nodes, targets[suffix][1])
                    or surface_values.shape != (len(targets[suffix][1]), len(time))
                    or not np.all(np.isfinite(surface_values))):
                raise ValueError("Invalid surface output")
            runs[radial][f"surface_{suffix}"] = surface_values.copy()
        settings[str(radial)] = {"mesh": mesh, "protocol": protocol,
                                 "solve_wall_time_s": elapsed, "termination": str(solution.termination)}
        del particle, surface, solution
    rows, curves = [], {}
    for suffix in ("n", "p"):
        for field in ("particle", "surface"):
            name = f"{field}_{suffix}"
            for coarse, fine in ((80, 160), (160, 320), (80, 320)):
                delta = runs[coarse][name] - runs[fine][name]
                absolute = np.abs(delta)
                index = np.unravel_index(np.argmax(absolute), absolute.shape)
                rows.append({"field": name, "coarse_radial_points": coarse,
                             "fine_radial_points": fine,
                             "max_abs_difference_mol_m3": float(absolute.max()),
                             "rms_difference_mol_m3": float(np.sqrt(np.mean(delta**2))),
                             "worst_time_s": float(time[index[-1]]),
                             "worst_x_m": float(targets[suffix][1][index[-2]]),
                             "worst_r_m": float(targets[suffix][0][index[0]]) if field == "particle" else radii[suffix],
                             "peak_at_first_positive_sample": bool(index[-1] == 1)})
                curves[f"{name}_{coarse}_{fine}"] = absolute.max(axis=tuple(range(absolute.ndim - 1)))
    now = datetime.now(timezone.utc)
    output = Path(__file__).resolve().parents[1] / "results" / (
        "fast_radial_transient_" + now.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    report = {"created_utc": now.isoformat(), "pybamm_version": pybamm.__version__,
              "model": "isothermal DFN", "parameter_set": "Chen2020", "initial_soc": 1.0,
              "solver": "IDAKLUSolver", "rtol": 1e-8, "atol": 1e-10,
              "time_s": time.tolist(), "cases": settings,
              "targets_m": {key: [c.tolist() for c in coords] for key, coords in targets.items()},
              "metrics": rows,
              "scope": "Both radial meshes change together; through-cell meshes stay at 80. "
                       "Linear radial interpolation to all native r80 centers; x coordinates "
                       "are identical. No time interpolation or spatial extrapolation. "
                       "Surfaces are separate native outputs. RMS is unweighted. Interpolation "
                       "contributes to differences. Peaks before 0.05 s or between samples "
                       "may be missed. The 320-point case is not exact; no full-discharge "
                       "accuracy certification or pass threshold is implied."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "samples.npz", time_s=time,
                        **{f"r{radial}_{name}": values for radial, fields in runs.items()
                           for name, values in fields.items()})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for column, suffix in enumerate(("n", "p")):
        for row, field in enumerate(("particle", "surface")):
            ax = axes[row, column]
            for coarse, fine in ((80, 160), (160, 320)):
                ax.plot(time, curves[f"{field}_{suffix}_{coarse}_{fine}"], label=f"r{coarse} vs r{fine}")
            ax.set(title=f"{field}_{suffix}", xlabel="Time [s]", ylabel="Max difference [mol/m3]")
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle("Fast radial transient | Fixed x80 mesh | 0.05 s output samples")
    fig.savefig(output / "fast_radial_transient.png", dpi=180)
    plt.close(fig)
    print("Field        Pair       Max [mol/m3]   RMS [mol/m3]   Peak time [s]")
    for row in rows:
        print(f"{row['field']:12s} {row['coarse_radial_points']:3d}-{row['fine_radial_points']:3d} "
              f"{row['max_abs_difference_mol_m3']:15.6f} {row['rms_difference_mol_m3']:14.6f} "
              f"{row['worst_time_s']:15.2f}")
    print("Sampled maxima only; 320 radial points are not an exact reference.")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
