"""Compare internal concentrations on shared interior points and output times."""

from datetime import datetime, timezone
import csv
import json
from pathlib import Path

from export_pybamm_reference import run_discharge
import matplotlib.pyplot as plt
import numpy as np
import pybamm
from scipy.interpolate import RegularGridInterpolator


def sample_spatial(values, coordinates, targets):
    """Interpolate spatial axes only; trailing time samples remain unchanged."""
    if values.shape[:-1] != tuple(len(c) for c in coordinates):
        raise ValueError("Spatial coordinate shape mismatch")
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite concentrations")
    for c in coordinates:
        if not np.all(np.isfinite(c)) or not np.all(np.diff(c) > 0):
            raise ValueError("Invalid spatial coordinates")
    query = np.stack(np.meshgrid(*targets, indexing="ij"), axis=-1)
    return RegularGridInterpolator(coordinates, values, method="linear", bounds_error=True)(query)


def main():
    resolutions = (20, 40, 80)
    specs = [
        ("c_s_n", "Negative particle concentration [mol.m-3]", "negative"),
        ("c_s_p", "Positive particle concentration [mol.m-3]", "positive"),
        ("c_s_surf_n", "Negative particle surface concentration [mol.m-3]", "negative"),
        ("c_s_surf_p", "Positive particle surface concentration [mol.m-3]", "positive"),
        ("c_e_n", "Negative electrolyte concentration [mol.m-3]", "electrolyte"),
        ("c_e_s", "Separator electrolyte concentration [mol.m-3]", "electrolyte"),
        ("c_e_p", "Positive electrolyte concentration [mol.m-3]", "electrolyte"),
    ]
    runs, targets, scales = [], {}, {}
    for points in resolutions:
        print(f"Running concentration study: {points} points/domain...", flush=True)
        solution, parameters, protocol, mesh, elapsed = run_discharge(
            points, "10 seconds", rtol=1e-8, atol=1e-10
        )
        t = np.asarray(solution["Time [s]"].entries).reshape(-1)
        if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
            raise ValueError("Invalid time samples")
        if abs(float(solution["Voltage [V]"].entries[-1]) - 2.5) > 0.01:
            raise ValueError("Voltage cutoff was not reached")
        fields = {}
        for identifier, name, material in specs:
            variable = solution[name]
            values = np.asarray(variable.entries)
            # These concentration variables live at native nodes, in r,x,t or x,t order.
            coordinates = [np.asarray(variable.mesh.nodes)]
            if variable.dimensions == 2:
                coordinates.append(np.asarray(variable.base_variables[0].secondary_mesh.nodes))
            elif variable.dimensions != 1:
                raise ValueError(f"Unexpected dimensions: {name}")
            if values.shape[-1] != len(t):
                raise ValueError("Time axis mismatch")
            if identifier not in targets:
                targets[identifier] = coordinates
                key = ("Initial concentration in electrolyte [mol.m-3]" if material == "electrolyte"
                       else f"Maximum concentration in {material} electrode [mol.m-3]")
                scales[identifier] = float(parameters[key])
            fields[identifier] = sample_spatial(values, coordinates, targets[identifier])
        runs.append({"time": t, "fields": fields, "mesh": mesh, "solve_wall_time_s": elapsed})
        del solution
    end = min(run["time"][-1] for run in runs)
    common_t = runs[0]["time"][runs[0]["time"] <= end]
    for run in runs:
        indices = np.searchsorted(run["time"], common_t)
        if (np.any(indices >= len(run["time"]))
                or not np.allclose(run["time"][indices], common_t, rtol=0, atol=1e-6)):
            raise ValueError("Output times do not match; no temporal interpolation is allowed")
        run["fields"] = {key: value[..., indices] for key, value in run["fields"].items()}
    rows, curves = [], {}
    for identifier, _, _ in specs:
        for lower, upper in ((0, 1), (0, 2), (1, 2)):
            difference = runs[lower]["fields"][identifier] - runs[upper]["fields"][identifier]
            absolute = np.abs(difference)
            maximum = float(absolute.max())
            worst = np.unravel_index(np.argmax(absolute), absolute.shape)
            row = {"field": identifier, "coarse_points": resolutions[lower],
                   "fine_points": resolutions[upper], "max_abs_difference_mol_m3": maximum,
                   "rms_difference_mol_m3": float(np.sqrt(np.mean(difference**2))),
                   "scale_mol_m3": scales[identifier],
                   "max_difference_percent_of_scale": 100 * maximum / scales[identifier],
                   "worst_sample_time_s": float(common_t[worst[-1]])}
            rows.append(row)
            curves[f"{identifier}_{resolutions[lower]}_{resolutions[upper]}"] = (
                100 * np.max(absolute, axis=tuple(range(absolute.ndim - 1))) / scales[identifier]
            )
    now = datetime.now(timezone.utc)
    output = (Path(__file__).resolve().parents[1] / "results"
              / ("concentration_convergence_" + now.strftime("%Y%m%dT%H%M%S%fZ")))
    output.mkdir(exist_ok=False)
    report = {
        "created_utc": now.isoformat(), "pybamm_version": pybamm.__version__,
        "model": "isothermal DFN", "parameter_set": "Chen2020", "protocol": protocol,
        "initial_soc": 1.0, "solver": "IDAKLUSolver", "rtol": 1e-8, "atol": 1e-10,
        "output_period_s": 10, "time_s": common_t.tolist(),
        "meshes": [run["mesh"] for run in runs],
        "cutoff_times_s": [float(run["time"][-1]) for run in runs],
        "solve_wall_times_s": [run["solve_wall_time_s"] for run in runs],
        "spatial_targets_m": {key: [c.tolist() for c in coords] for key, coords in targets.items()},
        "source_variables": {key: name for key, name, _ in specs}, "metrics": rows,
        "scope": "Joint spatial refinement at fixed tolerances. Linear spatial interpolation "
                 "to the 20-point mesh centers; no extrapolation or temporal interpolation. "
                 "Particle target axes are r,x. Electrolyte domains are compared separately. "
                 "RMS is unweighted over sampled points and times, not a volume integral. "
                 "Surface concentrations are separate PyBaMM boundary outputs. Exact radial "
                 "and electrode endpoints are not sampled. Scaling uses solid maximum "
                 "concentrations or initial electrolyte concentration, not local relative error. "
                 "Interpolation contributes to differences; extrema between samples may be missed. "
                 "The 80-point run is not exact. No convergence certificate or threshold is imposed.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "comparison_samples.npz", time_s=common_t,
                        **{f"{key}_mesh_{points}": run["fields"][key]
                           for points, run in zip(resolutions, runs) for key, _, _ in specs})
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), layout="constrained")
    for ax, (identifier, _, _) in zip(axes.flat, specs):
        for pair in ((20, 40), (40, 80)):
            ax.plot(common_t / 60, curves[f"{identifier}_{pair[0]}_{pair[1]}"],
                    label=f"{pair[0]} vs {pair[1]}")
        ax.set(title=identifier, xlabel="Time [min]", ylabel="Max difference [% of scale]")
        ax.grid(alpha=0.2)
        ax.legend()
    axes.flat[-1].set_visible(False)
    fig.suptitle("Concentration mesh refinement | Shared interior points")
    fig.savefig(output / "concentration_convergence.png", dpi=180)
    plt.close(fig)
    print("Field          Pair     Max [mol/m3]   RMS [mol/m3]   Max [% scale]")
    for row in rows:
        print(f"{row['field']:14s} {row['coarse_points']:2d}-{row['fine_points']:2d} "
              f"{row['max_abs_difference_mol_m3']:14.6f} {row['rms_difference_mol_m3']:14.6f} "
              f"{row['max_difference_percent_of_scale']:14.6f}")
    print("Sampled differences only; the finest mesh is not an exact solution.")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
