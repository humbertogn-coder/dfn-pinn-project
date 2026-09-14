"""Apply provisional sampled criteria to radial or through-cell DFN refinement."""

import gc
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("radial", "spatial"), default="radial")
    args = parser.parse_args()
    levels = (160, 320) if args.study == "radial" else (80, 160)
    specs = [("c_s_n", "Negative particle concentration [mol.m-3]", "negative"),
             ("c_s_p", "Positive particle concentration [mol.m-3]", "positive"),
             ("c_s_surf_n", "Negative particle surface concentration [mol.m-3]", "negative"),
             ("c_s_surf_p", "Positive particle surface concentration [mol.m-3]", "positive"),
             ("c_e_n", "Negative electrolyte concentration [mol.m-3]", "electrolyte"),
             ("c_e_s", "Separator electrolyte concentration [mol.m-3]", "electrolyte"),
             ("c_e_p", "Positive electrolyte concentration [mol.m-3]", "electrolyte")]
    runs, targets, scales, run_reports = {}, {}, {}, []
    for window, period, duration in (("startup", "0.05 seconds", 5), ("full", "10 seconds", None)):
        for level in levels:
            spatial, radial = (80, level) if args.study == "radial" else (level, 320)
            print(f"Assessing {window}: x{spatial}_r{radial}...", flush=True)
            solution, parameters, protocol, mesh, elapsed = run_discharge(
                spatial, period, radial_points=radial, duration_s=duration, rtol=1e-8, atol=1e-10)
            t = np.asarray(solution["Time [s]"].entries).reshape(-1)
            if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
                raise ValueError("Invalid times")
            voltage = np.asarray(solution["Voltage [V]"].entries).reshape(-1)
            if duration is None and abs(voltage[-1] - 2.5) > 0.01:
                raise ValueError("Voltage cutoff not reached")
            if duration is not None and abs(t[-1] - duration) > 1e-8:
                raise ValueError("Short run ended early")
            fields = {"voltage": voltage}
            native_bytes = 0
            for key, name, material in specs:
                variable = solution[name]
                values = np.asarray(variable.entries)
                coords = [np.asarray(variable.mesh.nodes)]
                if variable.dimensions == 2:
                    coords.append(np.asarray(variable.base_variables[0].secondary_mesh.nodes))
                if values.shape[-1] != len(t):
                    raise ValueError("Time axis mismatch")
                if key not in targets:
                    targets[key] = [c.copy() for c in coords]
                    if variable.dimensions == 2:
                        radius = float(parameters[f"{material.capitalize()} particle radius [m]"])
                        targets[key][0] = (np.arange(80) + 0.5) * radius / 80
                    scale_key = ("Initial concentration in electrolyte [mol.m-3]" if material == "electrolyte"
                                 else f"Maximum concentration in {material} electrode [mol.m-3]")
                    scales[key] = float(parameters[scale_key])
                fields[key] = sample_spatial(values, coords, targets[key])
                native_bytes += values.nbytes
            def signal(name):
                a = np.asarray(solution[name].entries).reshape(-1)
                if a.shape != t.shape or not np.all(np.isfinite(a)):
                    raise ValueError(f"Invalid signal: {name}")
                return a
            current = signal("Current [A]")
            if not np.allclose(current, float(parameters["Nominal cell capacity [A.h]"]), rtol=1e-8):
                raise ValueError("Expected positive constant 1C current")
            area = float(parameters.evaluate(pybamm.geometric_parameters.A_cc))
            residuals = []
            inventory = signal("Total lithium in electrolyte [mol]").copy()
            for domain, sign in (("negative", 1), ("positive", -1)):
                reaction = signal(f"X-averaged {domain} electrode volumetric interfacial current density [A.m-3]")
                length = float(parameters[f"{domain.capitalize()} electrode thickness [m]"])
                residuals.append(float(np.max(np.abs(area * length * reaction - sign * current))) / float(np.max(np.abs(current))))
                inventory += signal(f"Total lithium in {domain} electrode [mol]")
            lithium = float(np.max(np.abs(inventory - inventory[0])) / inventory[0])
            run_reports.append({"window": window, "radial_points": radial, "spatial_points": spatial, "mesh": mesh,
                                "protocol": protocol, "solve_wall_time_s": elapsed,
                                "end_time_s": float(t[-1]), "max_relative_current_residual": max(residuals),
                                "max_relative_lithium_drift": lithium,
                                "current_pass": max(residuals) <= 1e-5, "lithium_pass": lithium <= 1e-8,
                                "seven_concentration_arrays_uncompressed_MiB": native_bytes / 1024**2})
            runs[(window, level)] = (t.copy(), fields)
            del solution, variable, values, signal
            gc.collect()
    rows, curves = [], {}
    for window in ("startup", "full"):
        t0, a = runs[(window, levels[0])]
        t1, b = runs[(window, levels[1])]
        common = t0[t0 <= min(t0[-1], t1[-1])]
        if window == "full":
            common = common[common > 5]
        ia, ib = np.searchsorted(t0, common), np.searchsorted(t1, common)
        if not np.allclose(t1[ib], common, rtol=0, atol=1e-8):
            raise ValueError("Shared output samples required")
        for key in a:
            delta = a[key][..., ia] - b[key][..., ib]
            if not np.all(np.isfinite(delta)):
                raise ValueError("Nonfinite comparison")
            scale = 1 if key == "voltage" else scales[key]
            maximum = float(np.max(np.abs(delta)))
            rows.append({"window": window, "field": key, "max_abs_difference": maximum,
                         "unit": "V" if key == "voltage" else "mol/m3", "scale": scale,
                         "scaled_max_difference": maximum / scale, "limit": 0.001,
                         "sampled_pass": maximum / scale <= 0.001})
            spatial_axes = tuple(range(delta.ndim - 1))
            curves[f"{window}_{key}"] = np.max(np.abs(delta), axis=spatial_axes) / scale / 0.001
        curves[f"{window}_time"] = common
    output = Path(__file__).resolve().parents[1] / "results" / (
        f"reference_assessment_{args.study}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    report = {"criteria_version": 1, "study": args.study, "pybamm_version": pybamm.__version__,
              "rtol": 1e-8, "atol": 1e-10, "model": "isothermal DFN", "parameter_set": "Chen2020",
              "runs": run_reports, "comparisons": rows,
              "targets_m": {key: [c.tolist() for c in coords] for key, coords in targets.items()},
              "sampled_checks_pass": all(row["sampled_pass"] for row in rows)
                                     and all(run["current_pass"] and run["lithium_pass"] for run in run_reports),
              "overall_status": "PROVISIONAL: joint reference selection and sampling audit pending",
              "scope": f"{args.study.capitalize()} sampled checks only. Startup uses independent 5-second runs. "
                       "Later samples start at 10 seconds; 5-10 seconds and intersample extrema "
                       "are unresolved. Linear spatial interpolation to x80/r80 interior targets "
                       "contributes error; electrolyte regions stay separate. Cutoff "
                       "times reported separately. Storage estimates cover seven uncompressed "
                       "concentration arrays only, not a complete HDF5. Runtime is one measurement."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "error_curves.npz", **curves)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, window in zip(axes, ("startup", "full")):
        for key in runs[(window, levels[0])][1]:
            ax.plot(curves[f"{window}_time"], curves[f"{window}_{key}"], label=key)
        ax.axhline(1, color="black", linestyle="--", label="Provisional limit")
        ax.set(title=window, xlabel="Time [s]", ylabel="Sampled difference / allowed difference")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "reference_assessment.png", dpi=180)
    plt.close(fig)
    for row in rows:
        print(f"{row['window']:8s} {row['field']:12s} {'PASS' if row['sampled_pass'] else 'FAIL'} "
              f"difference/limit={row['scaled_max_difference']/row['limit']:.6f}")
    for run in run_reports:
        print(f"{run['window']} x{run['spatial_points']}_r{run['radial_points']}: current_pass={run['current_pass']}, "
              f"lithium_pass={run['lithium_pass']}, solve={run['solve_wall_time_s']:.2f}s")
    print(report["overall_status"])
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
