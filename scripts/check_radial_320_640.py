"""Compare startup voltage and surface concentrations at radial 320/640."""

import csv
import gc
from datetime import datetime, timezone
import json
from pathlib import Path

from export_pybamm_reference import run_discharge
import matplotlib.pyplot as plt
import numpy as np
import pybamm


def main():
    time = np.linspace(0, 10, 201)
    runs, settings, coordinates, scales = {}, {}, {}, {"voltage": 1.0}
    specs = {"voltage": "Voltage [V]",
             "surface_n": "Negative particle surface concentration [mol.m-3]",
             "surface_p": "Positive particle surface concentration [mol.m-3]"}
    for radial in (320, 640):
        print(f"Running 10-second comparison: x80_r{radial}...", flush=True)
        solution, parameters, protocol, mesh, elapsed = run_discharge(
            80, "0.05 seconds", radial_points=radial, duration_s=10,
            rtol=1e-8, atol=1e-10)
        actual_time = np.asarray(solution["Time [s]"].entries).reshape(-1)
        if actual_time.shape != time.shape or not np.allclose(actual_time, time, rtol=0, atol=1e-9):
            raise ValueError("Expected shared times from 0 to 10 seconds")
        runs[radial] = {}
        for key, name in specs.items():
            variable = solution[name]
            values = np.asarray(variable.entries)
            expected = (len(time),) if key == "voltage" else (80, len(time))
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid field: {key}")
            if key != "voltage":
                x = np.asarray(variable.mesh.nodes)
                if key in coordinates and not np.array_equal(x, coordinates[key]):
                    raise ValueError("Surface coordinates changed between runs")
                coordinates[key] = x.copy()
                domain = "negative" if key == "surface_n" else "positive"
                scales[key] = float(parameters[f"Maximum concentration in {domain} electrode [mol.m-3]"])
            runs[radial][key] = values.copy()
        settings[str(radial)] = {"mesh": mesh, "protocol": protocol,
                                 "solve_wall_time_s": elapsed, "termination": str(solution.termination)}
        del variable, values, solution
        gc.collect()
    rows, curves = [], {}
    for key in specs:
        delta = runs[320][key] - runs[640][key]
        curves[key] = np.abs(delta) if key == "voltage" else np.max(np.abs(delta), axis=0)
        for window, mask in (("0_to_5_s", time <= 5), ("5_to_10_s", time > 5),
                             ("0_to_10_s", np.ones(len(time), dtype=bool))):
            absolute = np.abs(delta[..., mask])
            index = np.unravel_index(np.argmax(absolute), absolute.shape)
            maximum = float(absolute.max())
            rows.append({"window": window, "field": key,
                         "max_abs_difference": maximum,
                         "unit": "V" if key == "voltage" else "mol/m3",
                         "rms_difference": float(np.sqrt(np.mean(absolute**2))),
                         "peak_time_s": float(time[mask][index[-1]]),
                         "peak_x_m": None if key == "voltage" else float(coordinates[key][index[0]]),
                         "fixed_scale": scales[key], "scaled_limit": 0.001,
                         "difference_over_limit": maximum / scales[key] / 0.001,
                         "sampled_pass": maximum / scales[key] <= 0.001})
    output = Path(__file__).resolve().parents[1] / "results" / (
        "radial_320_640_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    report = {"pybamm_version": pybamm.__version__, "parameter_set": "Chen2020",
              "model": "isothermal DFN", "initial_soc": 1.0, "solver": "IDAKLUSolver",
              "rtol": 1e-8, "atol": 1e-10, "cases": settings,
              "time_s": time.tolist(), "surface_coordinates_m": {k: v.tolist() for k, v in coordinates.items()},
              "source_variables": specs, "metrics": rows,
              "status": "PROVISIONAL; spatial refinement and intersample extrema remain unresolved",
              "scope": "Voltage and both native surface concentrations only. No cross-mesh "
                       "interpolation. Both radial meshes refined together, x fixed at 80. "
                       "The second window uses 5.05-10 s; 5 s is included in startup. "
                       "RMS is unweighted over sampled points and times. Limits are the "
                       "existing 1 mV and 0.1% material-maximum criteria. No full-discharge "
                       "acceptance, internal-particle accuracy or conservation certification."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "samples.npz", time_s=time,
                        **{f"r{radial}_{key}": value for radial, fields in runs.items() for key, value in fields.items()},
                        **{f"x_{key}": value for key, value in coordinates.items()})
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), layout="constrained")
    for ax, key in zip(axes, specs):
        factor = 1000 if key == "voltage" else 100 / scales[key]
        ax.plot(time, curves[key] * factor, label="r320 vs r640")
        ax.axhline(1 if key == "voltage" else 0.1, color="black", linestyle="--", label="Provisional limit")
        ax.axvline(5, color="gray", linestyle=":")
        ax.set(title=key, xlabel="Time [s]", ylabel="Difference [mV]" if key == "voltage" else "Difference [% of c_max]")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Radial 320/640 | Fixed x80 | Sampled startup assessment")
    fig.savefig(output / "radial_320_640.png", dpi=180)
    plt.close(fig)
    for row in rows:
        print(f"{row['window']:12s} {row['field']:10s} {'PASS' if row['sampled_pass'] else 'FAIL'} "
              f"difference/limit={row['difference_over_limit']:.6f}, peak={row['peak_time_s']:.2f}s")
    print(report["status"])
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
