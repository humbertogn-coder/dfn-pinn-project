"""Measure terminal-voltage sensitivity to joint spatial mesh refinement."""

from datetime import datetime, timezone
import json
from pathlib import Path

from export_pybamm_reference import run_discharge
import matplotlib.pyplot as plt
import numpy as np
import pybamm


def main() -> None:
    resolutions = (20, 40, 80)
    runs = []
    for points in resolutions:
        print(f"Running mesh with {points} points per domain...", flush=True)
        solution, _, protocol, mesh, elapsed = run_discharge(points, "1 second")
        t = np.asarray(solution["Time [s]"].entries).reshape(-1)
        v = np.asarray(solution["Voltage [V]"].entries).reshape(-1)
        if (len(t) < 2 or t.shape != v.shape or not np.all(np.isfinite(t))
                or not np.all(np.isfinite(v)) or not np.all(np.diff(t) > 0)):
            raise RuntimeError("Invalid simulation output.")
        if abs(v[-1] - 2.5) > 0.01:
            raise RuntimeError("The requested voltage cutoff was not reached.")
        runs.append((t, v, elapsed, mesh))

    # Sample only shared integer-second outputs; never extrapolate past cutoff.
    end_s = min(run[0][-1] for run in runs)
    common_t = np.arange(0, np.floor(end_s) + 1)
    voltages = []
    for t, v, _, _ in runs:
        indices = np.searchsorted(t, common_t)
        if not np.allclose(t[indices], common_t, rtol=0, atol=1e-6):
            raise RuntimeError("Expected shared one-second output times.")
        voltages.append(v[indices])
    reference = voltages[-1]
    rows = []
    for points, run, voltage in zip(resolutions, runs, voltages):
        difference_mv = (voltage - reference) * 1000
        rows.append({
            "points_per_domain": points,
            "max_abs_voltage_difference_mV": float(np.max(np.abs(difference_mv))),
            "rms_voltage_difference_mV": float(np.sqrt(np.mean(difference_mv**2))),
            "cutoff_time_s": float(run[0][-1]),
            "cutoff_time_difference_s": float(run[0][-1] - runs[-1][0][-1]),
            "solve_wall_time_s": run[2],
        })

    now = datetime.now(timezone.utc)
    output = (Path(__file__).resolve().parents[1] / "results"
              / ("mesh_convergence_" + now.strftime("%Y%m%dT%H%M%S%fZ")))
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_utc": now.isoformat(), "pybamm_version": pybamm.__version__,
        "model": "isothermal DFN", "parameter_set": "Chen2020",
        "protocol": protocol, "initial_soc": 1.0,
        "solver": "IDAKLUSolver", "rtol": 1e-6, "atol": 1e-8,
        "output_period_s": 1, "meshes": [run[3] for run in runs],
        "comparison_end_s": float(common_t[-1]), "reference_points_per_domain": 80,
        "interpretation": "Differences from the finest tested mesh, not exact errors. "
                          "No convergence certification or conservation audit is implied. "
                          "Solver tolerances and internal states require separate checks.",
        "metrics": rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savetxt(output / "metrics.csv", [[row[key] for key in rows[0]] for row in rows],
               delimiter=",", header=",".join(rows[0]), comments="")
    for points, (t, v, _, _) in zip(resolutions, runs):
        np.savetxt(output / f"voltage_mesh_{points}.csv", np.column_stack([t, v]),
                   delimiter=",", header="time_s,voltage_V", comments="")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), layout="constrained")
    for points, (t, v, _, _) in zip(resolutions, runs):
        axes[0].plot(t / 60, v, label=f"{points} points/domain")
    for points, voltage in zip(resolutions[:-1], voltages[:-1]):
        axes[1].plot(common_t / 60, (voltage - reference) * 1000,
                     label=f"{points} minus 80")
    axes[0].set(title="DFN mesh refinement | Chen2020 | 1C", ylabel="Voltage [V]")
    axes[1].set(xlabel="Time [min]", ylabel="Voltage difference [mV]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "mesh_convergence.png", dpi=180)
    plt.close(fig)
    print("Mesh   Max difference [mV]   RMS difference [mV]   Cutoff difference [s]")
    for row in rows:
        print(f"{row['points_per_domain']:4d}   "
              f"{row['max_abs_voltage_difference_mV']:19.4f}   "
              f"{row['rms_voltage_difference_mV']:19.4f}   "
              f"{row['cutoff_time_difference_s']:21.4f}")
    print("The 80-point mesh is a comparison reference, not an exact solution.")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
