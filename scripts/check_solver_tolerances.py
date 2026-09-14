"""Compare DFN terminal voltages with tighter solver tolerances on a fixed mesh."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from export_pybamm_reference import run_discharge
import matplotlib.pyplot as plt
import numpy as np
import pybamm


def main() -> None:
    tolerances = ((1e-6, 1e-8), (1e-7, 1e-9), (1e-8, 1e-10))
    runs = []
    for rtol, atol in tolerances:
        print(f"Running 80-point mesh: rtol={rtol:.0e}, atol={atol:.0e}...", flush=True)
        solution, _, protocol, mesh, elapsed = run_discharge(
            80, "1 second", rtol=rtol, atol=atol
        )
        t = np.asarray(solution["Time [s]"].entries).reshape(-1)
        v = np.asarray(solution["Voltage [V]"].entries).reshape(-1)
        if (len(t) < 2 or t.shape != v.shape or not np.all(np.isfinite(t))
                or not np.all(np.isfinite(v)) or not np.all(np.diff(t) > 0)):
            raise RuntimeError("Invalid simulation output.")
        if abs(v[-1] - 2.5) > 0.01:
            raise RuntimeError("The voltage cutoff was not reached.")
        runs.append((t, v, elapsed, str(solution.termination)))

    # Compare shared output times without interpolating or extrapolating voltages.
    common_t = np.arange(0, np.floor(min(run[0][-1] for run in runs)) + 1)
    voltages = []
    for t, v, _, _ in runs:
        indices = np.searchsorted(t, common_t)
        if (np.any(indices >= len(t))
                or not np.allclose(t[indices], common_t, rtol=0, atol=1e-6)):
            raise RuntimeError("Expected shared one-second output times.")
        voltages.append(v[indices])
    reference = voltages[-1]
    rows = []
    for (rtol, atol), run, v in zip(tolerances, runs, voltages):
        delta = 1000 * (v - reference)
        rows.append({
            "rtol": rtol, "atol": atol,
            "max_abs_voltage_difference_mV": float(np.max(np.abs(delta))),
            "rms_voltage_difference_mV": float(np.sqrt(np.mean(delta**2))),
            "cutoff_time_s": float(run[0][-1]),
            "cutoff_time_difference_s": float(run[0][-1] - runs[-1][0][-1]),
            "solve_wall_time_s": run[2],
        })
    now = datetime.now(timezone.utc)
    output = (Path(__file__).resolve().parents[1] / "results"
              / ("solver_tolerances_" + now.strftime("%Y%m%dT%H%M%S%fZ")))
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_utc": now.isoformat(), "python_version": platform.python_version(),
        "pybamm_version": pybamm.__version__, "model": "isothermal DFN",
        "parameter_set": "Chen2020", "protocol": protocol, "initial_soc": 1.0,
        "mesh_points": mesh, "solver": "IDAKLUSolver", "output_period_s": 1,
        "comparison_end_s": float(common_t[-1]),
        "reference_tolerances": {"rtol": tolerances[-1][0], "atol": tolerances[-1][1]},
        "terminations": [run[3] for run in runs], "metrics": rows,
        "interpretation": "Sampled terminal-voltage differences from the tightest tested "
                          "tolerances on a fixed mesh, not exact errors. This does not "
                          "certify mesh convergence, internal states, or conservation. "
                          "One-second sampling does not bound differences between samples.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savetxt(output / "metrics.csv", [[row[key] for key in rows[0]] for row in rows],
               delimiter=",", header=",".join(rows[0]), comments="")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), layout="constrained")
    for (rtol, atol), (t, v, _, _) in zip(tolerances, runs):
        label = f"rtol={rtol:.0e}, atol={atol:.0e}"
        np.savetxt(output / f"voltage_rtol_{rtol:.0e}.csv", np.column_stack([t, v]),
                   delimiter=",", header="time_s,voltage_V", comments="")
        axes[0].plot(t / 60, v, label=label)
    for (rtol, atol), v in zip(tolerances[:-1], voltages[:-1]):
        axes[1].plot(common_t / 60, 1000 * (v - reference),
                     label=f"rtol={rtol:.0e}, atol={atol:.0e}")
    axes[0].set(title="DFN solver tolerances | Chen2020 | 1C | 80 points/domain",
                ylabel="Voltage [V]")
    axes[1].set(xlabel="Time [min]", ylabel="Difference from tightest run [mV]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "solver_tolerances.png", dpi=180)
    plt.close(fig)
    print("rtol    atol    Max difference [mV]   RMS difference [mV]   Cutoff difference [s]")
    for row in rows:
        print(f"{row['rtol']:.0e}   {row['atol']:.0e}   "
              f"{row['max_abs_voltage_difference_mV']:19.6f}   "
              f"{row['rms_voltage_difference_mV']:19.6f}   "
              f"{row['cutoff_time_difference_s']:21.6f}")
    print("The tightest run is a comparison reference, not an exact solution.")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
