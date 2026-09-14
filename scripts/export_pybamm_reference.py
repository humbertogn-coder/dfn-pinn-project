"""Run an initial DFN discharge and export terminal signals for inspection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybamm


def run_discharge(points: int = 20, period: str = "10 seconds", *,
                  rtol: float = 1e-6, atol: float = 1e-8):
    """Solve the shared protocol with specified mesh and solver tolerances."""
    model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
    parameters = pybamm.ParameterValues("Chen2020")
    protocol = "Discharge at 1C for 2 hours or until 2.5 V"
    mesh = {name: points for name in ("x_n", "x_s", "x_p", "r_n", "r_p")}
    solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameters,
        experiment=pybamm.Experiment([protocol], period=period),
        solver=solver,
        var_pts=mesh,
    )
    started = time.perf_counter()
    solution = simulation.solve(initial_soc=1.0)
    elapsed = time.perf_counter() - started
    return solution, parameters, protocol, mesh, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["1C"], default="1C")
    args = parser.parse_args()
    print("Running isothermal DFN with Chen2020 parameters at 1C...", flush=True)
    solution, parameters, protocol, mesh, elapsed = run_discharge()
    signals = {
        "time_s": np.asarray(solution["Time [s]"].entries).reshape(-1),
        "voltage_V": np.asarray(solution["Voltage [V]"].entries).reshape(-1),
        "current_A": np.asarray(solution["Current [A]"].entries).reshape(-1),
        "discharge_capacity_Ah": np.asarray(
            solution["Discharge capacity [A.h]"].entries
        ).reshape(-1),
    }
    t = signals["time_s"]
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        raise RuntimeError("Expected strictly increasing time samples.")
    if any(values.shape != t.shape or not np.all(np.isfinite(values))
           for values in signals.values()):
        raise RuntimeError("Output signals contain invalid values or mismatched shapes.")
    expected_current = float(parameters["Nominal cell capacity [A.h]"])
    if not np.allclose(signals["current_A"], expected_current, rtol=1e-6):
        raise RuntimeError("Discharge current does not match the requested 1C rate.")
    if abs(signals["voltage_V"][-1] - 2.5) > 0.01:
        raise RuntimeError("The simulation did not reach the requested voltage cutoff.")

    now = datetime.now(timezone.utc)
    output = (Path(__file__).resolve().parents[1] / "results"
              / ("chen2020_1C_" + now.strftime("%Y%m%dT%H%M%S%fZ")))
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_utc": now.isoformat(),
        "pybamm_version": pybamm.__version__,
        "python_version": platform.python_version(),
        "model": "DFN", "thermal": "isothermal", "parameter_set": "Chen2020",
        "protocol_label": args.protocol, "protocol": protocol,
        "initial_soc": 1.0, "sample_period_s": 10,
        "solver": "IDAKLUSolver", "rtol": 1e-6, "atol": 1e-8,
        "mesh_points": mesh, "solve_wall_time_s": elapsed,
        "termination": str(solution.termination),
        "duration_s": float(t[-1]),
        "final_voltage_V": float(signals["voltage_V"][-1]),
        "nominal_capacity_Ah": expected_current,
        "validation_status": "Initial smoke run; mesh convergence and conservation audit pending",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    np.savetxt(output / "terminal_signals.csv", np.column_stack(list(signals.values())),
               delimiter=",", header=",".join(signals), comments="")
    with h5py.File(output / "reference.h5", "w") as handle:
        handle.attrs["metadata_json"] = json.dumps(metadata)
        for name, values in signals.items():
            handle.create_dataset(name, data=values)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.plot(t / 60, signals["voltage_V"], color="teal", linewidth=2)
    ax.set(xlabel="Time [min]", ylabel="Terminal voltage [V]",
           title="DFN discharge at 1C | Chen2020 | Isothermal")
    ax.grid(alpha=0.25)
    fig.savefig(output / "voltage.png", dpi=180)
    plt.close(fig)
    print(f"Simulation completed in {elapsed:.2f} s.")
    print(f"Discharge duration: {t[-1] / 60:.2f} min")
    print(f"Final voltage: {signals['voltage_V'][-1]:.4f} V")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
