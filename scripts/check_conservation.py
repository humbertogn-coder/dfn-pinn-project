"""Audit global reaction-current and lithium balances for the baseline DFN."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from export_pybamm_reference import run_discharge
import matplotlib.pyplot as plt
import numpy as np
import pybamm
from scipy.integrate import cumulative_trapezoid


def main() -> None:
    print("Running conservation audit: 80 points/domain, rtol=1e-8, atol=1e-10...", flush=True)
    solution, parameters, protocol, mesh, elapsed = run_discharge(
        80, "1 second", rtol=1e-8, atol=1e-10
    )
    variables = {
        "time_s": "Time [s]", "voltage_V": "Voltage [V]",
        "applied_current_A": "Current [A]",
        "negative_reaction_density_A_m3":
            "X-averaged negative electrode volumetric interfacial current density [A.m-3]",
        "positive_reaction_density_A_m3":
            "X-averaged positive electrode volumetric interfacial current density [A.m-3]",
        "negative_lithium_mol": "Total lithium in negative electrode [mol]",
        "positive_lithium_mol": "Total lithium in positive electrode [mol]",
        "electrolyte_lithium_mol": "Total lithium in electrolyte [mol]",
    }
    signals = {key: np.asarray(solution[name].entries).reshape(-1)
               for key, name in variables.items()}
    t = signals["time_s"]
    if (len(t) < 2 or not np.all(np.diff(t) > 0)
            or any(v.shape != t.shape or not np.all(np.isfinite(v))
                   for v in signals.values())):
        raise RuntimeError("Invalid or mismatched audit signals.")
    if abs(signals["voltage_V"][-1] - 2.5) > 0.01:
        raise RuntimeError("The voltage cutoff was not reached.")
    # Use the installed model's geometric area, consistent with its inventory units.
    area = float(parameters.evaluate(pybamm.geometric_parameters.A_cc))
    faraday = float(pybamm.constants.F.value)
    current = signals["applied_current_A"]
    if not np.allclose(current, float(parameters["Nominal cell capacity [A.h]"]), rtol=1e-8):
        raise RuntimeError("Expected a positive constant 1C discharge current.")
    for domain in ("negative", "positive"):
        length = float(parameters[f"{domain.capitalize()} electrode thickness [m]"])
        signals[f"{domain}_reaction_current_A"] = (
            area * length * signals[f"{domain}_reaction_density_A_m3"]
        )
    # PyBaMM's oxidation-positive convention gives +I in the negative electrode
    # and -I in the positive electrode during this discharge without side reactions.
    signals["negative_current_residual_A"] = signals["negative_reaction_current_A"] - current
    signals["positive_current_residual_A"] = signals["positive_reaction_current_A"] + current
    n = signals["negative_lithium_mol"]
    p = signals["positive_lithium_mol"]
    e = signals["electrolyte_lithium_mol"]
    total = n + p + e
    if np.any(n <= 0) or np.any(p <= 0) or np.any(e <= 0):
        raise RuntimeError("Expected positive lithium inventories.")
    transferred = cumulative_trapezoid(current, t, initial=0) / faraday
    signals["total_lithium_mol"] = total
    signals["total_lithium_drift_mol"] = total - total[0]
    signals["electrolyte_lithium_drift_mol"] = e - e[0]
    signals["negative_transfer_residual_mol"] = n - n[0] + transferred
    signals["positive_transfer_residual_mol"] = p - p[0] - transferred
    signals["charge_equivalent_transfer_mol"] = transferred
    metrics = {}
    for key in ("negative_current_residual_A", "positive_current_residual_A",
                "total_lithium_drift_mol", "electrolyte_lithium_drift_mol",
                "negative_transfer_residual_mol", "positive_transfer_residual_mol"):
        metrics["max_abs_" + key] = float(np.max(np.abs(signals[key])))
    metrics["max_relative_current_residual"] = max(
        metrics["max_abs_negative_current_residual_A"],
        metrics["max_abs_positive_current_residual_A"]
    ) / float(np.max(np.abs(current)))
    metrics["max_relative_total_lithium_drift"] = (
        metrics["max_abs_total_lithium_drift_mol"] / float(total[0])
    )
    now = datetime.now(timezone.utc)
    output = (Path(__file__).resolve().parents[1] / "results"
              / ("conservation_" + now.strftime("%Y%m%dT%H%M%S%fZ")))
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_utc": now.isoformat(), "python_version": platform.python_version(),
        "pybamm_version": pybamm.__version__, "model": "isothermal DFN without side reactions",
        "parameter_set": "Chen2020", "protocol": protocol, "initial_soc": 1.0,
        "mesh_points": mesh, "solver": "IDAKLUSolver", "rtol": 1e-8, "atol": 1e-10,
        "output_period_s": 1, "area_m2": area, "faraday_C_per_mol": faraday,
        "solve_wall_time_s": elapsed, "termination": str(solution.termination),
        "source_variables": variables, "metrics": metrics,
        "equations": ["I_n = A * L_n * x_average(a_n*j_n); residual = I_n - I",
                      "I_p = A * L_p * x_average(a_p*j_p); residual = I_p + I",
                      "N_total = N_negative + N_positive + N_electrolyte",
                      "negative transfer residual = N_n(t) - N_n(0) + integral(I dt)/F",
                      "positive transfer residual = N_p(t) - N_p(0) - integral(I dt)/F"],
        "scope": "Sampled global balances using PyBaMM spatial integrals. Not an "
                 "independent PDE validation, local residual audit, or mesh accuracy "
                 "certificate. No pass threshold is imposed. Charge integration uses "
                 "the trapezoidal rule, exact for this constant applied current.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savetxt(output / "balances.csv", np.column_stack(list(signals.values())),
               delimiter=",", header=",".join(signals), comments="")
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), layout="constrained")
    for domain in ("negative", "positive"):
        axes[0].plot(t / 60, signals[f"{domain}_current_residual_A"], label=domain.capitalize())
        axes[2].plot(t / 60, signals[f"{domain}_transfer_residual_mol"], label=domain.capitalize())
    axes[1].plot(t / 60, signals["total_lithium_drift_mol"], label="Total")
    axes[1].plot(t / 60, signals["electrolyte_lithium_drift_mol"], label="Electrolyte")
    axes[0].set(title="DFN global conservation audit | Chen2020 | 1C", ylabel="Current residual [A]")
    axes[1].set(ylabel="Lithium drift [mol]")
    axes[2].set(xlabel="Time [min]", ylabel="Transfer residual [mol]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    fig.savefig(output / "conservation.png", dpi=180)
    plt.close(fig)
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")
    print("Global balance diagnostics only; no accuracy certification is implied.")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
