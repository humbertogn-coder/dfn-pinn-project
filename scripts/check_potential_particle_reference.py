"""Finite-volume reference for a smooth potential-driven negative particle."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pybamm


RADIUS = 5.86e-6
DIFFUSIVITY = 3.3e-14
CMAX = 33133.
TEMPERATURE = 298.15
END = 10.
AMPLITUDE = .005


def solve(size, times, rtol=1e-9, atol=1e-11):
    parameters = pybamm.ParameterValues("Chen2020")
    ocp = parameters["Negative electrode OCP [V]"]
    exchange = parameters["Negative electrode exchange-current density [A.m-2]"]
    faraday = float(pybamm.constants.F.evaluate())
    gas = float(pybamm.constants.R.evaluate())
    model = pybamm.BaseModel()
    rho = pybamm.SpatialVariable("rho", domain=["sphere"], coord_sys="spherical polar")
    theta = pybamm.Variable("Stoichiometry", domain=["sphere"])
    current = pybamm.Variable("Current")
    charge = pybamm.Variable("Charge")
    surface = pybamm.boundary_value(theta, "right")
    # Raw Chen2020 functions: this new benchmark deliberately does not use RegPower.
    potential = ocp(pybamm.Scalar(.5))+AMPLITUDE*(1-pybamm.cos(np.pi*pybamm.t/END))/2
    j0 = exchange(pybamm.Scalar(1000.), surface*CMAX, CMAX, pybamm.Scalar(TEMPERATURE))
    bv = 2*j0*pybamm.sinh(faraday*(potential-ocp(surface))/(2*gas*TEMPERATURE))
    model.rhs = {theta: DIFFUSIVITY/RADIUS**2*pybamm.div(pybamm.grad(theta)), charge: current}
    model.algebraic = {current: current-bv}
    model.initial_conditions = {theta: pybamm.Scalar(.5), current: pybamm.Scalar(0), charge: pybamm.Scalar(0)}
    model.boundary_conditions = {theta: {"left": (pybamm.Scalar(0), "Neumann"),
                                        "right": (-RADIUS*current/(faraday*DIFFUSIVITY*CMAX), "Neumann")}}
    model.variables = {"Stoichiometry": theta, "Surface": surface, "Current": current,
                       "Charge": charge, "Potential": potential, "BV residual": current-bv}
    mesh = pybamm.Mesh({"sphere": {rho: {"min": 0, "max": 1}}},
                      {"sphere": pybamm.Uniform1DSubMesh}, {rho: size})
    pybamm.Discretisation(mesh, {"sphere": pybamm.FiniteVolume()}).process_model(model)
    solution = pybamm.IDAKLUSolver(rtol=rtol, atol=atol).solve(model, times, t_interp=times)
    values = {name: np.asarray(solution[name].entries) for name in model.variables}
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError("Nonfinite reference solution")
    np.testing.assert_allclose(solution.t, times, rtol=0, atol=1e-12)
    if values["Stoichiometry"].shape != (size, len(times)):
        raise ValueError("Unexpected field axes")
    for name in ("Surface", "Current", "Charge", "Potential", "BV residual"):
        values[name] = values[name].reshape(-1)
    nodes, edges = mesh["sphere"].nodes, mesh["sphere"].edges
    mean = (edges[1:]**3-edges[:-1]**3) @ values["Stoichiometry"]
    balance = mean-.5+3*values["Charge"]/(faraday*RADIUS*CMAX)
    if np.min(values["Stoichiometry"]) <= 0 or np.max(values["Stoichiometry"]) >= 1 or np.min(values["Surface"]) <= 0 or np.max(values["Surface"]) >= 1:
        raise ValueError("Reference left admissible concentration interval")
    values.update(nodes=nodes, mean=mean, balance=balance)
    return values


def main():
    times = np.unique(np.r_[0., np.geomspace(1e-6, .1, 81), np.linspace(0, END, 1001)])
    runs = {}
    for label, size, rtol, atol in (("r80", 80, 1e-9, 1e-11), ("r160", 160, 1e-9, 1e-11),
                                   ("r320", 320, 1e-9, 1e-11), ("r320_tight", 320, 1e-10, 1e-12)):
        print(f"Solving potential-driven particle: {label}...", flush=True)
        runs[label] = solve(size, times, rtol, atol)
    common = runs["r80"]["nodes"]
    comparisons = []
    for left, right in (("r80", "r160"), ("r160", "r320"), ("r320", "r320_tight")):
        a, b = runs[left], runs[right]
        record = {"pair": [left, right]}
        for field in ("Surface", "Current", "mean"):
            difference = np.abs(a[field]-b[field])
            record[field] = {"max_difference": float(difference.max()), "peak_time_s": float(times[difference.argmax()])}
        fields = [np.column_stack([np.interp(common, data["nodes"], data["Stoichiometry"][:, i])
                                    for i in range(len(times))]) for data in (a, b)]
        record["common_grid_concentration_max_difference"] = float(np.abs(fields[0]-fields[1]).max())
        comparisons.append(record)
        print(f"{left}/{right}: surface={record['Surface']['max_difference']:.6e}, current={record['Current']['max_difference']:.6e} A/m2", flush=True)
    diagnostics = {name: {"max_inventory_error": float(np.abs(data["balance"]).max()),
                           "max_bv_residual_A_m2": float(np.abs(data["BV residual"]).max()),
                           "current_range_A_m2": [float(data["Current"].min()), float(data["Current"].max())],
                           "surface_range": [float(data["Surface"].min()), float(data["Surface"].max())]}
                   for name, data in runs.items()}
    checks = {"refinement": all(row["Surface"]["max_difference"] <= 1e-5 and row["Current"]["max_difference"] <= 1e-4
                                and row["common_grid_concentration_max_difference"] <= 1e-5 for row in comparisons[1:]),
              "inventory": all(row["max_inventory_error"] <= 1e-9 for row in diagnostics.values()),
              "kinetics": all(row["max_bv_residual_A_m2"] <= 1e-7 for row in diagnostics.values())}
    output = Path(__file__).resolve().parents[1]/"results"/("potential_particle_reference_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    report = {"status": "SAMPLED_REFERENCE_CHECKS_PASS" if all(checks.values()) else "REFERENCE_CHECKS_FAIL",
              "checks": checks, "comparisons": comparisons, "diagnostics": diagnostics,
              "protocol": "phi_s=U_n(0.5)+0.005*(1-cos(pi*t/10))/2 V; phi_e=0 V",
              "settings": {"radius_m": RADIUS, "D_m2_s": DIFFUSIVITY, "cmax_mol_m3": CMAX, "T_K": TEMPERATURE,
                           "ce_mol_m3": 1000., "initial_stoichiometry": .5, "end_time_s": END, "mode": "raw", "solver": "IDAKLUSolver",
                           "mesh_sizes": [80, 160, 320], "rtol": 1e-9, "atol": 1e-11, "tight_rtol": 1e-10, "tight_atol": 1e-12},
              "pybamm_version": pybamm.__version__, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Independent of Torch/PINN implementation. Sampled mesh/tolerance checks, not exact truth. Shared semidiscrete inventory identity is not an independent accuracy certificate. No training or full DFN."}
    np.savez_compressed(output/"reference.npz", times_s=times, **{f"{label}_{key}": v for label, data in runs.items() for key, v in data.items()})
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(report["status"])
    print(f"Finest current range: {diagnostics['r320_tight']['current_range_A_m2']} A/m2")
    print(f"No training was run. Output directory: {output}")


if __name__ == "__main__":
    main()
