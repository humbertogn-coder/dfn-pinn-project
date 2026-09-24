"""Short PyBaMM DFN reference matching the synthetic-IC assembly settings."""

from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import time
import h5py
import numpy as np
import pybamm

from dfn_pinn.dfn_smoke import SETTINGS
from inspect_internal_variables import coordinate_record


def parameters_for(settings):
    p = pybamm.ParameterValues("Chen2020")
    checks = {}
    for index, domain in enumerate(("Negative electrode", "Separator", "Positive electrode")):
        checks[f"{domain} thickness [m]"] = settings["lengths_m"][index]
        checks[f"{domain} porosity"] = settings["porosities"][index]
        checks[f"{domain} Bruggeman coefficient (electrolyte)"] = 1.5
    for k, domain in enumerate(("Negative", "Positive")):
        checks.update({f"{domain} electrode active material volume fraction": settings["solid_fractions"][k],
                       f"{domain} particle radius [m]": settings["radii_m"][k],
                       f"{domain} particle diffusivity [m2.s-1]": settings["diffusivities_m2_s"][k],
                       f"Maximum concentration in {domain.lower()} electrode [mol.m-3]": settings["cmax_mol_m3"][k],
                       f"{domain} electrode conductivity [S.m-1]": settings["solid_conductivities_S_m"][k],
                       f"{domain} electrode Bruggeman coefficient (electrode)": 0.})
    checks.update({"Cation transference number": .2594, "Thermodynamic factor": 1.})
    for name, value in checks.items():
        if not np.isclose(float(p[name]), value, rtol=1e-12, atol=0):
            raise ValueError(f"Assembly parameter mismatch: {name}")
    area = p["Electrode height [m]"]*p["Electrode width [m]"]*p["Number of electrodes connected in parallel to make a cell"]
    if not np.isclose(area, settings["area_m2"], rtol=1e-12, atol=0):
        raise ValueError("Current-collector area mismatch")
    if settings["kinetics_mode"] != "pybamm_26_8" or pybamm.__version__ != "26.8.0.0":
        raise ValueError("Constitutive provenance requires re-audit")
    if pybamm.settings.tolerances["reg_power"] != .001:
        raise ValueError("Unexpected exchange-current regularization")
    p.update({"Initial concentration in negative electrode [mol.m-3]": settings["initial_stoichiometries"][0]*settings["cmax_mol_m3"][0],
              "Initial concentration in positive electrode [mol.m-3]": settings["initial_stoichiometries"][1]*settings["cmax_mol_m3"][1],
              "Initial concentration in electrolyte [mol.m-3]": settings["initial_ce_mol_m3"],
              "Current function [A]": settings["applied_current_A"],
              "Initial temperature [K]": settings["temperature_K"],
              "Ambient temperature [K]": settings["temperature_K"]})
    return p, checks


def export_fields(solution, times, group):
    specs = [("c_e", "Electrolyte concentration [mol.m-3]", "mol/m3"),
             ("phi_e", "Electrolyte potential [V]", "V"), ("i_e", "Electrolyte current density [A.m-2]", "A/m2")]
    for domain, suffix in (("Negative", "n"), ("Positive", "p")):
        specs.extend([(f"c_s_{suffix}", f"{domain} particle concentration [mol.m-3]", "mol/m3"),
                      (f"surface_{suffix}", f"{domain} particle surface concentration [mol.m-3]", "mol/m3"),
                      (f"phi_s_{suffix}", f"{domain} electrode potential [V]", "V"),
                      (f"j_{suffix}", f"{domain} electrode interfacial current density [A.m-2]", "A/m2"),
                      (f"i_s_{suffix}", f"{domain} electrode current density [A.m-2]", "A/m2")])
    group.create_dataset("time_s", data=times)
    for key, name, unit in specs:
        variable = solution[name]
        values = np.asarray(variable.entries)
        if values.shape[-1] != len(times) or not np.isfinite(values).all():
            raise ValueError(f"Invalid reference field: {name}")
        field = group.create_group(key)
        field.create_dataset("values", data=values, compression="gzip")
        axes = []
        for i, level in enumerate(("primary", "secondary")[:variable.dimensions]):
            mesh = variable.mesh if i == 0 else variable.base_variables[0].secondary_mesh
            axis = "r" if key.startswith("c_s_") and i == 0 else "x"
            record = coordinate_record(mesh, values.shape[i], axis, variable.base_variables[0].domains[level])
            field.create_dataset(axis+"_m", data=record.pop("values"))
            axes.append(record)
        field.attrs["metadata_json"] = json.dumps({"name": name, "unit": unit, "coordinates": axes,
                                                   "axis_order": [v["axis"] for v in axes]+["t"]})


def main():
    c = SETTINGS
    parameters, checks = parameters_for(c)
    times = np.unique(np.r_[0., np.geomspace(1e-6, .01, 51), np.linspace(0, c["duration_s"], 101)])
    output = Path(__file__).resolve().parents[1]/"results"/("dfn_smoke_reference_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    runs, common = {}, {}
    with h5py.File(output/"reference.h5", "w") as handle:
        for label, nx, nr in (("x40_r160", 40, 160), ("x80_r320", 80, 320)):
            print(f"Solving matched synthetic-IC DFN: {label}...", flush=True)
            model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
            mesh = {"x_n": nx, "x_s": nx, "x_p": nx, "r_n": nr, "r_p": nr}
            simulation = pybamm.Simulation(model, parameter_values=parameters.copy(), var_pts=mesh,
                                           solver=pybamm.IDAKLUSolver(rtol=1e-9, atol=1e-11))
            start = time.perf_counter()
            # Do not supply initial_soc: retain the explicitly matched concentrations.
            solution = simulation.solve([0., c["duration_s"]], t_interp=times)
            np.testing.assert_allclose(solution.t, times, rtol=0, atol=1e-12)
            voltage = np.asarray(solution["Voltage [V]"].entries).reshape(-1)
            current = np.asarray(solution["Current [A]"].entries).reshape(-1)
            np.testing.assert_allclose(current, c["applied_current_A"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(solution["Electrolyte concentration [mol.m-3]"].entries[..., 0], c["initial_ce_mol_m3"], rtol=0, atol=1e-8)
            for k, domain in enumerate(("Negative", "Positive")):
                np.testing.assert_allclose(solution[f"{domain} particle concentration [mol.m-3]"].entries[..., 0],
                                           c["initial_stoichiometries"][k]*c["cmax_mol_m3"][k], rtol=0, atol=1e-7)
            inventories = [np.asarray(solution[f"Total lithium in {domain} [mol]"].entries).reshape(-1)
                           for domain in ("negative electrode", "positive electrode", "electrolyte")]
            total = sum(inventories)
            conservation = {"max_total_lithium_drift_mol": float(np.abs(total-total[0]).max())}
            for k, domain in enumerate(("negative", "positive")):
                density = np.asarray(solution[f"X-averaged {domain} electrode volumetric interfacial current density [A.m-3]"].entries).reshape(-1)
                net = density*c["area_m2"]*c["lengths_m"][0 if k == 0 else 2]
                conservation[f"{domain}_reaction_current_error_A"] = float(np.abs(net-(1 if k == 0 else -1)*current).max())
            group = handle.create_group(label)
            export_fields(solution, times, group)
            group.create_dataset("voltage_V", data=voltage)
            group.create_dataset("current_A", data=current)
            runs[label] = {"mesh": mesh, "solve_and_export_s": time.perf_counter()-start,
                           "voltage_initial_V": float(voltage[0]), "voltage_final_V": float(voltage[-1]),
                           "termination": str(solution.termination), "conservation": conservation}
            common[label] = voltage
            print(f"Voltage {voltage[0]:.6f} -> {voltage[-1]:.6f} V; max lithium drift={conservation['max_total_lithium_drift_mol']:.3e} mol", flush=True)
    snapshot = {}
    for key, value in parameters.items():
        if isinstance(value, (int, float, str, bool)):
            snapshot[key] = value
        elif callable(value):
            try:
                digest = hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
            except (OSError, TypeError):
                digest = None
            snapshot[key] = {"callable": f"{value.__module__}.{value.__name__}", "source_sha256": digest}
        else:
            snapshot[key] = {"type": type(value).__name__, "representation": repr(value)}
    difference = np.abs(common["x40_r160"]-common["x80_r320"])
    root = Path(__file__).resolve().parents[1]
    report = {"status": "MATCHED_WORKING_REFERENCE_PROVISIONAL", "settings": c, "verified_parameters": checks,
              "initial_soc_override": False, "parameters": snapshot, "runs": runs,
              "pybamm_version": pybamm.__version__, "rtol": 1e-9, "atol": 1e-11,
              "time_samples": len(times), "max_voltage_mesh_difference_V": float(difference.max()),
              "voltage_difference_peak_time_s": float(times[difference.argmax()]),
              "reference_sha256": hashlib.sha256((output/"reference.h5").read_bytes()).hexdigest(),
              "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), root/"src/dfn_pinn/dfn_smoke.py", root/"scripts/inspect_internal_variables.py")},
              "scope": "Same synthetic ICs as assembly, not prior SOC=1 reference. Two joint mesh levels; no internal-field convergence certificate, tolerance study or PINN training."}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Maximum voltage mesh difference: {difference.max():.6e} V", flush=True)
    print(f"No PINN training. Provisional reference: {output}", flush=True)


if __name__ == "__main__":
    main()
