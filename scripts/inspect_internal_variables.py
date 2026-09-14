"""Inventory native DFN fields and coordinates before designing an HDF5 schema."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from export_pybamm_reference import run_discharge
import numpy as np
import pybamm


def coordinate_record(mesh, size, axis, domains):
    """Match an entries axis to native nodes or edges, excluding ghost points."""
    matches = [(location, np.asarray(getattr(mesh, location)))
               for location in ("nodes", "edges")
               if len(getattr(mesh, location)) == size]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous native coordinates for {axis}: {size} entries")
    location, values = matches[0]
    if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
        raise ValueError(f"Invalid coordinates for {axis}")
    return {"axis": axis, "unit": "m", "location": location, "domains": domains,
            "count": size, "min": float(values[0]), "max": float(values[-1]),
            "values": values.tolist()}


def collect_inventory():
    """Return the solved reference and checked inventory for reuse by exporters."""
    print("Inspecting DFN internal fields: 80 points/domain, 10-second outputs...", flush=True)
    solution, parameters, protocol, mesh, elapsed = run_discharge(
        80, "10 seconds", rtol=1e-8, atol=1e-10
    )
    specs = []
    for domain, suffix in (("Negative", "n"), ("Positive", "p")):
        specs.extend([
            (f"c_s_{suffix}", f"{domain} particle concentration [mol.m-3]", "mol/m3", "primary"),
            (f"c_s_surf_{suffix}", f"{domain} particle surface concentration [mol.m-3]", "mol/m3", "auxiliary"),
            (f"phi_s_{suffix}", f"{domain} electrode potential [V]", "V", "primary"),
            (f"j_{suffix}", f"{domain} electrode interfacial current density [A.m-2]", "A/m2", "primary"),
            (f"aj_{suffix}", f"{domain} electrode volumetric interfacial current density [A.m-3]", "A/m3", "auxiliary"),
            (f"j0_{suffix}", f"{domain} electrode exchange current density [A.m-2]", "A/m2", "auxiliary"),
            (f"eta_{suffix}", f"{domain} electrode reaction overpotential [V]", "V", "auxiliary"),
            (f"U_{suffix}", f"{domain} electrode open-circuit potential [V]", "V", "auxiliary"),
            (f"i_s_{suffix}", f"{domain} electrode current density [A.m-2]", "A/m2", "auxiliary"),
            (f"a_s_{suffix}", f"{domain} electrode surface area to volume ratio [m-1]", "1/m", "auxiliary"),
        ])
    specs.extend([
        ("c_e", "Electrolyte concentration [mol.m-3]", "mol/m3", "primary"),
        ("phi_e", "Electrolyte potential [V]", "V", "primary"),
        ("i_e", "Electrolyte current density [A.m-2]", "A/m2", "auxiliary"),
    ])
    t = np.asarray(solution["Time [s]"].entries).reshape(-1)
    if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise RuntimeError("Invalid time coordinates.")
    if abs(float(solution["Voltage [V]"].entries[-1]) - 2.5) > 0.01:
        raise RuntimeError("The voltage cutoff was not reached.")
    records = []
    for identifier, name, unit, role in specs:
        variable = solution[name]
        values = np.asarray(variable.entries)
        symbol = variable.base_variables[0]
        coordinates = []
        if variable.dimensions not in (1, 2):
            raise RuntimeError(f"Unexpected dimensions for {name}")
        # ProcessedVariable uses primary, secondary, then time for native entries.
        for index, level in enumerate(("primary", "secondary")[:variable.dimensions]):
            spatial_symbols = variable.spatial_variables[level]
            names = [spatial if isinstance(spatial, str) else spatial.name
                     for spatial in spatial_symbols if spatial != "tabs"]
            if names and all(name.startswith("r") for name in names):
                axis = "r"
            elif names and all(name.startswith("x") for name in names):
                axis = "x"
            else:
                raise ValueError(f"Unsupported spatial variables: {names}")
            native_mesh = variable.mesh if index == 0 else symbol.secondary_mesh
            record = coordinate_record(native_mesh, values.shape[index], axis,
                                       symbol.domains[level])
            record["spatial_variable_names"] = names
            coordinates.append(record)
        expected = tuple(coord["count"] for coord in coordinates) + (len(t),)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Invalid shape or nonfinite entries for {name}")
        record = {"id": identifier, "pybamm_name": name, "unit": unit, "role": role,
                  "shape": list(values.shape), "axis_order": [c["axis"] for c in coordinates] + ["t"],
                  "coordinates": coordinates, "domains": symbol.domains,
                  "minimum": float(values.min()), "maximum": float(values.max()),
                  "all_finite": True, "native_array_bytes": int(values.nbytes)}
        records.append(record)
        print(f"{identifier:12s} {str(values.shape):18s} {record['axis_order']} "
              f"range=[{record['minimum']:.6g}, {record['maximum']:.6g}] {unit}")
    now = datetime.now(timezone.utc)
    report = {
        "created_utc": now.isoformat(), "pybamm_version": pybamm.__version__,
        "python_version": platform.python_version(), "model": "isothermal DFN",
        "parameter_set": "Chen2020", "protocol": protocol, "initial_soc": 1.0,
        "mesh_points": mesh, "rtol": 1e-8, "atol": 1e-10, "solver": "IDAKLUSolver",
        "output_period_s": 10, "solve_wall_time_s": elapsed,
        "termination": str(solution.termination),
        "time": {"unit": "s", "values": t.tolist()},
        "layer_thicknesses_m": {d: float(parameters[f"{d} thickness [m]"])
                                for d in ("Negative electrode", "Separator", "Positive electrode")},
        "variables": records,
        "scope": "Inventory only; field arrays are not exported. Ranges cover sampled "
                 "native entries, not all boundary values. Native particle order is r,x,t. "
                 "Surface concentrations are separate boundary outputs, not the last radial node. "
                 "j uses active surface area; i_s and i_e use geometric area. "
                 "Accuracy and coordinate alignment need validation before PINN training.",
    }
    return solution, report


def main() -> None:
    _, report = collect_inventory()
    records = report["variables"]
    now = datetime.now(timezone.utc)
    output = (Path(__file__).resolve().parents[1] / "results"
              / ("internal_inventory_" + now.strftime("%Y%m%dT%H%M%S%fZ")))
    output.mkdir(parents=True, exist_ok=False)
    (output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# DFN Internal Variable Inventory", "",
             f"PyBaMM {pybamm.__version__}; Chen2020; 1C; 80 points/domain.", "",
             "| ID | Unit | Native shape | Axes | Minimum | Maximum |",
             "|---|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['id']} | {r['unit']} | {r['shape']} | "
                     f"{','.join(r['axis_order'])} | {r['minimum']:.6g} | {r['maximum']:.6g} |")
    lines.extend(["", "Coordinates and exact PyBaMM names are recorded in inventory.json.",
                  "", report["scope"], ""])
    (output / "inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Inspected {len(records)} variables; all entries finite and axes matched.")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
