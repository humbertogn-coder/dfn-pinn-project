"""Export the 23 inventoried DFN fields and verify an exact HDF5 round trip."""

from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

from inspect_internal_variables import collect_inventory


def write_reference(path, solution, report):
    """Write native arrays with per-field coordinates and HDF5 dimension scales."""
    with h5py.File(path, "x") as handle:
        handle.attrs["schema_version"] = "1.0"
        handle.attrs["validation_status"] = "round_trip_pending"
        handle.create_dataset("metadata_json", data=json.dumps(report),
                              dtype=h5py.string_dtype("utf-8"))
        time = handle.create_dataset("time", data=report["time"]["values"])
        time.attrs["unit"] = "s"
        time.make_scale("t")
        for record in report["variables"]:
            group = handle.create_group("fields/" + record["id"])
            data = group.create_dataset(
                "values", data=solution[record["pybamm_name"]].entries,
                compression="gzip", compression_opts=4, shuffle=True, fletcher32=True,
            )
            for key in ("pybamm_name", "unit", "role"):
                data.attrs[key] = record[key]
            data.attrs["axis_order_json"] = json.dumps(record["axis_order"])
            data.attrs["domains_json"] = json.dumps(record["domains"])
            for index, coordinate in enumerate(record["coordinates"]):
                axis = coordinate["axis"]
                coord = group.create_dataset("coordinates/" + axis, data=coordinate["values"])
                coord.attrs["unit"] = coordinate["unit"]
                coord.attrs["location"] = coordinate["location"]
                coord.attrs["domains_json"] = json.dumps(coordinate["domains"])
                coord.make_scale(axis)
                data.dims[index].attach_scale(coord)
                data.dims[index].label = axis
            data.dims[data.ndim - 1].attach_scale(time)
            data.dims[data.ndim - 1].label = "t"
        for name, variable, unit in (
            ("voltage", "Voltage [V]", "V"), ("current", "Current [A]", "A"),
        ):
            data = handle.create_dataset("terminal/" + name, data=solution[variable].entries)
            data.attrs["unit"] = unit
            data.attrs["pybamm_name"] = variable
            data.dims[0].attach_scale(time)
            data.dims[0].label = "t"


def verify_reference(path, solution, report):
    """Fail on changed values, dtypes, metadata, coordinates, or axis associations."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    with h5py.File(path, "r") as handle:
        require(handle.attrs["schema_version"] == "1.0", "Schema mismatch")
        require(json.loads(handle["metadata_json"].asstr()[()]) == report, "Metadata mismatch")
        require(np.array_equal(handle["time"][:], report["time"]["values"]), "Time mismatch")
        require(handle["time"].attrs["unit"] == "s", "Time unit mismatch")
        require(set(handle["fields"]) == {r["id"] for r in report["variables"]}, "Field list mismatch")
        for record in report["variables"]:
            group = handle["fields/" + record["id"]]
            data = group["values"]
            original = np.asarray(solution[record["pybamm_name"]].entries)
            require(data.dtype == original.dtype and np.array_equal(data[:], original),
                    f"Field mismatch: {record['id']}")
            for key in ("unit", "pybamm_name", "role"):
                require(data.attrs[key] == record[key], f"Attribute mismatch: {key}")
            require(json.loads(data.attrs["axis_order_json"]) == record["axis_order"], "Axis order mismatch")
            require(json.loads(data.attrs["domains_json"]) == record["domains"], "Domain mismatch")
            for index, coordinate in enumerate(record["coordinates"]):
                axis = coordinate["axis"]
                coord = group["coordinates/" + axis]
                require(np.array_equal(coord[:], coordinate["values"]), "Coordinate mismatch")
                require(len(coord) == data.shape[index], "Coordinate length mismatch")
                for key in ("unit", "location"):
                    require(coord.attrs[key] == coordinate[key], "Coordinate attribute mismatch")
                require(json.loads(coord.attrs["domains_json"]) == coordinate["domains"], "Coordinate domain mismatch")
                require(data.dims[index].label == axis and len(data.dims[index]) == 1
                        and data.dims[index][0].name == coord.name, "Spatial scale mismatch")
            last = data.dims[data.ndim - 1]
            require(last.label == "t" and len(last) == 1 and last[0].name == "/time", "Time scale mismatch")
        for name, variable, unit in (("voltage", "Voltage [V]", "V"), ("current", "Current [A]", "A")):
            data = handle["terminal/" + name]
            require(np.array_equal(data[:], solution[variable].entries), "Terminal data mismatch")
            require(data.attrs["unit"] == unit and data.attrs["pybamm_name"] == variable, "Terminal metadata mismatch")
            require(data.shape == handle["time"].shape and data.dims[0].label == "t"
                    and len(data.dims[0]) == 1 and data.dims[0][0].name == "/time", "Terminal scale mismatch")


def main():
    solution, report = collect_inventory()
    report["scope"] = (
        "Full native sampled arrays for the 23 inventoried fields. No resampling or "
        "normalization. Particle order is r,x,t; other fields use x,t. Surface "
        "concentrations are separate outputs. j uses active area; i_s and i_e use "
        "geometric area. Coordinates distinguish nodes and edges. Export fidelity "
        "does not certify physical accuracy or mesh convergence."
    )
    report["h5py_version"] = h5py.__version__
    report["numpy_version"] = np.__version__
    now = datetime.now(timezone.utc)
    output = (Path(__file__).resolve().parents[1] / "results"
              / ("internal_fields_" + now.strftime("%Y%m%dT%H%M%S%fZ")))
    output.mkdir(parents=True, exist_ok=False)
    pending = output / "reference.pending.h5"
    final = output / "reference.h5"
    write_reference(pending, solution, report)
    verify_reference(pending, solution, report)
    with h5py.File(pending, "r+") as handle:
        handle.attrs["validation_status"] = "exact_round_trip_verified"
    pending.rename(final)
    validation = {"status": "exact_round_trip_verified", "fields_checked": len(report["variables"]),
                  "checks": "Exact native arrays and dtypes, coordinates, units, metadata, dimension scales, terminal signals",
                  "file_size_bytes": final.stat().st_size}
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(f"Verified exact round trip for {len(report['variables'])} internal fields.")
    print(f"HDF5 size: {final.stat().st_size / 1024**2:.2f} MiB")
    print(f"Output file: {final}")


if __name__ == "__main__":
    main()
