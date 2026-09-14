"""Plot native internal fields from a verified HDF5 file without running PyBaMM."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_field(handle, identifier, axes, unit):
    """Reject incompatible axes, units, or coordinates before plotting."""
    group = handle["fields/" + identifier]
    dataset = group["values"]
    if json.loads(dataset.attrs["axis_order_json"]) != axes or dataset.attrs["unit"] != unit:
        raise ValueError(f"Unexpected axes or unit for {identifier}")
    coordinates = [group["coordinates/" + axis][:] for axis in axes[:-1]]
    coordinates.append(handle["time"][:])
    if dataset.shape != tuple(len(c) for c in coordinates):
        raise ValueError(f"Coordinate shape mismatch for {identifier}")
    for index, (axis, coordinate) in enumerate(zip(axes, coordinates)):
        if not np.all(np.isfinite(coordinate)) or not np.all(np.diff(coordinate) > 0):
            raise ValueError(f"Invalid coordinates for {identifier}")
        expected_path = "/time" if axis == "t" else group["coordinates/" + axis].name
        if (dataset.dims[index].label != axis or len(dataset.dims[index]) != 1
                or dataset.dims[index][0].name != expected_path):
            raise ValueError(f"Invalid dimension scale for {identifier}")
        coord_dataset = handle[expected_path]
        if coord_dataset.attrs["unit"] != ("s" if axis == "t" else "m"):
            raise ValueError(f"Invalid coordinate unit for {identifier}")
    values = dataset[:]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite data for {identifier}")
    return values, coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Reference HDF5; defaults to latest internal_fields run.")
    args = parser.parse_args()
    if args.input is None:
        results = Path(__file__).resolve().parents[1] / "results"
        candidates = sorted(results.glob("internal_fields_*/reference.h5"))
        if not candidates:
            parser.error("No reference file found. Run export_internal_fields.py first.")
        source = candidates[-1]
    else:
        source = args.input.resolve()
    print(f"Reading HDF5: {source}", flush=True)
    with h5py.File(source, "r") as handle:
        if (handle.attrs.get("schema_version") != "1.0"
                or handle.attrs.get("validation_status") != "exact_round_trip_verified"):
            raise ValueError("Expected a verified schema 1.0 reference.")
        metadata = json.loads(handle["metadata_json"].asstr()[()])
        t = handle["time"][:]
        indices = [0, int(np.argmin(np.abs(t - (t[0] + t[-1]) / 2))), len(t) - 1]
        colors = ["#007F86", "#B24727", "#65499C"]
        labels = [f"t = {t[i] / 60:.2f} min" for i in indices]
        output = source.parent / ("plots_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        output.mkdir(exist_ok=False)

        def save(fig, filename):
            for ax in fig.axes:
                ax.grid(alpha=0.2)
                ax.legend(fontsize=9)
            fig.savefig(output / filename, dpi=180)
            plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), layout="constrained")
        for ax, field, title in zip(axes, ["c_e", "c_s_surf_n", "c_s_surf_p"],
                                   ["Electrolyte", "Negative particle surface", "Positive particle surface"]):
            values, (x, _) = read_field(handle, field, ["x", "t"], "mol/m3")
            for i, color, label in zip(indices, colors, labels):
                ax.plot(x * 1e6, values[:, i], color=color, label=label)
            ax.set(title=title, xlabel="Through-cell position x [um]", ylabel="Concentration [mol/m3]")
        thicknesses = metadata["layer_thicknesses_m"]
        boundary = thicknesses["Negative electrode"]
        for x_boundary in (boundary, boundary + thicknesses["Separator"]):
            axes[0].axvline(x_boundary * 1e6, color="gray", linestyle="--", linewidth=1)
        fig.suptitle("Concentration snapshots | Native HDF5 samples")
        save(fig, "concentration_profiles.png")

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), layout="constrained")
        radial_locations = {}
        for ax, suffix, title in zip(axes, ["n", "p"], ["Negative", "Positive"]):
            values, (r, x, _) = read_field(handle, "c_s_" + suffix, ["r", "x", "t"], "mol/m3")
            location = int(np.argmin(np.abs(x - (x[0] + x[-1]) / 2)))
            radial_locations[suffix] = float(x[location])
            for i, color, label in zip(indices, colors, labels):
                ax.plot(r * 1e6, values[:, location, i], color=color, label=label)
            ax.set(title=f"{title} particle at x = {x[location]*1e6:.2f} um",
                   xlabel="Particle radius r [um]", ylabel="Concentration [mol/m3]")
        fig.suptitle("Radial profiles | Cell centers, excluding exact particle boundaries")
        save(fig, "particle_profiles.png")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for ax, suffix, title in zip(axes[0], ["n", "p"], ["Negative", "Positive"]):
            values, (x, _) = read_field(handle, "j_" + suffix, ["x", "t"], "A/m2")
            for i, color, label in zip(indices, colors, labels):
                ax.plot(x * 1e6, values[:, i], color=color, label=label)
            ax.set(title=f"{title} reaction current", xlabel="x [um]", ylabel="j [A/m2 active area]")
        for suffix, linestyle in [("n", "-"), ("p", "--")]:
            values, (x, _) = read_field(handle, "i_s_" + suffix, ["x", "t"], "A/m2")
            for i, color, label in zip(indices, colors, labels):
                axes[1, 0].plot(x * 1e6, values[:, i], color=color, linestyle=linestyle,
                                label=f"{suffix}: {label}")
        values, (x, _) = read_field(handle, "i_e", ["x", "t"], "A/m2")
        for i, color, label in zip(indices, colors, labels):
            axes[1, 1].plot(x * 1e6, values[:, i], color=color, label=label)
        axes[1, 0].set(title="Solid current (separate electrodes)", xlabel="x [um]", ylabel="i_s [A/m2 geometric area]")
        axes[1, 1].set(title="Electrolyte current", xlabel="x [um]", ylabel="i_e [A/m2 geometric area]")
        fig.suptitle("Current profiles | Native nodes and edges")
        save(fig, "current_profiles.png")
    manifest = {"source_hdf5": str(source.resolve()), "snapshot_indices": indices,
                "snapshot_times_s": [float(t[i]) for i in indices],
                "radial_profile_electrode_locations_m": radial_locations,
                "figures": ["concentration_profiles.png", "particle_profiles.png", "current_profiles.png"],
                "scope": "Visual inspection of stored samples; curves connect native points. "
                         "No new simulation or physical accuracy certification."}
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Created 3 figures from HDF5. No PyBaMM simulation was run.")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
