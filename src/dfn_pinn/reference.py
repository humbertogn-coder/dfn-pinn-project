"""Verified, bounded-memory access to working-reference bundles."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(values, scale):
    """Divide by a fixed physical scale, without fitting dataset extrema."""
    require(np.isfinite(scale) and scale > 0, "Scale must be positive and finite")
    return np.asarray(values) / scale


class ReferenceBundle:
    """Validate checksums and native layouts before exposing time slices."""

    order = ("fine_startup", "startup", "full")

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        manifest = json.loads((self.directory / "manifest.json").read_text())
        require(manifest["bundle_version"] == 1, "Unsupported bundle version")
        require(len(manifest["files"]) == 3, "Expected three segments")
        self.paths, self.times, self.metadata = {}, {}, {}
        baseline = None
        for entry in manifest["files"]:
            segment = entry["segment"]
            require(segment in self.order and segment not in self.paths, "Invalid segment")
            path = (self.directory / entry["path"]).resolve()
            require(path.parent == self.directory, "File must be inside bundle")
            require(path.stat().st_size == entry["bytes"], "File size mismatch")
            with path.open("rb") as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"],
                        "Checksum mismatch")
            with h5py.File(path, "r") as h:
                require(h.attrs["schema_version"] == "1.0", "Unsupported HDF5 schema")
                require(h.attrs["validation_status"] == "exact_round_trip_verified", "Unverified file")
                meta = json.loads(h["metadata_json"].asstr()[()])
                t = h["time"][:]
                require(t.ndim == 1 and len(t) > 1 and np.isfinite(t).all()
                        and (np.diff(t) > 0).all() and t[0] == 0, "Invalid time axis")
                require(h["time"].attrs["unit"] == "s", "Invalid time unit")
                require(np.array_equal(t, meta["time"]["values"]), "Metadata time mismatch")
                require(len(t) == entry["time_samples"] and t[-1] == entry["time_end_s"],
                        "Manifest time mismatch")
                require(meta["segment"] == segment, "Segment metadata mismatch")
                signature = {key: meta[key] for key in (
                    "mesh_points", "pybamm_version", "model", "parameter_set", "initial_soc",
                    "rtol", "atol", "solver", "layer_thicknesses_m")}
                signature["fields"] = []
                require(set(h["fields"]) == {r["id"] for r in meta["variables"]}, "Field mismatch")
                for record in sorted(meta["variables"], key=lambda r: r["id"]):
                    group = h["fields/" + record["id"]]
                    data = group["values"]
                    axes = record["axis_order"]
                    require(axes in (["x", "t"], ["r", "x", "t"]), "Unsupported axes")
                    require(list(data.shape) == record["shape"] and data.shape[-1] == len(t), "Invalid shape")
                    require(json.loads(data.attrs["axis_order_json"]) == axes, "Axis mismatch")
                    require(data.attrs["unit"] == record["unit"], "Unit mismatch")
                    for i, coord in enumerate(record["coordinates"]):
                        ds = group["coordinates/" + coord["axis"]]
                        values = ds[:]
                        require(np.array_equal(values, coord["values"]) and np.isfinite(values).all()
                                and (np.diff(values) > 0).all() and len(values) == data.shape[i], "Invalid coordinates")
                        require(ds.attrs["unit"] == "m", "Invalid coordinate unit")
                        require(data.dims[i].label == axes[i] and len(data.dims[i]) == 1
                                and data.dims[i][0].name == ds.name, "Invalid spatial dimension scale")
                    last = data.dims[data.ndim - 1]
                    require(last.label == "t" and len(last) == 1
                            and last[0].name == "/time", "Invalid time dimension scale")
                    signature["fields"].append({k: record[k] for k in
                                                ("id", "unit", "axis_order", "coordinates", "domains")})
                for name, unit in (("voltage", "V"), ("current", "A")):
                    ds = h["terminal/" + name]
                    require(ds.shape == t.shape and ds.attrs["unit"] == unit
                            and np.isfinite(ds[:]).all(), "Invalid terminal signal")
                if baseline is None:
                    baseline = signature
                require(signature == baseline, "Incompatible segment settings or coordinates")
                self.paths[segment], self.times[segment], self.metadata[segment] = path, t, meta
        require(self.times["fine_startup"][-1] < self.times["startup"][-1]
                < self.times["full"][-1], "Invalid segment coverage")
        self.fields = tuple(r["id"] for r in baseline["fields"])

    def selected_indices(self):
        """Fine startup owns its endpoint; later segments exclude earlier coverage."""
        end = -np.inf
        for segment in self.order:
            t = self.times[segment]
            yield segment, np.flatnonzero(t > end + 1e-10)
            end = t[-1]

    def iter_field(self, field, scale=1.0):
        """Yield (segment, time_s, native spatial slice); never concatenate volumes."""
        require(field in self.fields, "Unknown field")
        for segment, indices in self.selected_indices():
            with h5py.File(self.paths[segment], "r") as h:
                data = h[f"fields/{field}/values"]
                for index in indices:
                    values = data[..., int(index)]
                    require(np.isfinite(values).all(), "Nonfinite field values")
                    yield segment, float(self.times[segment][index]), normalize(values, scale)

    def audit_overlaps(self):
        """Report sampled absolute differences without assuming all-field tolerances."""
        results = []
        for left, right in (("fine_startup", "startup"), ("startup", "full"),
                            ("fine_startup", "full")):
            pairs = []
            for j, t in enumerate(self.times[right]):
                i = int(np.argmin(abs(self.times[left] - t)))
                if abs(self.times[left][i] - t) <= 1e-10:
                    pairs.append((i, j))
            require(pairs, "No common timestamps")
            with h5py.File(self.paths[left], "r") as a, h5py.File(self.paths[right], "r") as b:
                for field in self.fields + ("voltage", "current"):
                    path = f"fields/{field}/values" if field in self.fields else f"terminal/{field}"
                    maximum = 0.0
                    for i, j in pairs:
                        delta = np.abs(a[path][..., i] - b[path][..., j])
                        require(np.isfinite(delta).all(), "Nonfinite overlap")
                        maximum = max(maximum, float(delta.max()))
                    results.append(dict(left=left, right=right, field=field,
                                        common_times=len(pairs), max_abs_difference=maximum,
                                        unit=a[path].attrs["unit"]))
        return results
