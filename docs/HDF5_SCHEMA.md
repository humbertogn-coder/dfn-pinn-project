# Internal Fields HDF5 Schema 1.0

Run `python scripts/export_internal_fields.py` from the project directory.
Each run creates a new directory under `results/`. The file is first written
as `reference.pending.h5` and renamed to `reference.h5` only after verification.
Compression is lossless gzip; native floating-point values and axis order are
preserved. Generated files remain excluded from Git.

| Path | Content |
|---|---|
| `/metadata_json` | UTF-8 JSON: protocol, versions, solver settings, inventory and coordinates |
| `/time` | Actual output times in seconds, including the cutoff time |
| `/fields/<id>/values` | Full native array for one of the 23 inventoried fields |
| `/fields/<id>/coordinates/x` | Native spatial nodes or edges in meters |
| `/fields/<id>/coordinates/r` | Native radial nodes in meters, where applicable |
| `/terminal/voltage` | Terminal voltage in volts |
| `/terminal/current` | Applied current in amperes |

Each field carries its exact PyBaMM name, unit, role, JSON axis order and domains.
Each spatial coordinate carries its unit, node/edge location and domains.
HDF5 dimension scales explicitly link array axes to coordinates and time.
Particle fields use `(r, x, t)`; other internal fields use `(x, t)`.
Coordinates are per-field to avoid silently merging electrode and electrolyte
grids or nodes and edges. Surface concentration is not the last radial node.

The root `validation_status` becomes `exact_round_trip_verified` only after
reopening and comparing the file against the in-memory simulation. This is
serialization verification, not physical validation. The JSON sidecar summarizes
the verification. No interpolation, scaling or normalization is performed.

Example read:

```python
import h5py

with h5py.File("results/YOUR_RUN/reference.h5", "r") as reference:
    time_s = reference["time"][:]
    concentration = reference["fields/c_s_n/values"]
    final_profile = concentration[:, :, -1]  # radial position, electrode position
    unit = concentration.attrs["unit"]
```
