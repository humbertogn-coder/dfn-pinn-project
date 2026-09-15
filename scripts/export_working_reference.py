"""Export a versioned x80/r320 working-reference bundle with exact readback."""

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from export_internal_fields import write_reference, verify_reference
from inspect_internal_variables import collect_inventory


def main():
    now = datetime.now(timezone.utc)
    output = Path(__file__).resolve().parents[1] / "results" / (
        "working_reference_v1_" + now.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    manifest = {
        "bundle_version": 1, "created_utc": now.isoformat(),
        "status": "working_reference_not_exact",
        "selection_document": "docs/WORKING_REFERENCE.md",
        "files": [],
        "usage": "Independent simulations with common initial state and model settings. "
                 "Keep files separate; do not concatenate overlapping times or randomly split "
                 "overlapping samples between training and validation. Fine startup samples "
                 "are intended for transient inspection, not disproportionate weighting. "
                 "This is a working dataset, not a continuous-time or all-field accuracy certificate.",
    }
    for label, period, duration in (("full", 10, None), ("startup", 0.05, 10),
                                    ("fine_startup", 0.0005, 0.2)):
        print(f"Exporting working reference: {label}...", flush=True)
        solution, report = collect_inventory(radial_points=320, output_period_s=period,
                                              duration_s=duration)
        report.update({
            "reference_role": "working_reference_v1", "segment": label,
            "h5py_version": h5py.__version__, "numpy_version": np.__version__,
            "selection_document": "docs/WORKING_REFERENCE.md",
            "scope": "Full native fields at x80/r320. Native r,x,t or x,t axes and units "
                     "are preserved without resampling. Supported by sampled refinement "
                     "studies; not exact or certified at all points. No field normalization. "
                     "Each segment is an independent simulation, not a solver continuation.",
        })
        pending = output / f"{label}.pending.h5"
        final = output / f"{label}.h5"
        write_reference(pending, solution, report)
        verify_reference(pending, solution, report)
        with h5py.File(pending, "r+") as handle:
            handle.attrs["validation_status"] = "exact_round_trip_verified"
        pending.rename(final)
        with final.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        manifest["files"].append({"path": final.name, "segment": label,
                                   "sha256": digest, "bytes": final.stat().st_size,
                                   "mesh_points": report["mesh_points"],
                                   "pybamm_version": report["pybamm_version"],
                                   "time_start_s": report["time"]["values"][0],
                                   "time_end_s": report["time"]["values"][-1],
                                   "time_samples": len(report["time"]["values"]),
                                   "output_period_s": period, "fields_verified": len(report["variables"]),
                                   "validation_status": "exact_round_trip_verified"})
        print(f"Verified {label}: {final.stat().st_size/1024**2:.2f} MiB, 23 fields.", flush=True)
        del solution, report
        gc.collect()
    # A completed manifest is only published after all three exports pass readback.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Working reference bundle complete: 3 verified HDF5 files.")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
