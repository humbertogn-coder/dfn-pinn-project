"""Audit a working bundle and record physical normalization without simulation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pybamm

from dfn_pinn.reference import ReferenceBundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Completed working-reference directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidates = sorted((root / "results").glob("working_reference_v1_*/manifest.json"))
    if args.input is None and not candidates:
        parser.error("No completed working-reference bundle found")
    bundle = ReferenceBundle(args.input or candidates[-1].parent)
    meta = bundle.metadata["full"]
    if meta["parameter_set"] != "Chen2020" or meta["pybamm_version"] != pybamm.__version__:
        raise ValueError("Physical scales require the exporting PyBaMM version and Chen2020")
    parameters = pybamm.ParameterValues("Chen2020")
    for layer, thickness in meta["layer_thicknesses_m"].items():
        if parameters[f"{layer} thickness [m]"] != thickness:
            raise ValueError("Parameter geometry differs from export")
    scales = {
        "time_s": 3600.0,
        "x_m": sum(meta["layer_thicknesses_m"].values()),
        "r_n_m": float(parameters["Negative particle radius [m]"]),
        "r_p_m": float(parameters["Positive particle radius [m]"]),
        "c_s_n_mol_m3": float(parameters["Maximum concentration in negative electrode [mol.m-3]"]),
        "c_s_p_mol_m3": float(parameters["Maximum concentration in positive electrode [mol.m-3]"]),
        "c_e_mol_m3": float(parameters["Initial concentration in electrolyte [mol.m-3]"]),
    }
    field_scales = {"c_s_n": scales["c_s_n_mol_m3"], "c_s_surf_n": scales["c_s_n_mol_m3"],
                    "c_s_p": scales["c_s_p_mol_m3"], "c_s_surf_p": scales["c_s_p_mol_m3"],
                    "c_e": scales["c_e_mol_m3"]}
    print(f"Verified bundle: {bundle.directory}", flush=True)
    overlaps = bundle.audit_overlaps()
    selected = {s: len(i) for s, i in bundle.selected_indices()}
    times = np.concatenate([bundle.times[s][i] for s, i in bundle.selected_indices()])
    if not (np.diff(times) > 0).all():
        raise ValueError("Duplicate or nonmonotone selected times")
    ranges = {}
    for field, scale in field_scales.items():
        low, high = np.inf, -np.inf
        for _, _, values in bundle.iter_field(field, scale):
            low, high = min(low, float(values.min())), max(high, float(values.max()))
        ranges[field] = {"min": low, "max": high}
        print(f"{field}: normalized range [{low:.6g}, {high:.6g}]", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = bundle.directory / ("preparation_" + stamp)
    output.mkdir()
    report = dict(source_bundle=str(bundle.directory), pybamm_version=pybamm.__version__,
                  physical_scales=scales, field_scales=field_scales, normalized_ranges=ranges,
                  selected_samples=selected, total_times=len(times), overlap_diagnostics=overlaps,
                  status="prepared_for_inspection_not_approved_for_training",
                  limitations="Overlap differences are diagnostics, not an all-field acceptance test. "
                  "Scales are reconstructed from the matching installed Chen2020 version, not "
                  "a complete exported parameter snapshot. Potentials and currents remain dimensional. "
                  "Time scale is fixed at 3600 seconds for 1C, not fitted to cutoff. No random split "
                  "or training weights are created; dense startup must not dominate evaluation.")
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selected times: {selected}; total={len(times)}")
    for row in overlaps:
        if row["field"] in ("voltage", "c_s_surf_n", "c_s_surf_p"):
            print(f"Overlap {row['left']}/{row['right']} {row['field']}: "
                  f"{row['max_abs_difference']:.6e} {row['unit']}")
    print(f"No simulation was run. Report: {output / 'preparation.json'}")


if __name__ == "__main__":
    main()
