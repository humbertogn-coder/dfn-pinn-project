# Frozen DFN Reference Comparison

Status: SMOKE_DIAGNOSTIC_ONLY. No training was run.

Checkpoint: results/dfn_smoke_20260924T054745712381Z/checkpoint.pt.
Reference: results/dfn_smoke_reference_20260924T195608554436Z/reference.h5.
Report: reference_comparison_20260924T211656586255Z/report.json within the checkpoint directory.

The comparison verifies both source manifests, the reference HDF5 hash,
matching settings, and exact checkpoint replay. Predictions are evaluated on
the native x80_r320 coordinates and all saved times without reference-field
interpolation. Electrolyte branches are selected by physical region. Particle
inputs use r/R, x/L, t/t_ref; potentials use the shared physical voltage scale.
Terminal voltage is positive collector potential minus negative collector
potential. No fitted potential shift is applied.

## Exploratory Screening Criteria

These limits are fixed before future training, but are not preregistered for
the existing smoke checkpoint. They are engineering screening targets, not
publication-level or continuous-time error guarantees:

- Voltage and each potential: maximum absolute error <= 5 mV.
- Electrolyte concentration: maximum absolute error <= 10 mol/m3.
- Particle and surface concentrations: maximum error / c_max <= 0.001.
- Reaction current: maximum error / electrode j_ref <= 0.01.

The two reference meshes differ by up to 0.691 mV. Internal-field reference
uncertainty has not been certified for this setup. Passing this screen alone
cannot establish acceptance: independent PDE, boundary/interface, current and
inventory checks remain required before a physical acceptance decision. Their
normalizations and limits must be fixed before a full-budget training run.

## Observed Results

| Field | Maximum absolute error | Unit |
| --- | ---: | --- |
| Terminal voltage | 0.1365482 | V |
| Electrolyte concentration | 17.42876 | mol/m3 |
| Electrolyte potential | 0.1122558 | V |
| Negative particle concentration | 132.9568 | mol/m3 |
| Positive particle concentration | 293.6465 | mol/m3 |
| Negative surface concentration | 138.0359 | mol/m3 |
| Positive surface concentration | 329.2162 | mol/m3 |
| Negative reaction current | 0.3686045 | A/m2 active area |
| Positive reaction current | 0.4233218 | A/m2 active area |

Only the negative solid potential passes its field-error screen; all other
screens fail. This is a three-Adam-step connectivity checkpoint, not a trained
DFN solution. The result does not justify changing thresholds or claiming
full-model validation. Earlier particle limitations remain documented.

Run again only when needed:

```bat
python scripts/compare_dfn_reference.py
```

Code, focused tests and this summary are versioned. Generated HDF5, checkpoints
and detailed reports remain excluded from Git by project policy. A GitHub push
does not back them up; preserve the referenced result directories separately.
