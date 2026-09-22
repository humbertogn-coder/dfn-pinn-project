# Constant-Flux Candidate Comparison

Run `python scripts/compare_flux_candidates.py` from the project root.
This step does not train or change checkpoints. It selects the latest completed
baseline/legacy, startup/legacy, startup/full_radius and startup/surface_layer
runs. Explicit `--baseline`, `--legacy`, `--full-radius` and `--surface-layer`
directory arguments pin runs for reproduction. Different seeds, optimizer
budgets or loss weights are rejected rather than silently compared.

The script reloads checkpoints, reproduces saved predictions and all trainer
metrics, and recomputes physical-volume/time PDE RMS at two quadrature orders.
Reports include source paths, configurations and model/report SHA256 hashes.

Six minimized objectives are initial error, positive-time maximum and RMS
concentration errors, maximum flux residual, volume-mean balance error, and
physical-volume/time PDE RMS. A candidate is dominated only when another is
no worse in every objective and strictly better in at least one. This is an
exploratory numerical comparison, not a significance test or acceptance gate.
No post-hoc pass thresholds or aggregate score are introduced.

## Observed Seed-42 Results

Verified report: `results/flux_candidate_comparison_20260920T063804935514Z/report.json`.
All archived trainer metrics reproduced. Eight focused software tests passed.

| Candidate | Max concentration error | Mean balance error | Physical PDE RMS | Nondominated |
| --- | ---: | ---: | ---: | --- |
| baseline | 2.687363e-2 | 1.598407e-4 | 3.340682e-2 | yes |
| legacy | 1.864060e-3 | 4.901422e-4 | 3.511385e-2 | yes |
| full_radius | 2.683769e-3 | 3.912067e-4 | 3.064931e-2 | yes |
| surface_layer | 2.526867e-3 | 6.894379e-4 | 3.949125e-2 | no |

Startup/legacy dominates startup/surface_layer on the six selected objectives.
This does not negate the latter's improvement in the localized residual peak,
which is not a selected objective. Baseline's soft initial condition has an
error about 0.0303, despite its favorable mass metric; nondominance alone does
not establish suitability. Prioritize matched-seed legacy/full_radius tests,
retaining baseline and surface_layer evidence rather than deleting either.

The integration window is dimensionless time [1e-5, 0.02]. Grid errors and
quadrature checks are not continuous-domain bounds. One seed cannot establish
optimizer robustness. Different concentration and derivative objectives may
favor different candidates; nondominated does not mean adequate for DFN use.

Next decision: set application-motivated concentration, flux and mass budgets
before new experiments. Test retained startup designs across matched seeds
with fixed architecture, sampling budgets and optimizer settings. Keep negative
findings. Do not proceed to DFN coupling solely because one local residual fell.
