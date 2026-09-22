# Paired Constant-Flux Particle Study

Run `python scripts/check_flux_particle_seeds.py` from the project root.
This performs ten fresh sequential CPU trainings: seeds 0, 1, 2, 3 and 42,
each with startup/legacy and startup/full_radius. Training remains float64,
512 interior points, 2000 Adam steps and 300 L-BFGS maximum iterations.
Matching initial-weight hashes are required within each pair.

The timestamped results directory contains a log and full artifacts for each
run, an incrementally updated summary.json, and a final runs.csv. Existing
checkpoints are neither overwritten nor silently reused. A failed run is
reported and excluded from descriptive statistics; paired differences use
only complete pairs. Failure produces a nonzero study exit code. The timeout
is 30 minutes per trainer. Interrupted studies retain completed artifacts but
automatic resume is not implemented; another invocation starts a new study.

Each checkpoint's saved predictions and training points are verified using
the existing loader. Physical-volume/time PDE RMS uses identical 64/128 radial
quadratures and 4/8 time points in each of 16 panels. The quadrature difference
is recorded; it is not a rigorous bound. Other metrics use the trainer's fixed
analytic evaluation grid. Differences are full_radius minus legacy, so lower
is better and negative favors full_radius. All six candidate objectives remain
visible; no aggregate score or best-seed selection is used.

This is exploratory robustness evidence after seed-42 sampler development,
not an independent confirmatory test or evidence of DFN accuracy. Seed 42 is
included for continuity, not counted as previously unseen evidence. Both
initialization and collocation change with seed. Five pairs do not establish
statistical significance. Acceptance budgets still require a separate decision.

## Executed Study

Evidence: `results/flux_particle_seeds_20260920T064901803722Z/summary.json`.
Ten of ten runs completed, with five complete pairs and no failures.
The full software suite passed 122 tests. Seed-42 metrics reproduced the
previous runs to the displayed precision.

| Metric | Legacy median | Full-radius median | Legacy worst | Full-radius worst |
| --- | ---: | ---: | ---: | ---: |
| Maximum concentration error | 9.744515e-4 | 9.902478e-4 | 1.864060e-3 | 2.683769e-3 |
| RMS concentration error | 8.914187e-5 | 9.983170e-5 | 1.849001e-4 | 1.656522e-4 |
| Maximum flux residual | 2.013476e-2 | 1.351913e-2 | 3.094076e-2 | 4.558831e-2 |
| Mean balance error | 2.659649e-4 | 2.485368e-4 | 4.901422e-4 | 3.912067e-4 |
| Physical PDE RMS | 3.006443e-2 | 2.946739e-2 | 3.511385e-2 | 3.167303e-2 |

Initial error was zero in every run. Full-radius sampling lowered concentration
RMS in four of five matched pairs, and each other noninitial objective in three
of five. A median of paired differences is not a difference of marginal medians;
the concentration RMS figures illustrate why both are retained.

Interpretation: no uniform winner. Full-radius has favorable mass-balance and
physical-PDE worst cases in this study, but a worse maximum concentration error
and flux residual. Keep both as documented controls. Further changes require
explicit accuracy priorities; do not choose a best seed or infer readiness for
DFN coupling from this small exploratory study alone.
