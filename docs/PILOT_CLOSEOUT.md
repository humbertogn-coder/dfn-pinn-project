# Single-Particle Pilot Closeout

Status: CLOSED as an exploratory synthetic pilot on 2026-09-22, at the project
owner's request. This is not closure of the DFN research project, not full
Phase-2 acceptance, and not an unqualified physics-validation certificate.
No further seeds, training, Grace jobs or DFN coupling are authorized by this
closeout. Read this document before proposing any continuation.

## Scope Completed

- No-flux synthetic particle training and seed sensitivity study.
- Constant-flux reference, startup diagnostics, sampling comparisons and a
  soft mass-penalty trial, including negative and mixed results.
- Discrete inventory projection, derivative/gradient audits and five matched
  projected-versus-unprojected constant-flux seeds.
- Linear-ramp reference q(t)=50*t, mean concentration 1-75*t^2, and one
  full-budget seed-42 projected PINN.
- Frozen-checkpoint ramp audit with nested global/local grids and independent
  inventory quadratures. No training was run for this final audit.

All coordinates, concentrations and fluxes quoted below are dimensionless.
The positive-time validation window is [1e-5, 0.02].

## Constant-Flux Evidence

Evidence: `results/inventory_seed_study_v1/summary.json` and its progress file,
which pins the five run/control pairs and hashes. Five of five seeds passed
the post-hoc exploratory criteria; these were not preregistered independent
confirmatory seeds. Worst errors across seeds:

| Metric | Worst value | Exploratory limit |
| --- | ---: | ---: |
| Initial concentration | 0 | 1e-12 |
| Maximum concentration | 6.653202e-5 | 1e-3 |
| Surface flux | 2.727221e-3 | 1e-2 |
| Mean inventory | 1.277610e-9 | 6e-5 |

Concentration, flux, mass and physical-volume/time PDE RMS improved against
matched full-radius controls in all five pairs. Log-time-grid PDE RMS worsened
in three pairs. Equal optimizer budgets did not mean equal computational cost.

## Ramp Audit

Frozen model: `results/ramp_flux_pinn_20260922T052023662528Z`.
Final audit: `results/ramp_pilot_audit_20260922T055927812143Z/report.json`.
The report includes source artifact hashes and successful prediction replay.
The preceding audit without explicit instantaneous-flux normalization is
retained as development evidence, not substituted for this final report.

Global grids: 51x101, 101x201 and 201x401 (radius x time), plus nested 81x161
local patches near the error and overshoot peaks and a denser surface sweep.
Maximum global concentration errors were 5.479020e-5, 5.479020e-5 and
5.479585e-5. Refined findings:

| Quantity | Value | Location |
| --- | ---: | --- |
| Surface concentration error | 5.479760e-5 | r=1, t=0.001215068 |
| Concentration excess above initial value 1 | 9.905428e-6 | r=1, t=0.00001348719 |
| Maximum flux residual c_r+q | 3.579343e-3 | r=1, t=0.00001 |
| Minimum sampled concentration | 0.8827139625 | Across audited fields |

Independent mean-inventory errors were 1.017963e-12 (128 nodes), 1.014966e-12
(256), and 1.016298e-12 (512). The maximum 256/512 difference was 2.442491e-15.
The global 2048/4096-mode reference-series difference was 1.640050e-10.
These are quadrature/truncation sensitivity checks, not rigorous bounds.

### Retained Physical Limitation

The maximum flux residual is only 0.358% of peak imposed flux one, but that
normalization hides startup behavior. At t=1e-5 the imposed outward flux is
0.0005 and predicted outward flux is -0.003079343: the predicted sign is wrong.
Absolute error is 7.158686 times the instantaneous imposed flux there. The
positive surface concentration overshoot is likewise retained, not clipped.
Thus the ramp is NOT certified to preserve the maximum principle or the local
boundary flux near startup. Exact inventory does not eliminate these errors.

The training report's physical PDE RMS is 7.391384e-3 and its log-time-grid RMS
is 1.154726e-2. The final audit focused on extrema, flux and mass; these PDE
metrics were not reinterpreted as acceptance thresholds. Ramp evidence covers
one seed only. Final ramp extraction is 0.03, not the constant-flux case's 0.06.

## Closure Decision and Limits

The pilot demonstrates a reproducible implementation and a useful inventory
constraint for prescribed-flux synthetic diffusion. It does not establish
arbitrary-protocol generalization, coupled electrochemical accuracy, positivity,
continuous-time bounds, robust ramp training across seeds, inverse-parameter
identifiability, or full DFN performance. Earlier positive times below 1e-5
remain outside this audit. No additional experiments are scheduled.

Final software verification: all 173 repository tests passed, including six
audit tests. Passing tests establish implementation checks, not physical truth.

## Reproduction and Preservation

To repeat only the final audit:

```bat
python scripts/audit_ramp_pilot.py --run results/ramp_flux_pinn_20260922T052023662528Z
```

Code/tests/English documentation belong in Git. Generated results, checkpoints
and HDF5 files are excluded by .gitignore; GitHub pushes do not back them up.
Retain the full referenced run directories and seed-study artifacts in a
separate verified backup. This closeout does not claim a backup was performed.
No files or failed experiments were deleted, no commit/push was performed,
and unrelated untracked files were left untouched.

For future continuation, first review this closeout and the pinned reports.
Any work on the retained startup defect, new seeds, Grace or coupled DFN needs
a new explicit decision from the owner; it is not part of this closed pilot.
