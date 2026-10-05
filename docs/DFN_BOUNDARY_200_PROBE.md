# Matched 200-step solid-boundary experiment

## Protocol and checks

Command: `python scripts/probe_boundary_200.py`.
Run: `results/boundary_200_20260928T201841393914Z`.
Status: BOUNDARY_200_COMPLETE_NOT_VALIDATED. Recorded execution: 54.05 s,
excluding imports, verification of prior controls and initial preparation.

The experimental boundary representation uses exactly the saved raw initial
weights and samples from the original/ohmic comparison. Parameter keys are
mapped explicitly for the solid raw-network adapter, with equality checked.
Only the previously defined representation intervention differs; kinetics,
other branches, loss weights, physical settings and learning rate are unchanged.
This is not an isolated amplitude-only comparison and initial physical fields
need not match when raw weights match.

Budget: 200 Adam steps at 0.001, seed 42, float64 CPU, one thread, 128 interior
samples per region, 64 boundary times and inventory quadrature order 32.
No L-BFGS, sweep or extension. The held-out diagnostic sample set was reused
from earlier experiments and is not an untouched final test set.

Before comparison, source/checkpoint hashes and final metrics of both controls
were verified from disk. No control training was repeated. New checkpoints at
0, 20, 50, 100 and 200 include Adam state, RNG state and explicit representation
metadata. Metrics and optimizer state replay exactly at all milestones; a
disposable continuation step also matches. No step 201 is accepted or saved.
All prior artifact hashes remained unchanged. The new experiment uses its own
checkpoint schema with strict architecture restoration delegated to the tested
boundary loader. Historical full-run factories are not modified.

## Final held-out comparison

All residuals are normalized RMS; total is the sum of 34 term MSEs.

| Metric | Original | Ohmic | Hard solid boundaries |
| --- | ---: | ---: | ---: |
| Total MSE | 19.174062 | 16.545547 | 12.4425 |
| Positive collector solid current | 0.999639 | 0.426217 | 3.40e-17 |
| Positive solid separator current | 0.000703 | 0.434904 | 0 |
| Negative solid charge | 0.081882 | 0.105240 | 0.045319 |
| Positive solid charge | 2.259575 | 1.924571 | 0.074816 |
| Positive kinetics | 0.814105 | 0.286454 | 0.226762 |
| Positive electrolyte charge | 2.260664 | 2.257146 | 2.257477 |
| Positive particle surface flux | 0.988271 | 0.988535 | 0.990039 |
| Positive particle inventory | 7.08532e-5 | 6.80238e-5 | 6.76035e-5 |
| Positive region total-current diagnostic | 1.001301 | 0.492617 | 0.592143 |

The hard-boundary residuals are satisfied by construction, not learned.
Their maximum sampled positive-collector error at the final checkpoint is
2.220446e-16; separator errors and negative gauge are zero.

Positive solid-charge RMS decreases from 0.138089 at step 20 to 0.074816 at
step 200. Negative solid-charge RMS is not monotonic: 0.034983, 0.020679,
0.032957, 0.045319 at steps 20, 50, 100, 200. Negative kinetics remains
0.979858 and negative particle flux 0.978346 at the final checkpoint.

## Interpretation and decision

The experiment fixes the previously observed separator-current failure and
improves learned solid-charge residuals, especially in the positive electrode.
However, electrolyte charge and particle surface flux remain large. Interior
total current remains incorrect and is worse than the ohmic control in the
positive region, despite much smaller collector errors. Correct endpoints
therefore must not be presented as conservation of total current everywhere.

No reference-field comparison or full physical acceptance audit ran, and no
variant is validated. Lower summed loss is not sufficient evidence of accuracy.

Keep this candidate and the two controls frozen. The next bounded question is
where the remaining coupling fails: evaluate spatial profiles of solid and
electrolyte current, reaction sources and particle surface flux from the saved
checkpoint. Check signs and physical units before proposing another change.
Do not automatically increase budget or add hard electrolyte/particle constraints.
