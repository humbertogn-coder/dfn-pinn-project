# Matched solid-potential amplitude probe

## Protocol

Run: `results/solid_potential_probe_20260928T190218279084Z`.
Command: `python scripts/probe_solid_potential_scaling.py`.
Status: BOUNDED_PROBE_COMPLETE_NOT_VALIDATED. Measured probe time: 13.30 s,
excluding imports and initial preparation.

Both arms use C0 direct kinetics with learned reaction currents, without hard
current projection. Fresh seed 42, identical raw parameter tensors, float64 CPU,
one thread, 20 Adam steps at 0.001, no L-BFGS. Each uses 128 interior points per
region, 64 boundary times and inventory quadrature order 32. Training samples
use seed 20260925; held-out same-distribution samples use seed 20260928.
Architecture, samples, weights and physics are unchanged; only the two solid
potential amplitudes differ. Identical raw weights do not imply identical
physical potential fields after amplitude scaling.

The pinned reference identity was verified. No reference labels entered the
loss. All 34 residual terms and diagnostics were recorded initially and finally,
on both sample sets. Saved models explicitly include amplitude metadata and
replay both final evaluations exactly. Probe checkpoints have their own schema;
they are not full-run audit inputs. Sources, input bundle and checkpoints are
hashed. Existing full-run checkpoints were not modified.

## Results

All values below are final normalized RMS residuals on held-out points, except
the total MSE (sum of residual mean squares). Lower is better.

| Metric | Original amplitude | Ohmic amplitude |
| --- | ---: | ---: |
| Total MSE | 324.18048 | 325.68340 |
| Positive collector solid current | 1.000445 | 1.053695 |
| Positive kinetics | 0.991248 | 0.236645 |
| Negative solid charge | 1.324043 | 1.916779 |
| Positive solid charge | 2.285086 | 2.371566 |
| Positive solid separator current | 0.000208 | 0.018897 |
| Negative solid separator current | 0.032093 | 0.010440 |
| Positive particle surface flux | 0.999544 | 0.999544 |

Training total MSE fell from 513.28680 to 322.13568 for the original and from
515.06288 to 323.64877 for the candidate. The candidate collector residual
improved from its own worse initialization (1.106291 to 1.053695 on held-out
points), but remained worse than the control. Reporting only its relative
improvement would hide this distinction.

## Interpretation and next step

Scaling changes early optimization, notably improving positive kinetics while
worsening several charge/current residuals. It is not an overall winner in
this short experiment. Twenty steps and one seed do not establish asymptotic
performance or reject the candidate for all training budgets. The held-out
sample check is not the independent physical acceptance audit.

Do not launch a full-budget rerun on this evidence. Next bounded diagnostic:
inspect the frozen short-run positive-potential gradients for kinetics, charge
and collector terms, and distinguish potential-value sensitivity from spatial
slope sensitivity. Use these findings before selecting any further change;
do not simultaneously change representation, loss weights and kinetics.
