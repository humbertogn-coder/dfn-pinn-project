# Matched Direct-Kinetics Hard-Current Attempt

## Frozen Contrast

Control: results/dfn_baseline_full_20260927T023312902451Z, completed and audited
FAIL. Candidate: direct_bv_hard_current with fixed physical projection order
32. This is a C0/F contrast, not the complete planned soft/hard factorial study;
the control has no explicit integrated-current penalty and is not relabeled C.

The candidate uses the same physical settings, initial raw-network parameters
(seed 42), 34 residual terms, weights, training points, learning rate and
optimizer limits. Its only model change is the shared additive reaction-current
projection. Direct Butler-Volmer is unchanged. Applying the constraint changes
the effective admissible current functions and removes uniform-offset freedom;
identical parameter counts do not mean identical effective degrees of freedom.

The trainer verifies exact training-sample equality against the control saved
checkpoint, as well as identical settings/configuration and the control hash.
No reference labels, warm start, tuned weights, new seeds or budget extension.
The objective is a controlled single-seed comparison, not a robustness claim.

## Execution and Identity

```bat
python scripts/train_dfn_projected.py
```

Budget: 2000 Adam steps then up to 200 L-BFGS iterations and at most 250 closure
evaluations, two-hour cap checked at optimizer/closure boundaries. CPU float64,
one thread. Checkpoints every 100 Adam steps, before L-BFGS and after successful
bounded optimizer completion. Failure behavior is the same as the C0 trainer.

The original C0 training script and existing source modules remain unchanged
to preserve their recorded hashes. The separate projected entry point retains
the same training loop and adds explicit variant/control metadata. This small
duplication preserves reproducibility of the already frozen scientific run.

The new variant loader selects the model from explicit metadata, rejects
unknown specifications and mismatched report/checkpoint labels, and checks
the fixed quadrature nodes, weights, active area and target before loading.
Historical models still use DFNSmoke. Both report and checkpoint store the
variant specification. Component audits and the integrated report carry it
and reject evidence from a differently labeled variant.

Fourteen focused tests passed in 8.82 s (variant loading/identity, field
comparison helpers and full-run completion guards); entry-point syntax checks
passed. No full test suite was run.

## Evaluation Policy

Use the existing native PyBaMM field comparison, independent balances and
per-equation PDE audits, with unchanged thresholds and grids. Projection
integrals are checked independently with quadrature orders different from
the projection order; exact training-node integrals alone cannot establish
physical accuracy. Terminal voltage, phase currents, kinetics, particle flux,
local/global lithium and PDE residuals remain separate required criteria.

Report wall time in addition to equal optimizer budgets. Count the added
projection cost; do not infer an efficiency advantage from equal step counts.
Any failed result is retained without a retry or changed acceptance threshold.

## Run

Launched: results/dfn_projected_full_20260927T232900582653Z.
Completed normally: 2000 Adam steps, 200 L-BFGS iterations, 212 evaluations.
Training wall time was 777.37 s (12.96 minutes), excluding independent audits,
versus 524.60 s for C0. The measured ratio is about 1.48, not a controlled
hardware benchmark. Final training loss was 9.467663, and exact saved prediction
replay passed. The control sample arrays were verified exactly equal.

Reports within the candidate directory:

- reference_comparison_20260927T234227231435Z/report.json
- balance_audit_20260927T234402152389Z/report.json
- pde_audit_20260927T234429595142Z/report.json
- combined_audit_20260927T234451471138Z/report.json

Overall: **FULL_ATTEMPT_AUDITED_FAIL**. All 83 metrics are present. Passing
counts are 1/11 field, 15/42 balance and 14/30 PDE/quadrature checks. These
counts include integration checks and are not an accuracy score.

## Matched Findings

| Metric | C0 control | Hard-current candidate | Limit |
| --- | ---: | ---: | ---: |
| Maximum voltage error [V] | 0.1327931 | 0.1206058 | 0.005 |
| Maximum electrolyte concentration error [mol/m3] | 17.58952 | 13.51329 | 10 |
| Negative reaction-current max error [A/m2] | 1.793478 | 0.2679810 | 0.01488247 |
| Positive reaction-current max error [A/m2] | 2.132470 | 0.4670248 | 0.01685021 |
| Electrode source-integral relative error | 1.012005 | 8.881784e-16 | 0.01 |
| Negative total-current relative error | 1.021945 | 0.2023760 | 0.01 |
| Positive total-current relative error | 1.000237 | 0.9394741 | 0.01 |
| Positive collector solid-current residual | 0.9999988 | 0.9999957 | 0.01 |
| Negative normalized particle-flux residual | 0.04544796 | 1.069593 | 0.01 |
| Positive normalized particle-flux residual | 0.01202510 | 1.035333 | 0.01 |
| Negative normalized local inventory error | 1.832586e-5 | 2.568066e-4 | 1e-6 |
| Positive normalized local inventory error | 2.936082e-5 | 1.623443e-4 | 1e-6 |
| Total lithium drift / initial inventory | 6.585932e-6 | 4.388740e-6 | 1e-7 |

The current source integral agrees to floating-point precision under
independent quadrature, but the positive collector solid current remains
nearly zero relative to its imposed target. Projection does not directly
constrain potential derivatives, so this is not a contradiction.

Both particle-flux residuals and local inventory errors worsen substantially.
Nine of ten physical-measure PDE RMS criteria still fail. Positive solid
charge RMS rises from 0.02724 to 2.28624; positive electrolyte charge RMS rises
from 0.02731 to 0.99878. Particle PDE RMS decreases (negative 0.05683 to
0.02719, positive 0.05693 to 0.02763) while its surface flux worsens. Thus the
improved interior residual cannot stand in for correct boundary transport.

All reported 32/64 quadrature gap checks pass; no order-128 escalation was
needed. Sampled maxima and provisional-reference limitations remain. This is
one seed, one one-second protocol and two equally budgeted, but not equally
costly, attempts. It does not establish statistical robustness or a general
benefit/harm of projection, and neither model is ready for parameter inference.

## Next Bounded Decision

Preserve both failed checkpoints. Before another training change, use frozen
residual/gradient diagnostics to understand the unresolved positive solid
current, kinetics and particle flux. A later direct/inverse kinetics contrast
must retain explicit residual normalization and common SI acceptance metrics;
do not combine new weights, longer training and a new kinetic formulation in
one supposedly isolated change. No extra seeds, retry, inverse training,
Grace job, commit or GitHub push occurred after this comparison.
