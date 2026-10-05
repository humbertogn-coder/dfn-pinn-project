# Joint direct/inverse kinetic feasibility results

## Execution

Command: `python scripts/train_joint_kinetics_feasibility.py`.
Run: `results/joint_kinetics_feasibility_20260929T200959116945Z`.
Status: JOINT_PAIR_COMPLETE_NOT_VALIDATED. Paired execution: 101.27 s after
preparation, source checks and initial replay. Both arms completed 200 Adam
steps, no L-BFGS or budget extension.

This implements the pinned joint feasibility specification. The source config
retains its historical specification-only text because its hash is part of
this run; this report records actual implementation and execution. Historical
source modules and checkpoints were not altered.

Both arms start from the same confined-particle checkpoint, with fresh Adam
states at 0.001. ALL neural parameters are trainable, including electrolyte
fields and learned reaction currents. Raw initial tensors and common physical
metrics match exactly. Only the two kinetic loss expressions differ. The
architecture, 34-term inventory, unit weights, physical settings and training
points are identical. No additional projection or clipping is introduced.

The inverse implementation also computes direct kinetics as a finite common
physics guard; consequently a direct-BV overflow would stop that arm too.
No overflow occurred. This implementation does not claim extreme-range
advantages of inverse evaluation. The configured explicit constitutive mode
is unchanged.

## Verification

Four focused tests passed in 8.69 s. They check that only kinetic residuals
change, common direct-form metrics agree, gradients reach every branch, all
parameters are trainable and inconsistent checkpoint metadata is rejected.

Milestones 0/20/50/100/200 preserve optimizer state and explicit model identity.
Common metrics on all four sample sets replay exactly after loading, as do
optimizer states. Disposable next-step continuation matches model and Adam
tensors. Source artifact hashes remain unchanged. Domain checks run every
step; sampled concentration ranges remain physical and initial profiles and
hard solid-boundary identities are maintained. These are sampled checks,
not continuous-domain guarantees.

The additional evaluation set uses seed 20261001 and never enters training.
Other sets are historical diagnostics. Kinetic metrics below always use
`(j-direct_BV(eta,j0,T))/j_ref`, even for the inverse-trained arm. Raw training
losses from different kinetic formulations are not ranked as equivalent.

## Common final metrics

Normalized RMS values on the new evaluation set:

| Metric | Shared start | Joint direct | Joint inverse |
| --- | ---: | ---: | ---: |
| Negative particle diffusion | 0.494867 | 0.255222 | 0.083604 |
| Positive particle diffusion | 0.169657 | 0.110298 | 0.089073 |
| Negative particle flux | 0.037272 | 0.035317 | 0.026674 |
| Positive particle flux | 0.025438 | 0.005001 | 0.005074 |
| Negative particle inventory | 3.33331e-5 | 1.67145e-5 | 9.02035e-6 |
| Positive particle inventory | 3.72110e-5 | 2.76920e-5 | 2.25494e-5 |
| Negative kinetic consistency | 0.979706 | 0.946522 | 0.933677 |
| Positive kinetic consistency | 0.386945 | 0.230324 | 0.232387 |
| Negative solid charge | 0.045933 | 0.132102 | 0.133298 |
| Positive solid charge | 0.073097 | 0.125313 | 0.119158 |
| Negative electrolyte charge | 1.948498 | 1.691738 | 1.785732 |
| Positive electrolyte charge | 2.257333 | 1.995619 | 1.988723 |
| Negative interior total-current error | 0.567860 | 0.526055 | 0.557118 |
| Positive interior total-current error | 0.543867 | 0.491605 | 0.490287 |

The inverse arm improves particle diffusion and inventory more, but has worse
negative electrolyte charge and negative interior total current than direct.
Both worsen solid-charge residuals relative to their shared starting state.
No universal winner or robustness claim follows from one short warm-start pair.

## Why neither passes

Original thresholds are unchanged. Representative inverse-arm sampled failures:

- Negative/positive particle PDE RMS: 0.0836/0.0891 versus limit 0.01;
  maxima 0.6869/0.7483 versus limit 0.1.
- Negative/positive flux maxima: 0.06014/0.01169 versus limit 0.01.
  Positive flux RMS below 0.01 does not satisfy the maximum-error criterion.
- Local inventory maxima: 2.59736e-5/5.21579e-5 versus limit 1e-6.
- Negative kinetic maximum: 0.93638 versus limit 0.01.
- Interior total-current maxima: 0.94729/0.85629 versus limit 0.01.
- Electrolyte charge RMS remains near 1.79/1.99, far above 0.01.

The direct arm also fails multiple common criteria. Full reference-field,
independent quadrature and global lithium audits have NOT run for these new
variants. Full validation is therefore incomplete, with already demonstrated
sampled criterion failures; it must never be reported as PASS.

## Close this feasibility pair

This was an actual coupled-DFN training attempt, not another frozen-particle
test. It yields a characterized partial improvement, not a validated solution.
Do not automatically extend the budget, launch Grace, repeat seeds or add
another architecture variant. Preserve both outcomes.

The next strategy decision must address persistent electrolyte transport,
kinetic consistency and conservation together. Any residual normalization or
constraint-weighting change should be specified as a separate controlled
intervention using physical scales/acceptance goals, not tuned retrospectively
until one selected metric looks good. This experiment alone does not establish
that reweighting will fix the failures. Review that decision with the owner
before another training campaign; the original C-F factorial is still pending.
