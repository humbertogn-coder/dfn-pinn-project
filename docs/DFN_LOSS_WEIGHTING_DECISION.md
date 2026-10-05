# Loss weighting review and bounded decision

Execution update (September 30): the specified pair is implemented and complete.
See `DFN_INVENTORY_WEIGHT_RESULTS.md`. Both runs fail physical criteria; the
implementation-pending status below is historical. No further weight trial
is automatically authorized.

## Verified evidence

The joint direct/inverse pair remains physically unvalidated. This review uses
its recorded training losses, not a new model evaluation or training run.
Command: `python scripts/review_joint_loss_weights.py`.
Report: `results/joint_loss_review_20260929T220702054006Z/review.json`.
Historical source hashes, the 34-term inventory, finite nonnegative entries,
and component sums were checked. No broad test suite was run for this
read-only arithmetic review.

These values describe the loss BEFORE update 200, after 199 updates. They are
not the final checkpoint's independent evaluation metrics.

| Loss group | Direct MSE sum | Direct share | Inverse MSE sum | Inverse share |
| --- | ---: | ---: | ---: | ---: |
| Electrolyte charge | 6.918691 | 87.4673% | 7.196178 | 39.3935% |
| Kinetics | 0.9482632 | 11.9881% | 11.02728 | 60.3658% |
| Particle inventory | 1.014864e-9 | 1.28301e-8% | 5.763182e-10 | 3.15490e-9% |
| All other terms | 0.0430781 | 0.544601% | 0.0439681 | 0.240691% |

Small inventory loss does not mean acceptable inventory error: the physical
maximum-error tolerance is 1e-6. Large charge/kinetic losses do not prove their
gradients dominate Adam updates, nor that these equations need more weight.
Direct and inverse kinetic losses have different meanings; their total loss
values cannot rank physical accuracy.

## Selected intervention for the next implementation

Test only inventory weighting, using DIRECT kinetics in both arms. This is
not a declaration that direct kinetics is better. It avoids introducing an
additional conversion from overpotential error to current-error tolerance.
This narrow test addresses an identified conservation weighting mismatch;
it is NOT expected or guaranteed to resolve electrolyte transport or kinetics.

The objective will be:

`L = sum(MSE(R_k), k not inventory) + w * (MSE(R_inventory_n) + MSE(R_inventory_p))`.

Control: w=1. Candidate: w=1e8, fixed before training, no sweep. The ratio
comes from `(0.01 / 1e-6)^2`: a typical normalized PDE RMS scale and the
existing inventory maximum-error tolerance. This is an optimization scale
choice, not a proof of equal gradient influence. An RMS objective cannot
enforce a maximum-error gate; independent maxima remain required. Leave every
other residual normalization and coefficient unchanged. In particular, do
not divide exact center identities by their 1e-12 acceptance tolerance.

Both arms must copy `joint_direct_0000.pt` from
`results/joint_kinetics_feasibility_20260929T200959116945Z`, with identical
training samples and fresh Adam states. Pin its recorded hash, source report,
samples and source code before execution. Do not start from different final
direct/inverse states or compare a continuation against an earlier warm start.
All fields remain trainable. Keep the confined particle and hard solid-boundary
maps, float64 CPU, one thread, Adam lr=0.001, 200 steps per arm, milestones
0/20/50/100/200, paired wall cap 1200 s, and no L-BFGS. No architecture,
current/inventory projection, sampling, clipping or reference-label change.

The control must reproduce the previous direct trajectory within the same
runtime, with exact state/metric checks; a mismatch stops interpretation.
Use identical evaluation sets in both arms, including an additional reserved
set with seed 20261002. Historical evaluation sets are already inspected and
must not be described as untouched. Save raw and weighted losses separately,
optimizer state, explicit weight metadata and verified reload/continuation.

## Decision rules and scope

Evaluate common physical RMS AND maxima for every equation, flux, inventory,
kinetics, interfaces and total current. Keep all original acceptance criteria.
Record sampled concentration bounds and hard identities; stop on invalid or
nonfinite states, without clipping or automatic restart. Improving inventory
alone is not coupled progress. If other physical errors worsen, report the
tradeoff instead of selecting a winner by weighted total loss. No repeated
weight tuning, extra seeds, full-budget extension or Grace job is authorized
by this specification. Independent reference/global audits remain necessary
before any full validation claim.

Status: REVIEW_COMPLETE; WEIGHTED_TRAINER_NOT_IMPLEMENTED; NO_NEW_TRAINING.
The next bounded task is to implement and execute this one paired intervention,
with focused weight/gradient/metadata checks integrated into that task, not a
separate sequence of smoke experiments. Preserve historical hashed modules.
