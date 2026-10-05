# Frozen-current particle subproblem

Command: `python scripts/probe_frozen_particle_coupling.py`.
Run: `results/frozen_particle_coupling_20260929T070848372472Z`.
Status: PARTICLE_SUBPROBLEM_COMPLETE_NOT_VALIDATED. Execution: 31.25 s.

## Fixed protocol

Warm start from the step-200 hard-solid-boundary DFN checkpoint. Only the two
solid concentration networks are trainable. All reaction-current functions,
electrolyte fields and solid potentials remain frozen. Use a fresh Adam
optimizer at 0.001 for exactly 200 additional particle-only steps, float64 CPU,
one thread. These are additional diagnostic steps, not a budget-matched
comparison against the earlier 200-step experiments.

The objective sums the unchanged normalized MSEs for particle diffusion,
surface flux, inventory and center symmetry in both electrodes (eight terms,
unit weights). Kinetics is excluded from optimization and monitored only.
The full 34-term diagnostic is still reported at milestones. Initial profiles
remain enforced by the existing concentration architecture. No field labels,
new parameterization, loss-weight tuning or reference simulation were added.

Training samples and the historical diagnostic samples are unchanged. A new
same-distribution evaluation set uses seed 20260929, with 128 interior points
per region, 64 boundary times and minimum positive time 1e-6 s. It is never
used by the optimizer; it is not a dense independent acceptance audit.

## Verification

Source checkpoint metrics and source hashes were checked before training.
At steps 0, 20, 50, 100 and 200, all non-particle tensors remain exactly equal
to their original values. Frozen parameters receive no accumulated gradients.
Particle parameters changed and their gradients remained finite. Disk reload
reproduces all metrics on all three sample sets and Adam state exactly.
A disposable next-step comparison verifies continuation equivalence without
accepting a step 201. All original artifact hashes remain unchanged.

The new checkpoint schema explicitly labels the particle-only objective and
trainable prefix. It must not be loaded as an ordinary coupled-DFN training run.

## Results on the new evaluation set

All quantities below are normalized RMS residuals.

| Metric | Before | After 200 particle-only steps |
| --- | ---: | ---: |
| Negative particle diffusion | 0.0213220 | 0.0124668 |
| Positive particle diffusion | 0.0785772 | 0.00938012 |
| Negative surface flux | 0.9783858 | 0.9782663 |
| Positive surface flux | 0.9899699 | 0.9899277 |
| Negative inventory | 9.36212e-5 | 9.23600e-5 |
| Positive inventory | 7.66467e-5 | 6.87275e-5 |
| Negative kinetics, diagnostic only | 0.9799799 | 0.9799799 |
| Positive kinetics, diagnostic only | 0.2345291 | 0.2341831 |
| Both center gradients | 0 | 0 |

Training-set positive diffusion RMS falls from 0.0839094 to 0.00787635,
while its surface-flux RMS remains near 0.9899. The behavior is therefore
not limited to the new evaluation set. Existing electrolyte/charge failures
are unchanged because their fields and reaction sources are frozen.

## Interpretation and next decision

The subproblem did not learn the prescribed surface exchange: diffusion
improved while flux scarcely moved. It is not a successful particle solution,
and neither inventory accuracy nor coupled DFN validation is established.
This result shows that coupling to evolving electrolyte/reaction networks is
not necessary for the observed stall under this particular representation,
loss and fixed budget. It does not prove which of those choices causes it,
nor that a longer run could never improve.

Do not extend the budget automatically. Next examine a startup-aware particle
representation as one isolated candidate, using the existing pilot theory and
preserving the frozen current diagnostic. First check analytically the initial
profile, center symmetry, differentiability and surface-flux sign. Uniform
initial concentration and a suddenly nonzero flux have a corner incompatibility:
do not require the classical initial profile derivative and nonzero boundary
flux to hold simultaneously at exactly t=0. Any correction must be justified
for t>0, keep the physical equations unchanged, and be compared under a fixed
budget before use in the coupled DFN. This does not reopen the closed pilot.
