# Frozen coupling scales and branch sensitivity

Command: `python scripts/diagnose_boundary_scales.py`.
Report: `results/boundary_scale_gradients_20260929T070440646816Z/report.json`.
Input is the step-200 boundary checkpoint. No optimization, simulation or
weight change ran. Source/artifact hashes and both saved metric sets replayed;
model tensors and artifacts remained unchanged. All 34 term gradients were
evaluated on training and held-out diagnostic points and their sum checked
against the total gradient. Existing tested gradient helpers were reused.

## Physical normalization

Charge residual divisor: 281744.744 A/m3. Electrolyte mass divisor:
0.277778 mol/m3/s. These physical units differ; their numerical magnitudes
must not be compared as though they were interchangeable weights.

| Particle quantity | Negative | Positive |
| --- | ---: | ---: |
| Diffusion PDE divisor [mol/m3/s] | 9.203611 | 17.528889 |
| Flux/kinetics divisor [A/m2 active] | 1.488247 | 1.685021 |
| Molar flux divisor [mol/m2/s] | 1.54246e-5 | 1.74640e-5 |
| Coefficient multiplying normalized radial derivative in flux | 12.096600 | 2.768868 |
| Diffusion number D*t_ref/R^2 | 3.459563 | 0.528471 |

Both particle raw output amplitudes are 0.01. Their initial-state construction
multiplies the correction by physical time/duration, which suppresses it near
zero time. This is a representation fact, not a demonstrated sole failure cause.
The reported scale formulas are consistent with the implemented operators;
this diagnostic is not an independent certification of all dimensional physics.

## Gradient evidence

Euclidean norms below refer to a specified branch, not the whole network.

| MSE term and affected branch | Training | Held-out |
| --- | ---: | ---: |
| Positive particle diffusion -> positive concentration | 0.121698 | 0.367492 |
| Positive particle surface flux -> positive concentration | 0.004239 | 0.003697 |
| Positive particle surface flux -> positive reaction current | 0.047046 | 0.046813 |
| Negative kinetics -> electrolyte potential | 0.012310 | 0.012363 |
| Negative kinetics -> reaction current | 0.054976 | 0.054994 |
| Negative kinetics -> solid potential | 3.70748e-5 | 3.92422e-5 |
| Positive electrolyte charge -> electrolyte potential | 0.048076 | 0.048828 |
| Positive electrolyte charge -> reaction current | 0.245212 | 0.243986 |

The positive flux signal acting on particle concentration is about 29 times
smaller than diffusion on training samples and 99 times smaller on held-out
samples, despite flux MSE around 0.98 and diffusion MSE around 0.007.
These magnitudes depend on parameterization and are not Adam update sizes.
The two concentration gradients are not strongly opposed: cosine is +0.205
on training points and +0.251 on held-out points. Thus a simple claim that
diffusion and flux necessarily conflict is not supported here.

Large coupling losses have nonzero gradients, so the sampled diagnostic does
not indicate an autograd disconnection. On reaction-current branches, several
coupling terms have larger norms than on their associated field branches;
this does not prove that Adam will reduce reaction current or identify which
branch dominates the actual optimizer trajectory.

Positive solid-charge versus kinetics gradients remain opposed on the positive
solid-potential branch, with cosine -0.840 (training) and -0.340 (held-out).
Negative particle diffusion/flux cosine changes sign between sample sets.
Avoid generalizing one sampled angle into a universal optimization mechanism.

## Decision

Do not multiply loss weights by the observed norm ratios: one snapshot and
one parameterization do not justify that rule. Do not call the DFN validated.

A bounded next experiment should isolate the particle coupling: freeze the
saved reaction-current function and inspect whether the concentration branch
can satisfy that prescribed flux together with diffusion and initial inventory.
This would be a separate diagnostic subproblem, not a coupled-DFN solution or
a return to the closed historical synthetic pilot. Fix its budget and report
both diffusion and flux on independent points before changing the full model.
No such training was launched in this step; the electrolyte and kinetics
failures remain unresolved and must not be hidden by an isolated success.
