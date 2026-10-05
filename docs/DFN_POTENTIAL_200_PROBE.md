# Fixed 200-step potential-amplitude comparison

## Protocol and verification

Command: `python scripts/probe_potential_200.py`.
Run: `results/potential_scale_200_20260928T192014760285Z`.
Status: BOUNDED_200_COMPLETE_NOT_VALIDATED. Recorded execution: 72.57 s.

Two fresh C0 arms use the previous paired initial tensors and sample arrays,
seed 42, direct kinetics, learned reaction currents, unchanged loss weights,
float64 CPU, one thread, 200 Adam steps at learning rate 0.001, no L-BFGS.
Only the solid-potential output amplitudes differ. Training and held-out sample
sets are unchanged. No reference labels enter the loss. The previously examined
held-out set is a diagnostic set, not an untouched final test set.

Each arm stores checkpoints at steps 0, 20, 50, 100 and 200. Checkpoints include
optimizer moments, model tensors, amplitude metadata, settings and RNG state.
All 34 normalized residuals and current diagnostics are evaluated on both
sample sets at each checkpoint. Saved metrics and optimizer state replay exactly.
The first 20 loss/gradient histories, final tensors and metrics exactly match
the historical short probe. A disposable next-step check verifies optimizer
continuation equivalence at step 200; no step-201 checkpoint is accepted.
Prior source/artifact hashes are verified before and after the experiment.
No historical run was overwritten. No full physical acceptance audit ran.

## Final held-out results

Residuals are normalized RMS, not physical-unit errors. Total is summed MSE.

| Metric | Original | Ohmic |
| --- | ---: | ---: |
| Total MSE | 19.174062 | 16.545547 |
| Positive collector solid current | 0.999639 | 0.426217 |
| Positive kinetics | 0.814105 | 0.286454 |
| Negative solid charge | 0.081882 | 0.105240 |
| Positive solid charge | 2.259575 | 1.924571 |
| Positive electrolyte charge | 2.260664 | 2.257146 |
| Positive solid separator current | 0.000703 | 0.434904 |
| Positive particle surface flux | 0.988271 | 0.988535 |
| Positive particle inventory | 7.08532e-5 | 6.80238e-5 |
| Positive region total-current diagnostic | 1.001301 | 0.492617 |

The ohmic collector residual progresses 1.053695, 0.952757, 0.815238,
0.426217 at steps 20, 50, 100, 200. Its separator residual simultaneously
progresses 0.018897, 0.073698, 0.191062, 0.434904. Its kinetics RMS reaches
0.119733 at step 50, then rises to 0.286454 by step 200. Improvement is not
uniform or monotonic across physical requirements.

## Interpretation

Unlike the 20-step snapshot, the scaled representation now improves collector
current, kinetics, positive solid charge and total loss relative to control.
However, it allows substantial solid current at the separator, where the
boundary condition requires zero. The experiment shows a tradeoff, not proof
that one error causes another. Both variants retain large charge and particle
flux residuals. Neither is a validated DFN solution, and lower summed loss does
not establish physical acceptance or accuracy against the reference.

Do not extend the budget or start seed sweeps automatically. The next bounded
design question is whether the solid-potential representation can satisfy the
collector and separator current conditions together while retaining the needed
potential offset and curvature. Any hard-boundary representation would be a
separate intervention, not part of this amplitude-only experiment. Check it
analytically before training, preserve these two controls, and keep kinetics,
weights and sampling fixed if a later paired probe is authorized.
