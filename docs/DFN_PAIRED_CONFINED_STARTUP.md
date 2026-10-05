# Fixed paired confined-startup comparison

Command: `python scripts/compare_confined_startup.py`.
Run: `results/paired_confined_startup_20260929T090202400051Z`.
Status: PAIRED_COMPLETE_NOT_VALIDATED. Execution: 88.86 s after preparation.

## Fixed design and verification

Both particle representations use identical fresh raw-network tensors, seed 42,
four inputs and hidden widths (12,12). They inherit the same frozen reaction
currents and other fields from the boundary step-200 DFN checkpoint. The only
intervention is the declared representation: unrestricted sqrt-time versus
surface-confined sqrt-time plus linear-time bulk. Identical raw tensors do not
imply identical physical fields at positive time.

Each arm receives 200 Adam steps at 0.001, float64 CPU, one thread. Only cs.*
parameters train. Diffusion, surface flux, inventory and center symmetry retain
their existing definitions and unit MSE weights. Kinetics is diagnostic only.
No current projection, inventory projection, clipping, loss-weight search,
L-BFGS, seed sweep or budget extension was added.

Training and historical evaluation points are unchanged. A separate evaluation
set uses seed 20260930 (128 points per region, 64 boundary times, positive time
minimum 1e-6 s). It is excluded from optimization and is not a dense acceptance
audit. Concentration ranges and initial values are checked each step on the
17-radius, five-x, ten-time grid, including t=0. Both arms remain within [0,1]
at these samples; this is not a global bound.

Checkpoints at steps 0, 20, 50, 100 and 200 have explicit architecture metadata,
optimizer states and hashes. All three sample-set metrics and optimizer states
replay exactly. Disposable next-step checks reproduce model and Adam states.
Frozen branch tensors and historical artifacts remain unchanged. No accepted
step 201 or updated coupled-DFN checkpoint was produced.

## Final results on the new evaluation set

All values are normalized RMS residuals, not concentration errors against a
reference solution.

| Metric | Unconfined final | Confined initial | Confined final |
| --- | ---: | ---: | ---: |
| Negative diffusion | 0.421714 | 0.0179454 | 0.367018 |
| Positive diffusion | 18.358865 | 0.397140 | 0.387414 |
| Negative surface flux | 0.934231 | 0.962778 | 0.0355546 |
| Positive surface flux | 0.891511 | 0.794167 | 0.0270336 |
| Negative inventory | 9.87874e-5 | 1.00031e-4 | 3.32955e-5 |
| Positive inventory | 9.15557e-5 | 9.91823e-6 | 4.03372e-5 |
| Negative kinetics (not optimized) | 0.979594 | 0.979594 | 0.979596 |
| Positive kinetics (not optimized) | 0.237842 | 0.265997 | 0.437954 |

Final confined maximum absolute flux residuals are 0.0528204 and 0.0602423;
maximum diffusion residuals are 3.67359 and 3.37954. Final inventory maximum
errors are 8.18948e-5 and 7.46015e-5. These sampled results are not acceptable
particle or full DFN accuracy merely because they outperform the other arm.

Final confined sampled concentration ranges are [0.797638,0.800038] and
[0.4,0.404809]. Initial values remain exact and center symmetry is preserved.
The positive flux RMS falls from 0.658246 at step 20 to 0.476672 at 50,
0.107808 at 100 and 0.0270336 at 200, so it did not repeat the earlier stall.

## Decision: partial improvement, no joint acceptance

Confining startup greatly improves flux learning and removes the previous large
bulk-startup residual relative to the unconfined control. However, it does not
deliver the agreed simultaneous solution: negative diffusion worsens relative
to its own initialization, positive inventory worsens about fourfold, and
positive kinetics worsens while excluded from the subproblem objective.
Electrolyte failures cannot improve because those fields remain frozen.

Preserve the confined representation as a candidate, not an accepted solution.
Do not automatically launch full DFN training or another local representation
variant. Close this representation comparison and review the training strategy
with the owner. The next decision should explicitly address the balance among
diffusion, flux, inventory and kinetic coupling, with fixed physical acceptance
criteria and a bounded experiment. A lower aggregate loss or one better flux
metric must not replace that decision. The previous fixed-budget study remains
evidence, not a reason for unlimited extra optimization steps.
