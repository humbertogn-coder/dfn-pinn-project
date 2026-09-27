# Bounded DFN Training Dry Run

Completed September 26, 2026 local time (September 27 UTC artifact timestamp).
This is a technical feasibility check, not a converged DFN or acceptance audit.

## Protocol

Fresh seed-42 initialization, unchanged DFNSmoke architecture and direct
kinetics, float64 CPU, one thread. Fixed baseline samples use 128 interior
points per region, 64 boundary times, and inventory quadrature order 32.
Twenty Adam steps at learning rate 0.001; no L-BFGS or reference labels.
Loss weights and all 34 residual terms remain unchanged. A 600-second cap is
checked at step boundaries, not an interrupt within a single optimizer step.

```bat
python scripts/run_dfn_dry_run.py
```

The new dfn_run_v1 record separates physical identity from known training and
architecture metadata. Unknown settings remain part of physical identity.
Only the bounded dry-run completion contract is implemented: it cannot issue
full-budget completion, even after all 20 requested steps. The historical
audit runners are unchanged and must not be used on this new run by bypassing
their settings checks. Full-run completion and audit loading remain to be
connected explicitly when implementing the full trainer.

## Evidence

Run: results/dfn_dry_run_20260927T021932378596Z.

- Five focused identity/completion tests passed in 6.00 s.
- All 20 Adam steps completed with finite, nonzero gradient norms in 12 branches.
- Loss before the first update: 513.2868; before the twentieth: 330.0773.
  These are training losses, not independent errors or post-update scores.
- Median Adam step: approximately 0.277 s.
- Training/replay section wall time: approximately 6.49 s, excluding setup.
- Exact frozen prediction replay passed after saving and loading checkpoint.pt.
- Two disposable model/optimizer copies produced exactly equal next-step
  diagnostics and model tensors after restoring Adam state. These two replay
  updates do not advance the saved 20-step checkpoint.
- The saved checkpoint hash remained unchanged during verification.

The report stores per-term losses, per-branch total-loss gradient norms,
step timing, source/config/reference hashes and completion metadata. Gradient
norms here are NOT per-equation gradient attribution or Hessian diagnostics.
On error, a failed report is retained rather than claiming completion.

## Cost and Next Step

The median-based estimate for 2000 Adam steps is 553.86 s (about 9.2 minutes).
This excludes L-BFGS, audits, I/O and runtime variability. It is not a full
experiment time prediction or an argument that the PINN is faster than PyBaMM.
No Grace job or larger campaign is needed on the basis of this dry run alone.

Next: implement the single fixed-budget trainer and its versioned completion
evidence, connect physical reference compatibility to the existing auditors,
then run the agreed 2000-Adam/200-L-BFGS attempt under the fixed wall cap.
Do not warm-start from this dry-run checkpoint or silently extend the budget.
No full-budget run, independent accuracy audit of this model, commit or push
was performed in this step. Generated artifacts require separate backup.
