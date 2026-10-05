# Research Direction v2

## Current Decision (September 29)

September 30 potential diagnosis: see `DFN_CHARGE_KINETIC_OFFSETS.md`.
Frozen kinetics has a negative-electrode potential-level deficit, while
electrolyte spatial slopes remain insufficient for the applied current.
Disposable constant shifts separate the effects but do not solve the DFN.
Next specify a controlled potential level/shape intervention; no new
representation or training has run. This supersedes next-step notes below.

September 30 closeout: the fixed inventory-weight pair is complete; see
`DFN_INVENTORY_WEIGHT_RESULTS.md`. Exact control replay passed. Inventory
improves but negative diffusion and positive surface flux worsen; charge and
kinetics remain unsolved. Close the experiment without another weight search
or automatic training extension. Earlier implementation-pending notes below
are historical.

Latest: the recorded-loss review is complete; see
`DFN_LOSS_WEIGHTING_DECISION.md`. The next specified intervention isolates
particle-inventory weighting in a matched direct-kinetics pair, with unchanged
physical acceptance criteria. No weighted trainer or new run exists yet.
This supersedes the next-step suggestions below, not their recorded evidence.

Update: the joint direct/inverse feasibility pair is now implemented and
complete; see `DFN_JOINT_KINETICS_RESULTS.md`. Both 200-step runs trained all
fields and passed technical replay checks, but fail multiple common physical
criteria. Close this pair without automatic extensions or architecture changes.
The specification status mentioned below is historical, not current execution.

The local potential/particle representation sequence is closed with partial,
unvalidated improvements. See `DFN_PAIRED_CONFINED_STARTUP.md` and
`DFN_JOINT_TRAINING_DECISION.md`. The next bounded implementation is a matched
joint direct/inverse kinetic feasibility pair on the same confined candidate,
with all fields trainable and unchanged common physical acceptance criteria.
The pinned config is specification-only; this experiment has not run. This
supersedes earlier next-step notes below, not historical results or the planned
C-F factorial study. No further local variant or full-budget run is automatic.

Recorded September 25, 2026 after the owner's updated methodology and approval
to proceed. This document supersedes historical next-step suggestions, not
historical results. All closed-pilot limitations and failed trials remain.

## Research Questions

Primary: test whether explicitly learned reaction current, inverse symmetric
Butler-Volmer residuals, and hard electrode-integrated current projection
improve full nonlinear DFN PINN accuracy and training robustness.

Conditional secondary: identify a small, demonstrably identifiable parameter
set using compatible experimental data, only after adequate forward accuracy.
Neither improvement, novelty, speedup nor successful calibration is presumed.

## Existing Assets and Limits

The current DFN already learns j and uses direct kinetics. Its integrated
reaction current is diagnostic, not an explicit global-current soft penalty;
do not silently label it the proposed soft-current variant C. Inverse kinetics
and electrode-current projection primitives exist but are not connected to
the full DFN. Particle inventory projection is a different constraint.

The latest committed baseline is 0cf18ca. Subsequent baseline preparation,
balance audit and PDE audit are development additions. The 1 s matched
reference uses synthetic initial stoichiometries and must not be confused
with the older full-discharge working reference. No validated full-budget
DFN training is available at this milestone.

## Controlled Comparisons

Preserve the existing diagnostic baseline as C0. Version any added soft
electrode-current loss explicitly. The planned C-F factorial comparison is
learned j with direct/inverse kinetics and soft/hard integrated-current
constraints. Keep other initial/boundary constraints, physical parameters,
evaluation grids and acceptance thresholds fixed. Record any changed loss
term and its normalization. Direct and inverse residuals have different units
and effective weighting; equal scalar weights do not make equivalent losses.
Audit physical kinetic consistency using common metrics across variants.

A conventional direct formulation and a faithful Lee bypass implementation
are subsequent comparators, not labels applied to the current architecture.
Reproduction details must be established before implementing that comparator.
Compare multiple matched seeds only after a bounded feasibility trial. Report
wall time, resources and preprocessing/training costs, not optimizer steps
alone. A fixed-parameter trained PINN is not a parameterized surrogate.

Hard projection must use fixed physical quadrature for each queried time,
preserve gradients and signed electrode targets, and feed the same projected
current into all sources, fluxes, kinetics and inventory. Test independence
from query batch/chunk composition before integration. No positivity clipping
or hidden kinetic regularization may be introduced as an unreported change.

## Ordered Backlog

1. DONE: independent physical balance audit of the frozen smoke checkpoint.
2. DONE: local PDE RMS/maxima and quadrature checks; see DFN_PDE_AUDIT.md.
3. DONE: historical and full-run field/balance/PDE integration, versioned
   completion contract and physical reference compatibility separate from
   training settings. Missing evidence never passes. See DFN_FULL_BASELINE.md.
4. DONE: bounded 20-Adam-step cost/gradient/replay dry run, no sweep.
   See DFN_DRY_RUN.md. New physical identity and dry-run completion contract
   implemented; full trainer and auditor loading integration now completed.
5. DONE: one fixed 2000-Adam/200-L-BFGS C0 attempt, completed and audited FAIL.
   See DFN_FULL_BASELINE.md for evidence. A characterized failure is a valid
   result; indefinite baseline convergence is not required before ablations.
6. Controlled C-F feasibility experiments, then matched seeds and longer
   protocols if warranted. Consider Grace only for justified larger jobs.
   The direct-kinetics hard-current building block, technical probe, explicit
   variant loading and one matched full C0/F comparison are complete. Both
   full attempts failed physical criteria; see DFN_PROJECTED_FULL.md. Frozen
   charge/kinetic/flux gradient diagnosis is now complete; see
   DFN_GRADIENT_DIAGNOSIS.md for saturation, scaling and sampling evidence.
7. Synthetic inverse recovery with noise, parameter sensitivities and
   correlations, then held-out experimental protocols if feasible.

Steps 1-5 have evidence, including one failed full-budget C0 attempt. The next
step is a controlled potential-representation/scaling proposal informed by
the frozen diagnosis, not an unbounded retry. Projection fixed source
integrals but not local physics. Keep representation changes distinct from
the planned direct/inverse kinetics comparison.
No multi-seed ablation campaign, sweep or Grace job has been executed.

## Literature and Data Follow-up

Prepare a claim-by-claim literature matrix including Hassanaly (2024), Lee
(2025), PyBaMM inverse kinetics, and DiffLiB (preprint April 2025). Individual
ingredients are not claimed novel. Verify published version dates and exact
reproduction details before making comparative claims.

Inspect metadata and one raw file before large downloads. Prioritize original
LG M50 characterization data, check license, I/V/time/temperature channels,
protocols, initial SOC, SOH and thermal adequacy. M50T is not automatically
interchangeable with M50; synthetic datasets are not experimental validation.
Use only a few identifiable parameters; local sensitivity/Fisher information
does not prove uniqueness. Include nuisance initial-state uncertainty,
synthetic recovery, multistart/uncertainty and held-out profiles. Compare with
PyBaMM optimization; differentiable DFN is an optional further comparator.

## Preservation

Keep prior thresholds, checkpoints, unsuccessful runs and pilot closeout.
Commit code/configuration/documentation after review. Generated results and
checkpoints excluded by Git require a separate verified backup; a GitHub
push alone does not preserve them. No commit, push or backup is claimed here.
