# Joint training strategy decision

## Close the local representation sequence

The confined particle map improved flux markedly but did not simultaneously
solve diffusion, inventory and kinetics. This is partial evidence, not an
accepted particle solution. Preserve all failures and stop adding local
representation variants. The closed historical pilot is not reopened.

Two facts matter for the next decision: frozen electrolyte fields cannot fix
the observed transport failure, and excluding kinetics from the particle
objective permits concentration changes that worsen kinetic consistency.
These limitations follow from the experiment design, not a newly discovered
code bug. There is also no guarantee that unit MSE weights adequately enforce
inventory: its normalized error is much smaller numerically than several other
terms. Do not conceal that limitation or choose weights from a single gradient
snapshot.

## Selected next experiment

Return to the approved research question in RESEARCH_ROADMAP_V2: direct versus
inverse symmetric Butler-Volmer, with all DFN fields trained jointly. Keep the
confined concentration map and hard solid-current boundary map identical in
both arms. Start from identical copies of the pinned confined step-200 state,
and reset Adam in both arms. This is a warm-start feasibility comparison, not
a fresh baseline, a robustness claim or the original soft/hard C-F factorial.
Label arms joint_direct and joint_inverse, not C, D, E or F.

All parameters become trainable, including electrolyte potentials and
concentrations, solid potentials, particle concentrations and reaction current.
Use the same learned current everywhere. Keep all 34 loss terms and all unit
weights; replace only the two kinetic residual expressions in the inverse arm.
Do not add integrated-current projection, inventory projection or reweighting
in the same experiment. Existing hard solid boundary identities remain.

Direct residual: `(j-BV(eta,j0,T))/j_ref`.
Inverse residual: `(eta-inverse_BV(j,j0,T))/phi_ref`.

Both encode the same symmetric one-electron kinetic relation for positive j0,
but their gradients, units and effective weighting differ. Equal scalar weights
do not imply equivalent losses. Evaluate BOTH arms with the same physical
current-form kinetic consistency metric and the same PDE/flux/inventory checks.
Do not rank models by comparing their two raw total training losses.

## Fixed scope and stopping rules

The non-executable specification is
`configs/dfn_joint_kinetics_feasibility_v1.json`. It pins source checkpoint,
report and sample hashes. The implementation remains pending; no new coupled
training has run under this specification.

Budget: 200 Adam steps per arm at 0.001, float64 CPU, one thread, no L-BFGS.
The paired wall cap is **1200 seconds (20 minutes)**.
The cap is a safety limit, not an expected runtime. No automatic extension,
seed sweep, weight search or Grace job. Save milestones 0/20/50/100/200 and
optimizer state. Preserve sampled domain failures; do not clip or regularize
kinetics silently. For clarity, this is one integration-and-comparison task,
not a new series of separate smoke stages.

At each milestone report training and evaluation PDE RMS/maxima, flux,
inventory, common kinetic consistency, interior total current, interfaces,
boundaries and concentration ranges. Initial conditions and hard identities
must remain intact. New evaluation points never enter the optimizer; reused
historical diagnostic points must be identified as such.

The original physical acceptance thresholds are unchanged. Examples include
PDE RMS 0.01 and sampled maximum 0.1 per equation, normalized maximum flux/
kinetics error 0.01, local inventory maximum error 1e-6 and total-current
relative maximum error 0.01. These are gates, not automatic loss weights.
Sampled milestone diagnostics alone do not supply the independent reference,
quadrature and global-balance evidence required for full acceptance.

## Decision after this pair

If common metrics improve jointly but do not pass, record partial progress and
make an explicit budget/strategy decision; do not call it validated or launch
a full run automatically. If both arms fail substantially, stop the feasibility
sequence and review the formulation and weighting as a separate intervention.
If required independent evidence is missing, status is INCOMPLETE, never PASS.
Further full-budget training is not justified by one improved metric alone.

No claim is made that inverse kinetics will solve the electrolyte or inventory
problem. This choice tests a central planned hypothesis while restoring the
coupling excluded from the recent particle-only diagnostics.
