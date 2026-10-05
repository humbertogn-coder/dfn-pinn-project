# Project Language

- Communicate with the project owner in Spanish.
- Write all new or modified project content in English, including documentation,
  code comments, docstrings, CLI messages, configuration descriptions, and reports.
- Use English for new commit messages.
- Preserve proper names, existing filesystem paths, URLs, and external identifiers.

# Research Workflow

- Develop in small, reviewable steps and explain each step to the project owner.
- Distinguish planned methods from implemented and validated functionality.
- Record simulation settings and validation evidence before interpreting results.

# Current Research Status

- The single-particle exploratory pilot is closed as of 2026-09-22 at the owner's request.
- Read `docs/PILOT_CLOSEOUT.md` before continuing this research.
- Preserve the documented ramp startup overshoot and wrong-sign flux limitation.
- Do not infer full DFN validation or resume training, seed studies, Grace jobs,
  or coupling work from historical next-step notes; await an explicit new request.
- The owner subsequently authorized the next stage after uploading the closeout.
  Current bounded work is documented in `docs/PARTICLE_COUPLING_STAGE.md`;
  the closed pilot's limitations remain in force.

- The owner approved the revised full-DFN methodology on 2026-09-25.
  Read `docs/RESEARCH_ROADMAP_V2.md` and `docs/DFN_PDE_AUDIT.md` for current
  scope, evidence and the next bounded step. Do not silently resume old plans.
  Historical audit integration is documented in `docs/DFN_COMBINED_AUDIT.md`;
  the completed bounded dry run is documented in `docs/DFN_DRY_RUN.md`.
  The first completed full-budget baseline and its failed independent audit
  are recorded in `docs/DFN_FULL_BASELINE.md`. Preserve this C0 result. The
  next bounded decision is the controlled current-constraint comparison;
  do not extend budgets, relabel C0 as soft-current C, or claim validation.
  The hard-current variant building block and bounded technical probe are
  documented in `docs/DFN_PROJECTED_CURRENT.md`. Explicit variant-aware loading
  and the full projected comparison are now complete; see
  `docs/DFN_PROJECTED_FULL.md`. Both full runs failed physics criteria. Preserve
  the mixed findings; next diagnose frozen charge/kinetics/flux gradients,
  without automatic retraining or claims that exact integrals solve the DFN.
  That frozen diagnosis is now complete: `docs/DFN_GRADIENT_DIAGNOSIS.md`.
  It found weak positive-collector sensitivity, sampled hidden-layer saturation
  and sampling-sensitive gradients. Next isolate potential representation/
  scaling; do not treat these observations as proof of a sole failure cause.
  The isolated amplitude check is complete: see
  `docs/DFN_SOLID_POTENTIAL_SCALING.md` (8 focused tests passed; no training).
  The candidate is not wired into trainers or checkpoint loaders. Next is a
  bounded matched fresh-initialization probe, not a full-budget rerun.
  That 20-step paired C0 probe is complete; see
  `docs/DFN_POTENTIAL_SCALE_PROBE.md`. Results are mixed: positive kinetics
  improves, but collector current and several charge residuals worsen versus
  control. Next inspect the frozen short-run gradients; no automatic full run.
  Frozen short-run gradients are now documented in
  `docs/DFN_POTENTIAL_PROBE_GRADIENTS.md`: strongest selected opposition is
  local solid charge versus kinetics, not collector versus kinetics. Neither
  short-run positive branch shows sampled collector saturation. No full run
  or architecture change is justified by this diagnostic alone.
  The actual next Adam direction was checked after exact 20-step replay;
  see `docs/DFN_PROBE_ADAM_DIRECTION.md`. Disposable candidate only, no accepted
  update: scaled collector and kinetics improve together, positive charge
  slightly worsens, and electrolyte mass dominates total loss reduction.
  Next proposed bounded experiment is a matched 200-step Adam probe, with
  optimizer checkpoints and per-term held-out metrics; no automatic full run.
  The matched 200-step probe is complete: `docs/DFN_POTENTIAL_200_PROBE.md`.
  Ohmic scaling improves the positive collector and kinetics versus control,
  but sharply worsens zero solid current at the separator. Both remain invalid.
  All milestones and optimizer continuation replay were verified. Do not
  automatically extend training; next examine joint solid-current boundary
  representation as a separate, analytically checked intervention.
  The isolated joint-boundary candidate passed 10 analytic checks; see
  `docs/DFN_SOLID_BOUNDARY_POTENTIAL.md`. It is not trainer-integrated and does
  not solve local charge, kinetics or the DFN. Next is explicit variant wiring
  and minimal neural/replay verification before a bounded training comparison.
  Explicit neural wiring is complete: `docs/DFN_BOUNDARY_VARIANT_SMOKE.md`.
  Four focused tests and a three-step smoke passed, with exact checkpoint and
  optimizer continuation replay. Hard boundary identities persist; interior
  physics is unsolved. Next is a separately authorized matched 200-step probe,
  not a full run or use of historical factories for this new checkpoint schema.
  The matched boundary 200-step probe is complete; see
  `docs/DFN_BOUNDARY_200_PROBE.md`. Hard boundaries persist and solid charge
  improves, but electrolyte charge, particle flux and interior total current
  remain poor. All controls and milestone/optimizer replays were verified.
  Next inspect frozen spatial current/source/particle-flux profiles in SI units;
  no automatic budget extension or additional hard constraints.
  Frozen SI profiles are complete: `docs/DFN_BOUNDARY_COUPLING_PROFILES.md`.
  At 1 s, separator electrolyte current is about 0.45 versus required 48.69
  A/m2 geometric; particle flux is near zero relative to learned reaction j,
  with wrong-sign positive-particle flux. Negative BV current is also near zero.
  No training ran. Next isolate residual scaling/branch sensitivity before
  choosing one coupling intervention; do not infer the cause from profiles alone.
  Frozen scale/branch diagnostics are complete; see
  `docs/DFN_BOUNDARY_SCALE_GRADIENTS.md`. Coupling gradients are nonzero;
  positive-particle flux gradients on concentration are much smaller than PDE
  gradients, without strong opposition on that branch. No automatic reweighting.
  Proposed next bounded subproblem freezes reaction current to isolate particle
  diffusion/flux learning; it is not full DFN validation or a reopened pilot.
  The frozen-current particle subproblem is complete; see
  `docs/DFN_FROZEN_PARTICLE_SUBPROBLEM.md`. After 200 particle-only steps,
  diffusion improves but surface flux remains near 0.98-0.99. Frozen branches,
  checkpoint metrics and optimizer replay were verified. No successful particle
  or coupled solution is claimed. Next examine a startup-aware representation
  analytically, respecting the t=0 uniform-profile/nonzero-flux incompatibility;
  do not automatically extend training or reopen the historical pilot.
  A dimensionally scaled startup candidate passed eight analytic/neural checks;
  see `docs/DFN_PARTICLE_STARTUP_CANDIDATE.md`. It is not trainer-integrated,
  has no hard flux/inventory/bounds, and is not a diffusion solution. Next is
  explicit metadata and minimal frozen-current wiring/replay, followed only
  with authorization by a clearly initialized paired comparison. Four inputs
  prevent direct loading of historical three-input particle weights.
  Startup particle wiring and a three-step smoke are complete; see
  `docs/DFN_PARTICLE_STARTUP_SMOKE.md`. Four focused tests and exact model/Adam
  replay pass, frozen fields stay unchanged and sampled concentrations stay
  within [0,1]. However, positive diffusion RMS is about 549: no physical
  improvement is claimed. Do not automatically launch 200 steps; first localize
  the early residual and check for an unbalanced bulk square-root-time term.
  Localization is complete: `docs/DFN_STARTUP_BULK_DIAGNOSIS.md`. At 1e-6 s,
  the positive peak lies at the center: time term 2586.56 versus radial term
  -2.57e-7, entirely in the defined bulk component. No training ran. Next
  analytically check a separate surface-confined sqrt-time plus O(time) bulk
  representation; preserve the failed unrestricted-startup candidate.
  Confined startup plus linear-time bulk passed six focused checks; see
  `docs/DFN_PARTICLE_CONFINED_STARTUP.md`. No training ran. Next is the agreed
  fixed 200-step paired frozen-current comparison, with metadata/replay checks
  included in that task (not another stand-alone smoke). Require diffusion,
  flux and inventory evidence; if it stalls again, stop local variants and
  review strategy with the owner rather than extending budgets automatically.
  The paired 200-step experiment is complete; see
  `docs/DFN_PAIRED_CONFINED_STARTUP.md`. Confined flux improves to RMS 0.0356/
  0.0270, but negative diffusion and positive inventory/kinetics worsen versus
  its initialization. No joint acceptance. Both arms, optimizer replays and
  frozen identities are verified. Close this local representation comparison;
  discuss a bounded training-strategy decision before more variants/full runs.
  Strategy review is recorded in `docs/DFN_JOINT_TRAINING_DECISION.md` and
  `configs/dfn_joint_kinetics_feasibility_v1.json` (specification-only). Next
  implement one bounded joint direct/inverse pair, all fields trainable, same
  pinned warm start and physical metrics. No new coupled run has executed.
  Keep weighting/projections fixed; do not relabel this as the C-F factorial.
  That joint pair is now complete: `docs/DFN_JOINT_KINETICS_RESULTS.md`.
  Both arms ran 200 all-field Adam steps with common physical metrics and
  exact model/optimizer replay. Inverse improves some particle metrics but
  neither passes; electrolyte charge/negative kinetics remain major failures.
  No independent full reference/global audit ran. Close the pair, preserve
  both failures, and review a separate strategy decision before more training.
  The recorded-loss review and weighting decision are now complete; see
  `docs/DFN_LOSS_WEIGHTING_DECISION.md`. Charge/kinetics already dominate loss
  magnitude; this is not evidence about gradient dominance. Next bounded task
  is one matched direct-kinetics inventory-weight pair (1 versus 1e8), same
  pinned initial state, no other intervention. Trainer is not implemented and
  no new training ran. Preserve all common physical gates and mixed outcomes.
  The fixed inventory-weight pair is now complete; see
  `docs/DFN_INVENTORY_WEIGHT_RESULTS.md`. Three focused tests, exact historical
  control replay and both optimizer continuations passed. Inventory improves
  but remains above tolerance; negative diffusion and positive flux worsen.
  No coupled acceptance, reference/global audit or weight search. Close this
  experiment; review unresolved charge/kinetic coupling before further runs.
  September 30 recovery verified recorded inputs/sources and checkpoints;
  no training was repeated after interruption.
  Frozen potential diagnosis is complete: `docs/DFN_CHARGE_KINETIC_OFFSETS.md`.
  Negative kinetics has an approximately 86.6 mV mean eta deficit at 1 s;
  disposable constant shifts reduce its RMS but worsen positive kinetic RMS.
  Nonkinetic residuals stay unchanged. Separator electrolyte current remains
  about one tenth of applied current. No shifted checkpoint was accepted.
  Next specify one potential level/spatial-shape representation intervention;
  no automatic calibration, training, weight search or physical success claim.
