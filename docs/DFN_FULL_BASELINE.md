# Fixed-Budget DFN Baseline

Implementation date: September 26, 2026 local time.

## Frozen Protocol

This is one C0 baseline attempt: learned reaction current and direct kinetics,
with no explicit soft electrode-integrated current loss or hard projection.
The physical model, 34 residual terms, loss weights, architecture, reference,
sampling and acceptance thresholds are unchanged. Initial stoichiometries
are synthetic (0.8/0.4); this is a one-second case, not the older complete
Chen2020 discharge. Initialization is fresh seed 42, never the dry-run state.

The configuration remains byte-for-byte unchanged to preserve existing audit
hashes. Its historical SPECIFICATION_ONLY status records preparation-time
state, not a new numerical setting. The implemented runner and this document
record subsequent launch preparation: independent component audits, integrated
historical evidence, and the completed baseline-size cost/replay dry run.

Budget: 2000 Adam steps at 0.001, then one L-BFGS call with at most 200
iterations, 250 closure evaluations, history size 50 and strong Wolfe search.
Float64 CPU, one thread. A two-hour cap is checked at optimizer/closure
boundaries; it cannot interrupt the middle of a single evaluation. No automatic
retry, seed sweep, weight change or budget extension is implemented.

```bat
python scripts/train_dfn_baseline.py
```

## Checkpoints and Completion

Every 100 Adam steps, an atomic checkpoint and updated report are written.
before_lbfgs.pt preserves the final accepted Adam state. Final checkpoint.pt
is only produced after a normal bounded L-BFGS return and finite final loss.
It contains model, optimizer, samples, prediction replay values, RNG state,
phase and versioned completion metadata. Interrupted runs retain earlier
checkpoints; automated resumption is not implemented in this first runner.

A failed line search is not a completed result. If its evaluation cap, wall
cap or finite checks fail, the report records STOPPED_NOT_COMPLETED and the
pre-L-BFGS checkpoint is retained. No provisional line-search point is promoted
to the final model. Original failure evidence is preserved.

The dfn_full_run_v1 contract records actual Adam steps, actual L-BFGS iteration
and evaluation counts, wall time and stopping reason. A normal early return
is allowed within the prescribed maximum budget but is not labeled optimizer
convergence. Completed execution is separate from physical acceptance.

## Audit Compatibility

The field, balance/PDE and summary entry points now recognize the versioned
full-run schema. They verify report/checkpoint settings, completion identity,
checkpoint/config/reference hashes and exact replay. Physical settings must
match the reference; known optimizer and sampling differences do not invalidate
that physical comparison. Legacy checkpoints retain strict historical checks.
Stored source hashes are not rewritten to bypass provenance checks.

After a completed run, use its explicit directory:

```bat
python scripts/compare_dfn_reference.py --run results/RUN_DIRECTORY
python scripts/audit_dfn_balances.py --run results/RUN_DIRECTORY
python scripts/audit_dfn_balances.py --run results/RUN_DIRECTORY --pde-only
python scripts/summarize_dfn_audit.py --run results/RUN_DIRECTORY --fields FIELD_REPORT_PATH --balances BALANCE_REPORT_PATH --pdes PDE_REPORT_PATH
```

Component report paths are explicit, relative to the selected run or absolute;
the summary does not guess the latest artifacts. Its defaults remain pinned
to the old smoke, so all component arguments must be supplied for a new run.
Overall FAIL is retained if a completed attempt fails sampled physical metrics.
Even a complete sampled PASS remains provisional, not a general DFN guarantee.

## Focused Software Verification

Sixteen tests passed in 8.09 s: full-run completion and compatibility guards,
existing native-field comparison helpers, and fail-closed report integration.
The full test suite was not run. These tests do not establish physical accuracy.

## Execution Evidence

The single attempt was launched in
results/dfn_baseline_full_20260927T023312902451Z (UTC timestamp).
The attempt completed normally: 2000 Adam steps, 200 L-BFGS iterations and
220 closure evaluations. Training wall time was 524.60 s (8.74 minutes),
excluding independent audits. Final training loss was 1.0723246463, versus
513.2868 before the first Adam update. Exact saved prediction replay passed.

Pinned reports within that run:

- reference_comparison_20260927T024225003710Z/report.json
- balance_audit_20260927T024302003266Z/report.json
- pde_audit_20260927T024352511381Z/report.json
- combined_audit_20260927T024431947943Z/report.json

Overall status: **FULL_ATTEMPT_AUDITED_FAIL**. All 83 required metrics were
present; 1/11 native-field, 21/42 balance and 16/30 PDE/quadrature metrics pass.
Counts include integration checks and are not an accuracy score.

| Metric | Observed | Limit |
| --- | ---: | ---: |
| Maximum voltage error | 0.1327931 V | 0.005 V |
| Maximum electrolyte concentration error | 17.58952 mol/m3 | 10 mol/m3 |
| Negative total-current relative error | 1.021945 | 0.01 |
| Positive collector solid-current normalized residual | 0.9999988 | 0.01 |
| Electrode reaction-integral relative error | 1.012005 | 0.01 |
| Total lithium drift / initial inventory | 6.585932e-6 | 1e-7 |
| Negative local inventory error | 1.832586e-5 | 1e-6 |
| Positive local inventory error | 2.936082e-5 | 1e-6 |

Sampled negative reaction current is only 0.06699-0.06764 A/m2 active area;
positive reaction current is +0.01932 to +0.02032 A/m2, contrary to the expected
negative discharge sign in that electrode. The positive collector solid
current is nearly zero relative to its imposed target. Thus reduced residuals
coexist with incorrect driven-current behavior. This is consistent with a
low-current nonphysical solution, not proof of a unique optimizer mechanism.

Nine of ten physical-measure PDE RMS values fail the 0.01 threshold. The
largest are negative/positive electrolyte salt (0.09454/0.09588). Negative
particle peak residual is at rho=1, x=0, t=1e-6 s; positive particle peak is
at rho=1, x=97.2 micrometers, t=1 s. Peak locations and signs for every equation
are retained in the PDE report. Orders 32/64 satisfy all reported quadrature
gap checks; this does not certify continuous-time extrema.

## Interpretation and Next Decision

This is the first fully budgeted, independently audited C0 result, one seed
and one short protocol only. It is not a validated forward solver and cannot
support experimental parameter identification or a speed advantage claim.
No thresholds, weights or budgets were changed in response to the outcome.

Preserve this failed baseline. The next bounded methodological comparison
should investigate the current constraint under the planned ablation protocol,
with current projection and inverse kinetics kept as separate factors. Do
not silently relabel C0 as the explicit soft-current C variant, or assume
projection will fix local currents merely because it fixes source integrals.
No ablation training, repeat seed, automatic budget extension, commit or push
was performed after this result. Back up generated evidence separately.
