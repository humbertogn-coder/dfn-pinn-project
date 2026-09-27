# Independent DFN Balance Audit

Update: the separate local PDE audit is now implemented and documented in
[DFN_PDE_AUDIT.md](DFN_PDE_AUDIT.md). The historical balance report below is
unchanged. Combined acceptance and future-run compatibility remain pending.

Status: PARTIAL_PHYSICAL_AUDIT_NOT_ACCEPTANCE. September 25 2026.
This is the first implemented part of the independent baseline auditor, not
the full launch gate. Per-equation PDE RMS/maxima, combined native field
evaluation and full-budget completion checks remain pending. Even if every
metric below passed, this stage would never issue overall acceptance.

## Implementation

src/dfn_pinn/dfn_balance_audit.py evaluates frozen fields without calling
DFNSmoke.residuals or using training losses as audit scores. It shares the
previously tested local operators. Independence refers to samples, integration
and audit assembly, not a separately coded implementation of every equation.
Source/reference identities, exact replay, and unchanged checkpoint/model
tensors are verified. Parameters are frozen; coordinate derivatives remain
enabled. No training or reference simulation is run.

Positive times combine 201 uniform and 121 logarithmic points from 1e-6 to 1 s,
giving 320 unique times. Each region uses 41 positions, including one-sided
boundary traces; particle admissibility also uses 41 radial positions. Initial
fields are checked separately at zero. The audit covers total phase current
in both electrodes AND separator; particle flux and kinetics; initial state
and center symmetry; admissibility; interface jumps; collector conditions;
electrode reaction integrals; local particle inventory; and global lithium.

Physical area, dx, porosity, solid fraction and spherical 3*rho^2 weights are
explicit in inventory integration. Predicted concentrations, not prescribed
inventory targets, determine the global totals. Local means use independently
integrated learned current at fixed x/time points. Independent orders 32/64
are compared with the protocol's 5%-of-limit metric-gap requirement; order 128
is used only if needed. This is not a rigorous quadrature error bound.

## Frozen Smoke Findings

Run: results/dfn_smoke_20260924T054745712381Z.
Report: balance_audit_20260925T052740252744Z/report.json within that run.
Source identity and exact replay passed; the checkpoint was unchanged.
Orders 32/64 passed all integration gap checks; order 128 was unnecessary.
The 42 scalar checks are not 42 independent experiments or a complete audit.

| Diagnostic | Observed maximum | Limit |
| --- | ---: | ---: |
| Negative total current relative error | 1.171674 | 0.01 |
| Separator total current relative error | 1.001373 | 0.01 |
| Positive total current relative error | 1.002573 | 0.01 |
| Electrode reaction integral relative error | 0.002909003 | 0.01 |
| Total lithium drift / initial total lithium | 8.740623e-4 | 1e-7 |
| Electrolyte lithium drift / initial electrolyte lithium | 4.995121e-4 | 1e-5 |
| Negative local normalized inventory error | 6.131435e-4 | 1e-6 |
| Positive local normalized inventory error | 4.472233e-4 | 1e-6 |
| Negative normalized surface-flux residual | 1.006861 | 0.01 |
| Positive normalized surface-flux residual | 1.003454 | 0.01 |

Initial concentration and center checks pass with zero errors. Several
interface/collector conditions pass, while kinetics and other currents fail.
The report retains every metric rather than combining them into one score.

Initial total lithium is 0.309642673 mol. Maximum drift is 2.706469928e-4 mol:
about 0.0874% of initial inventory but 5.223 times the charge-equivalent lithium
transfer I*duration/F. The latter is an auxiliary scale, not a statement that
transferred lithium should leave the closed cell; total lithium should remain
constant.

An integrated reaction source can pass while phase current fails. The smoke
reaction networks start near nominal source values, but the potentials do not
yet generate the required current. This illustrates why source integrals alone
are insufficient; it does not prove a training-failure mechanism. Three Adam
steps do not constitute a full-budget attempt.

## Focused Verification and Remaining Work

Five tests passed: physical inventory/geometry, a deliberately wrong positive
current, exact inventory with wrong surface flux, radial underresolution, and
nonfinite-field rejection. The constant-current manufactured fields in these
tests are NOT claimed to solve diffusion/kinetics. No full suite was run.

```bat
python scripts/audit_dfn_balances.py
```

This entry point currently requires exact smoke/reference settings and its
checkpoint schema. Finish PDE metrics and reference/completion integration
before adapting it to the future training configuration and clearing the
launch gate. The original configuration and preparation manifest are retained.
