# Direct-Kinetics Hard-Current Projection

Update: the subsequent matched full-budget attempt and its independent failed
audit are documented in DFN_PROJECTED_FULL.md. The original probe evidence
below is preserved; it was not a physical acceptance result.

Status: implemented and technically probed; no full-budget ablation or physical
acceptance. September 26, 2026 local time (September 27 UTC artifacts).

## Controlled Change

DFNProjectedCurrent subclasses the unchanged historical DFNSmoke assembly.
Only the two reaction-current callables are wrapped. Each query evaluates
the raw current on fixed physical electrode quadrature nodes at that query's
time, then adds (signed target - integral(a*j dx))/integral(a dx).
Targets are +I/A for the negative electrode and -I/A for the positive electrode.

The same callable is consumed by salt/charge sources, particle surface flux,
direct Butler-Volmer kinetics and particle charge/inventory integration. No
detach, clipping, positivity constraint or inverse kinetics is introduced.
All 34 residual terms and existing weights remain. Query batches are never
used as integration grids. Separate fixed quadrature at each query time
preserves pointwise x/time derivatives and parameter gradients.

This is the building block for planned variant F. The historical C0 result
remains a diagnostic-current baseline, not the explicit soft-current C arm.
A future C0/F contrast isolates adding the hard constraint, but does not
replace the complete planned soft/hard by direct/inverse comparison.

An additive raw-current constant is removed by projection, so its parameter
direction can legitimately have zero gradient. Tests require finite gradients
and a nonzero aggregate norm per branch, not nonzero derivatives for every
individual parameter. Exact integrated sources do not enforce local phase
currents, particle conservation, current signs everywhere or terminal voltage.

## Focused Checks

```bat
python scripts/check_dfn_projected_current.py
```

Five tests passed in 8.66 s after the final test update:

- Analytic projected polynomial values, x/time derivatives, second radial-free
  spatial derivative and parameter gradient.
- Batch/chunk invariance and zero cross-query Jacobian entries.
- Both signed electrode integrals with independent order-64 quadrature, and
  temporal charge integration through the shared current object.
- Deliberately underresolved projection detected by an independent quadrature.
- All 34 DFN residuals, full backward pass and exact in-memory checkpoint replay.

No historical source module was edited to add this variant. Existing reference
and full-baseline provenance remain intact.

## Technical Probe

```bat
python scripts/probe_dfn_projected_current.py
```

Twenty Adam steps, fresh seed 42, float64 CPU/one thread, baseline sampling
(128 interior points per region, 64 boundary times, temporal/radial inventory
order 32), projection order 32 and unchanged learning rate 0.001. This is not
a warm start from the failed C0 model and uses no reference labels.

Run: results/dfn_projected_current_probe_20260927T035754713638Z.

Median step: 0.546817 s; training/checkpoint section wall time: 11.2335 s.
The earlier unprojected dry run measured approximately 0.277 s per step.
That suggests about twice the cost here, not a controlled performance
benchmark; hardware load and timing variability were not paired.

All 20 steps retained finite gradients in the 12 branches. Loss before the
first update was 511.2254 and before the twentieth was 330.8565. These losses
do not establish superior accuracy. Frozen prediction replay was exact.
Independent order-64 spatial integration at 0, 1e-6, 0.01, 0.1, 0.5 and 1 s
gave maximum relative current-integral error 2.918910e-16 in each electrode.
This is sampled numerical agreement, not a continuous-domain error bound.

## Next Step and Limits

Connect an explicit variant-aware trainer and checkpoint loader to the
existing independent audits, retaining variant specification and quadrature
order in provenance. Historical DFNSmoke loaders must not silently load the
projected checkpoint, which has different state keys. Then execute the fixed
full-budget comparison, without changing direct kinetics or adding other
constraints. The projection may fix integrals and still fail voltage/local
balances; those remain independent acceptance criteria.

No full projected training, seed study, inverse kinetics experiment, Grace
job, commit or push was performed in this step. Preserve both probes and the
failed full C0 baseline. Generated artifacts need backup outside Git.
