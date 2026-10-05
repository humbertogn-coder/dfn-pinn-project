# Neural solid-boundary variant: minimal wiring check

## Implemented scope

`DFNBoundaryVariant` is an explicit experimental subclass in a new module.
Historical factories and source files remain unchanged. It replaces only the
two solid-potential branches, reusing their initial raw neural weights and
time features. The adapter removes the former output scaling and base before
the boundary representation applies its own lifting, scaling and gauge.
Direct Butler-Volmer kinetics, learned reaction sources, the remaining branches
and all 34 residual terms remain present. No source-current projection is used.

This is not an amplitude-only change: the boundary lifting, spatial coordinate
map and hard negative gauge form a separate representation intervention.
Geometry, amplitudes, base potentials, time factors and variant identity are
explicit checkpoint metadata. Restoration recomputes them from settings,
rejects mismatches, verifies fixed buffers and loads neural tensors strictly.
Existing full-run audit tools do not yet accept this experimental schema.

## Verification

`python -m pytest tests/test_dfn_boundary_variant.py -q -p no:cacheprovider`

Result: **4 passed in 7.81 s**. Checks cover equal initial raw solid-network
weights, unchanged residual inventory, finite gradients with a nonzero aggregate
in every neural branch, neural boundary identities and rejection of altered
variant metadata, amplitudes and fixed base buffers. An individual parameter
may legitimately have zero gradient, for example a negative additive offset
removed by the gauge subtraction.

`python scripts/smoke_dfn_boundary_variant.py`

Run: `results/dfn_boundary_smoke_20260928T200805250554Z`.
Status: SMOKE_DIAGNOSTIC_ONLY. Three Adam steps at 1e-4, seed 42, float64 CPU,
one thread, eight interior samples per region, three boundary times and
inventory quadrature order eight. This uses the small wiring settings, not
the 200-step comparison settings; its loss must not be compared directly to
those runs.

Pre-step losses: 500.912, 499.639, 498.369. Final sampled loss: 497.101830.
Across initialization and all three updates, sampled solid separator-current
residuals and the negative gauge were zero. The maximum sampled positive
collector-current residual was 2.220446e-16. This confirms neural wiring of
the analytic identities, not independently learned boundary accuracy.

The checkpoint includes optimizer state, samples, RNG state and explicit
representation metadata. Final metrics replay exactly after disk reload.
A further step on disposable copies matches both neural tensors and Adam
state exactly; it is not saved as an accepted fourth training step.

## Remaining limitations and next step

Final normalized RMS residuals include positive solid charge 0.311976,
positive kinetics 0.845228 and positive particle flux 1.002838. The interior
physics is not solved. No reference simulation, physical acceptance audit,
full test suite, seed sweep or long training ran.

The next bounded experiment can compare this explicit boundary representation
with the original and ohmic 200-step controls using exactly their saved raw
initial weights, physical settings, samples, learning rate and fixed budget.
Preserve the earlier controls and use separate variant-aware checkpoints.
Report the benefit of enforced boundaries separately from learned interior
equations; do not declare success merely because boundary residuals vanish.
