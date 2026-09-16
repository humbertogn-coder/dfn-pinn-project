# Integral Current Projection

Implemented in `src/dfn_pinn/projection.py`. For each electrode and time:

    correction = (target - sum(w*a*j_raw)) / sum(w*a)
    j_projected = j_raw + correction

Here w is physical dx [m], a is active-area density [1/m], j is interfacial
current [A/m2 active area], and target is signed geometric current [A/m2].
For discharge use target=+I/A in the negative electrode and -I/A in the positive
electrode. The caller supplies these signs explicitly; do not pass amperes.

`gauss_legendre` returns fixed physical x and dx, including the interval Jacobian.
If the network takes x/L, normalize its input but do not renormalize those weights
or multiply them by L again. This helper does not differentiate geometry.

`project_current` accepts raw arrays shaped (..., number_of_quadrature_nodes).
Area density is positive and scalar or node-wise; weights are a positive vector.
The target is scalar or exactly the leading batch shape. All tensors must share
dtype/device. Empty arrays, invalid shapes, nonfinite values and zero area are
rejected. The current implementation excludes partly inactive regions (a=0).

For nonquadrature query points call `current_correction` using raw currents on
the fixed quadrature grid, then add that same correction to raw query currents
at the same times. Do not recompute the correction from arbitrary collocation
batches, and do not detach it: its dependence on model parameters and time is
part of the differentiable projected field. Use projected j everywhere in the
future coupled PDE, flux and kinetic residuals.

The correction is additive and does not force a local sign. Its weighted sum
matches the target up to floating-point roundoff, but local conservation and
the exact continuous integral are not guaranteed. Large cancellation can impair
accuracy. Monitor the balance and the correction magnitude during training.

## Verification

Run `python scripts/check_projection.py`. Tests cover both electrode signs,
rest, variable active area, physical interval Jacobians, idempotence, batches,
the analytic current Jacobian, time-dependent targets, first/second derivatives,
invalid inputs and a float32 smoke check.

Independent quadrature tests reuse the original correction on a different grid.
A resolved polynomial preserves the target. A deliberately unresolved x^2
example with one construction node gives an independent integral error of 1/12,
despite exact balance on the construction grid. This is expected and demonstrates
why an independent quadrature audit is necessary.

No DFN training, HDF5 modification or continuous-conservation claim is included.
