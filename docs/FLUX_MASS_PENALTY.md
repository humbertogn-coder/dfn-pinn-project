# Optional Mean-Inventory Penalty

First exploratory run: startup/full_radius, seed 42, mass weight 1, unchanged
2000 Adam steps and 300 L-BFGS maximum iterations. The network, initial weights,
PDE/initial/flux points and their existing loss weights remain unchanged.
Additional integral constraints increase computation per optimization step;
equal optimizer budgets do not imply equal wall time or total training points.

The new loss is mean(((3 integral(r^2 c dr) - (1-3t))/0.06)^2).
The target follows conservation for unit initial concentration and prescribed
unit outward flux, not fitted interior labels. It is a soft constraint, not
an exact projection. Weight 1 is an exploratory starting choice, not optimized.
Training uses 64-point Gauss-Legendre radial quadrature and the union of 32
linear and 32 logarithmic times. Evaluation retains 128 radial points and 120
positive logarithmic times. Record settings in each report; inspect sensitivity
to further quadrature refinement before drawing strong conservation conclusions.

Run `python scripts/train_constant_flux_particle.py --variant startup --sampling full_radius --seed 42 --mass-weight 1`.
Omitting --mass-weight keeps the original objective and four-column history.
The original candidate/diagnostic run selector excludes penalized runs to avoid
silently changing its comparison. A five-seed penalty study is not yet run.

Check concentration, flux, mass, and PDE residuals together. Preserve any
regressions. Existing exploratory acceptance limits are unchanged.

## Seed-42 Result

Control: `results/constant_flux_pinn_20260920T054646100272Z`.
Penalty: `results/constant_flux_pinn_20260920T074543493433Z`.
Initial weight hashes match, and all three original training point arrays are
exactly equal. The full suite passed 134 tests.

| Metric | Control | Mass weight 1 |
| --- | ---: | ---: |
| Maximum concentration error | 2.683769e-3 | 1.211685e-3 |
| RMS concentration error | 1.656522e-4 | 1.274017e-4 |
| Maximum flux residual | 4.558831e-2 | 2.557441e-2 |
| Maximum mean balance error | 3.912067e-4 | 3.502633e-4 |
| Original grid PDE RMS | 2.557995e-1 | 3.511120e-1 |
| Physical-volume/time PDE RMS | 3.064931e-2 | 3.873021e-2 |

Reloaded checkpoint verification succeeded. Independent physical PDE quadratures
gave 0.03873088310 (64 radial, 4 time nodes/panel) and 0.03873020681
(128 radial, 8 time nodes/panel). The final normalized training mass loss was
9.371528162e-6, 9.371528161e-6 and 9.371528162e-6 using 64, 128 and 256 radial
nodes at the same mass-training times. These checks are numerical sensitivity
evidence, not continuous-domain bounds.

Concentration and flux improved, but mass improved only modestly and physical
PDE RMS regressed. The concentration, flux and mass exploratory targets still
fail for this seed. Do not promote this run to a five-seed success claim or
replace controls. Before broader training, examine whether hard inventory
enforcement can avoid the competing soft-loss objectives; this is a proposed
next experiment, not implemented functionality.
