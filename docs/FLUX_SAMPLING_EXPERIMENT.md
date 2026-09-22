# Sampling-Only Startup Experiment

Run `python scripts/train_constant_flux_particle.py --variant startup --sampling full_radius`.
Then run `python scripts/diagnose_flux_residual.py --sampling-comparison` to compare
the latest startup models with legacy and full-radius sampling. The original
diagnosis without this flag still compares baseline/startup legacy runs.

The first 384 interior points remain exactly unchanged. The last 128 are changed
from outer-region/log-time sampling to a jittered 16 radial by 8 log-time grid
covering the full radius. Total interior count remains 512; initial and boundary
points, random seed, initial network weights, loss coefficients and optimizer
budgets are unchanged. This changes the empirical distribution of the PDE loss;
it is not an importance-weighted estimate of an unchanged sampling measure.

New runs archive actual interior, initial and surface coordinates plus SHA256
in training_points.npz/report.json. Initial network weights are also hashed.
The diagnosis verifies saved coordinates against reconstruction and checks the
weights hash where available. Older checkpoints have no initial-weight archive;
their reproducibility still relies on the known seed, version and source.

The evaluation grid and physical quadratures stay fixed. Any improvements or
regressions must be reported together, including concentration, flux, mean
balance and independent PDE residual. One seed is a controlled pilot, not a
robustness claim. Existing checkpoints are never overwritten.

## Targeted Surface Layer

Run `python scripts/train_constant_flux_particle.py --variant startup --sampling surface_layer`.
Then `python scripts/diagnose_flux_residual.py --surface-layer-comparison` compares
the latest full-radius and surface-layer runs with the unchanged evaluation.

Exactly 64 former outer-region points (indices 320:384) are replaced by a jittered
8x8 design in log distance 1-rho in [1e-6,1e-2] and log time in [1e-5,1e-3].
The 128 full-radius points, 256 original uniform points and 64 outer points stay
unchanged. Total count stays 512, with no changes to architecture, initialization,
loss coefficients or optimizer budgets. This keeps early-core coverage while
adding thin-layer coverage. It still changes the empirical PDE loss measure.

Recorded surface-layer run `constant_flux_pinn_20260920T061044842318Z` and audit
`flux_residual_diagnosis_20260920T061134149321Z`, seed 42, default budgets:

| Diagnostic | Full radius | Surface layer |
|---|---:|---:|
| Maximum concentration error | 2.683769e-3 | 2.526867e-3 |
| RMS concentration error | 1.656522e-4 | 2.585203e-4 |
| Maximum mean balance error | 3.912067e-4 | 6.894379e-4 |
| Maximum surface flux residual | 4.558831e-2 | 4.045767e-2 |
| Original log-time grid PDE RMS | 0.2557995 | 0.04377585 |
| Dense sampled PDE maximum | 7.181266 | 0.4985870 |
| Volume/time-weighted PDE RMS | 0.03064931 | 0.03949125 |

The new dense maximum is at rho=0.97, tau=1e-5. Early-core coverage remains
26 points. Software tests: 110 passed. The shared-scale diagnostic figure was
inspected. Targeted sampling reduces extreme early residuals but worsens the
physical aggregate and mean balance; this is not an unconditional improvement.
Define explicit acceptance budgets before further tuning or DFN coupling.
