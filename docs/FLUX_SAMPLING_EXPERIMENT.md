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
