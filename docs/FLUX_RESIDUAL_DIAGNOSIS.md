# Saved-Model Residual Diagnosis

Run `python scripts/diagnose_flux_residual.py`. It chooses the latest baseline
and startup runs by recorded variant (old reports without a variant are baseline).
Use `--baseline <directory> --startup <directory>` to select explicit runs.
Matching seed and optimizer budgets are required. No weights or source run
artifacts are modified; results go to a new diagnosis directory.

The loader uses weights_only=True, checks architecture/configuration/PyTorch
version, reproduces archived concentration predictions and the old-grid PDE RMS.
It reconstructs the 512 training points from the current trainer's seed/RNG
sequence. This requires the recorded version and known historical sampler;
points were not archived in those old runs. It checks that both variants have
identical reconstructed points. Future trainers should save coordinates directly.

Outputs include report.json, residual_maps.npz and residual_maps.png. The map
uses a shared logarithmic color scale and overlays training locations in white.
It adds radial points near the surface, keeps logarithmic time, and records
the largest sampled residual together with its time-derivative and diffusion
terms. The center and surface are included as diagnostic limits, not interior
collocation points. Early/later regions split at t=1e-3; outer means rho>=0.9.
Map and regional RMS depend on this nonuniform sampling and must not be treated
as physical integrals or directly equated with training averages.

Physical RMS is sqrt(integral(R^2*3*rho^2 drho dt)/integral(3*rho^2 drho dt))
over rho in [0,1], t in [1e-5,0.02]. Radial Gauss quadrature and Gauss rules on
16 log-placed time panels integrate physical dt, not d(log t). Two resolutions
(64 radial/4 nodes per time panel and 128 radial/8 per panel) assess quadrature
sensitivity. A stable pair is evidence, not a continuous error bound.

No training, loss reweighting, best-checkpoint selection or new accuracy claim
is performed. Locating residual errors informs the next controlled experiment;
it does not establish that sampling alone is their cause.

## Recorded Diagnosis

Study `flux_residual_diagnosis_20260917T001646605183Z` used the user's baseline
`constant_flux_pinn_20260916T192518874707Z` and startup
`constant_flux_pinn_20260916T234041990793Z`. Reloaded predictions matched exactly,
the old-grid RMS was reproduced, and reconstructed training points matched.

| Metric | Baseline | Startup |
|---|---:|---:|
| Training-point PDE RMS | 0.0261423 | 0.0279035 |
| Original log-time grid PDE RMS | 0.0251633 | 0.274090 |
| Volume/physical-time PDE RMS, finer quadrature | 0.0334068 | 0.0351138 |
| Dense sampled maximum absolute residual | 0.253185 | 1.446107 |
| Peak rho | 1.0 | 0.1 |
| Peak time | 0.02 | 0.00001 |

Coarser weighted RMS values were 0.0334125 and 0.0351083 respectively.
The physical weighting makes the two aggregate errors much closer, but does
not remove the startup model's localized early-time defect. At its peak,
c_tau=1.446038 and the diffusion term=-6.852298e-5: the time derivative
dominates the mismatch, rather than cancellation of two large terms.

Direct inspection of reconstructed coordinates found zero training samples
with rho<0.7 and tau<=1e-4. The log-time portion of the original sampler is
restricted to the outer region. This is a concrete coverage gap, not proof of
causation. The square-root-time representation can also amplify time derivatives
near zero; a controlled sampling-only experiment is needed to separate effects.

The shared-scale map was visually inspected; all 106 project tests passed.
Next recommended experiment: improve early-time coverage throughout the radius,
keeping sample count, architecture, optimizer budget and evaluation fixed. Save
actual collocation points with future runs. Preserve both current checkpoints.
