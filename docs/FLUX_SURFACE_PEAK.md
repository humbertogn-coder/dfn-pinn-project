# Early Surface Peak Audit

Run `python scripts/check_flux_surface_peak.py`. It loads the latest startup
checkpoints with legacy and full-radius sampling, checks matching seed/budgets,
and reuses the existing checkpoint/archived-prediction verification. No training
or checkpoint modification occurs.

Nested 81x81, 161x161 and 321x321 grids evaluate the same saved neural functions
on rho=[0.9,1], tau=[1e-5,1e-3]. Radial spacing is quadratic toward the surface;
time is logarithmic. This tests evaluation resolution, not solver convergence.
The report separates boundary-inclusive maxima, strict-interior maxima and
fixed-distance probes. It records c_tau and the spherical diffusion term
separately, signed residuals, and training counts in early surface bands.

The output directory contains report.json, samples.npz and surface_peak.png.
The figure compares signed residual maps on one shared scale and component
profiles at the earliest evaluated time. The surface PDE limit is a diagnostic,
not the surface flux boundary residual. Interior probes establish whether a
defect extends beyond that limit. No continuous bound or extrapolation to times
below 1e-5 is implied by a stable sampled maximum.

## Recorded Audit

Run `flux_surface_peak_20260920T055554383581Z`. Legacy source:
`constant_flux_pinn_20260916T234041990793Z`; full-radius source:
`constant_flux_pinn_20260920T054646100272Z`. All 109 tests passed and the figure
was visually inspected. The three nested grid maxima agree at printed precision:

| Quantity at tau=1e-5 | Legacy | Full radius |
|---|---:|---:|
| Surface absolute PDE residual | 0.8013863 | 7.181266 |
| Strict-interior sampled maximum | 0.7963147 | 7.166360 |
| Surface time derivative | -175.2199 | -178.7548 |
| Surface diffusion term | -174.4185 | -171.5735 |

Strict-interior maxima occur at rho=0.999999023. The defect is not solely an
exact-boundary diagnostic. Full-radius sampling has zero interior training
points with 1-rho<=0.001 and tau<=1e-4, and only one with 1-rho<=0.01 over
that time interval. Boundary flux training points do not replace interior PDE
collocation in these bands.

This supports testing targeted near-surface early-time interior coverage while
retaining early-core coverage and the same total budget. It does not prove
that sampling alone causes the defect. Both PDE terms are large near the
surface; their imperfect balance persists in the saved model. No retraining
or loss-weight adjustment was performed in this audit.
