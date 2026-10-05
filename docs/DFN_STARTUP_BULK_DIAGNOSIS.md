# Frozen startup residual localization

Command: `python scripts/diagnose_startup_bulk.py`.
Input: `results/particle_startup_smoke_20260929T073928375126Z`.
Output: `results/startup_bulk_diagnosis_20260929T074626918060Z`.
Artifacts: `report.json` and `profiles.csv`. No training or model edit ran.

## Method and verification

Source/checkpoint hashes and both saved metric sets replayed exactly. Model
tensors and source artifacts remained unchanged. Seven positive physical times
from 1e-6 to 1 second, three global x positions per electrode and 113 radial
points were sampled. The radial grid includes the center and surface and a
near-surface geometric refinement. These are sampled extrema, not certified
continuous bounds or quadrature-based RMS values.

The normalized diffusion residual is split into its time derivative T and
radial diffusion contribution D_r, so R=T-D_r. The implementation checks this
against the existing diffusion operator, including the regular center limit.

A second algebraic split defines the bulk term by evaluating the same raw
network with its surface-layer feature set to zero. The layer contribution is
the full concentration minus that bulk contribution. Time, radial and residual
components are checked to sum back to their full counterparts. This is a
decomposition of a fixed neural function, not two separately solved PDEs or
a claim that the layer is exactly zero at every interior point.

## Main finding

At t=1e-6 s in the positive electrode, the largest sampled residual occurs at
rho=0 and global X=0.5625:

| Normalized term | Value |
| --- | ---: |
| Time derivative | 2586.5599773 |
| Radial diffusion contribution | -2.56701e-7 |
| Residual | 2586.5599775 |
| Bulk contribution to residual | 2586.5599775 |
| Layer contribution at this point | 0 at machine precision |

The physical residual at this point is approximately 45339.5 mol/m3/s.
This is the erroneous neural prediction, not a physical reference value.
The layer component has a separate sampled maximum residual magnitude of
127.76, so surface-layer errors also exist even though they do not explain
the largest positive-particle peak.

Positive-particle maximum sampled residual magnitudes are:

| Physical time [s] | Maximum absolute normalized residual |
| --- | ---: |
| 1e-6 | 2586.56 |
| 1e-4 | 258.450 |
| 1e-2 | 25.6900 |
| 1 | 2.98360 |

The early sequence is consistent with a dominant inverse-square-root-time
derivative. It is not a fitted asymptotic proof. The negative-particle maximum
at 1e-6 s is about 129.30 near rho=0.999351, with bulk residual -129.33 and
layer residual +0.0259 at that point. Its interior region rho<=0.9 still has
maximum magnitude 122.86, so the negative error is not exclusively surface-local.

## Interpretation and next bounded change

For this frozen checkpoint, the excessive positive residual is directly
localized to the bulk time derivative rather than a matching diffusion term.
A generic raw network multiplied everywhere by sqrt(time) permits this failure.
This does not prove that every possible network with these features fails,
but it supplies concrete evidence for changing the candidate representation.

Do not train the current candidate longer automatically. Next construct a
separate candidate in which the square-root correction is confined to the
surface layer, while a bulk correction starts at O(time). First verify initial
values, center symmetry, inward/outward flux signs and finite positive-time
derivatives, and show on simple fixtures that the isolated bulk term no longer
has an inverse-square-root time derivative. The surface layer can still have
large residuals; no concentration bounds, inventory conservation or full DFN
validation follow automatically. Preserve this failed checkpoint and reports.
