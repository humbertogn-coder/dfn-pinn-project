# Local DFN PDE Audit

Implemented and checked on September 25, 2026. This is a diagnostic audit,
not full DFN acceptance or permission to launch a training campaign.

## Scope

The frozen-field auditor assembles ten equations without calling the training
residual assembly: electrolyte salt and charge in three regions, solid charge
in two electrodes, and spherical diffusion in both particles. It shares the
tested differential operators with training; it is not an independent second
implementation of the governing equations.

Each equation receives a physical-measure RMS and a sampled maximum with a
signed residual and physical peak location. RMS uses uniform physical time,
physical dx within each region, and 3*rho^2*d_rho for particles, normalized by
the integrated measure. Residuals retain their existing nondimensional scales.
Gauss-Legendre orders 32 and 64 are compared; 128 is used if their RMS gap
exceeds 5% of the RMS acceptance limit. Quadrature agreement is not an error
bound and does not certify unsampled extrema.

Maxima use the fixed protocol's 41 x positions per region, 41 radial positions,
and 320 distinct mixed uniform/logarithmic positive times. Region endpoints
are evaluated as separate one-sided traces. The time interval is [1e-6, 1] s;
the initial current-step corner is excluded. Initial conditions and center
symmetry remain separate checks in the balance auditor.

## Reproduction

```bat
python scripts/audit_dfn_balances.py --pde-only
```

This command verifies historical source identity, matched reference identity,
exact checkpoint prediction replay, and unchanged checkpoint/model tensors.
It does not train, run PyBaMM, repeat the balance audit, or alter old reports.
The default runner still requires the original smoke checkpoint schema and
exact reference settings. Future training compatibility is not yet claimed.

## Recorded Evidence

Checkpoint: results/dfn_smoke_20260924T054745712381Z.
Report: pde_audit_20260925T171946440055Z/report.json within that directory.

| Equation | Physical-measure normalized RMS | Sampled maximum |
| --- | ---: | ---: |
| Negative electrolyte salt | 15.14902 | 15.64444 |
| Negative electrolyte charge | 2.031115 | 2.033234 |
| Negative solid charge | 1.642807 | 1.833703 |
| Negative particle diffusion | 1.368717 | 1.769266 |
| Separator salt | 2.433385 | 2.545333 |
| Separator charge | 0.0007011992 | 0.0008499455 |
| Positive electrolyte salt | 15.46235 | 16.14472 |
| Positive electrolyte charge | 2.292350 | 2.293195 |
| Positive solid charge | 2.292095 | 2.292658 |
| Positive particle diffusion | 2.160512 | 3.216552 |

Limits remain RMS <= 0.01 and sampled maximum <= 0.1. Nine equations fail
both checks. All ten order-32/64 RMS gap checks pass; no order-128 evaluation
was necessary. These magnitudes use equation-specific normalization, not a
common SI unit. They do not establish which optimizer gradient dominates.
The checkpoint has only three Adam steps, so this is not evidence that a
full-budget baseline or the proposed reformulation fails.

Five focused tests passed in 11.03 s: spherical/physical-time weights,
manufactured signed sources, radial polynomial derivatives including the
center, complete equation coverage without training assembly or model changes,
and underresolution/nonfinite detection. No full test suite was run.

## Remaining Gate

Combine native reference-field errors, balance metrics and PDE metrics under
one versioned report with matching checkpoint/reference identities. Separate
physical compatibility from optimizer/sampling settings for future runs, and
check completion/budget metadata explicitly. Then perform the bounded cost,
gradient and checkpoint-replay dry run. No overall PASS is currently issued.
