# Coupled Particle V1 Results

One full-budget seed-42 run was completed with the unchanged v1 configuration:
2000 Adam steps followed by LBFGS. No budget extension or threshold relaxation.
Run: results/coupled_particle_full_20260922T204227948958Z.
Audit: audit_20260922T204358449519Z/audit.json inside that run.
Status: SAMPLED_TARGETS_FAIL. This is a manufactured single-particle case, not DFN validation.

| Metric | Value | Limit |
| --- | ---: | ---: |
| Maximum concentration error | 5.519567e-6 | 1e-3 |
| Maximum current error [A/m2] | 1.422361e-3 | 1e-3 |
| Maximum flux residual [A/m2] | 9.782961e-4 | 1e-3 |
| Maximum BV residual [A/m2] | 1.422630e-3 | 1e-3 |
| Maximum inventory error | 3.573574e-6 | 6e-5 |
| Initial profile error | 1.110223e-16 | 1e-12 |
| Physical normalized PDE RMS | 8.249065e-3 | 1e-2 |

Sampled current and concentration-derived outward flux are positive. Other
configured checks passed, but the current and BV criteria failed. Exact saved
model replay and saved training loss reproduction were verified.

## Peak Diagnosis

No retraining. Two uniform time grids (1001 and 2001 points before excluding
zero) augmented with 281 log-spaced times from 1e-8 to 0.1 s found the same
largest current and BV residuals at t=10 s:

- Learned current: 0.09857763855 A/m2.
- Reference current: 0.1 A/m2.
- Current implied by BV at learned surface concentration: 0.1000002681 A/m2.
- Concentration-derived outward flux expressed as current: 0.09863006323 A/m2.
- Signed current error: -0.001422361449 A/m2.
- Signed BV residual: -0.001422629579 A/m2.

At sampled times in (0,0.1] s both errors stayed below 0.000978776 A/m2;
in (0.1,1] s below 0.000901296 A/m2. This suggests an end-time coupling
mismatch rather than the earlier synthetic pilot's startup defect. It does
not prove a cause or establish continuous-time bounds. Times below 1e-8 s
remain unchecked. No additional seed or full test suite was run for diagnosis.

Reproduce with:

```bat
python scripts/diagnose_coupled_peaks.py --run results/coupled_particle_full_20260922T204227948958Z
```

Before changing optimization, retain this baseline and agree on one controlled
change. Do not silently increase the budget or change acceptance criteria.
Code, tests and this summary are versioned. Ignored run directories and model
checkpoints need separate backup; pushing Git does not back them up.
