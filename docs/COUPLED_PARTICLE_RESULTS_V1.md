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

## Controlled Kinetics-Weight Experiment

Owner authorized one run changing only the kinetics loss weight from 1 to 10.
The original trainer was preserved byte-for-byte for baseline hash verification.
`scripts/train_coupled_kinetics10.py` is a frozen copy with only the weight
override and separate output prefix. This duplication preserves reproducibility;
it is not a second evolving general-purpose trainer.

Run: results/coupled_particle_kinetics10_full_20260922T235420827096Z.
Audit: audit_20260922T235601341245Z/audit.json within that run.
Checkpoint configurations were compared: only loss_weights.kinetics changed.
All frozen training point tensors were exactly equal to the baseline. Same
seed, architecture, optimizer settings, step budget and acceptance limits.

| Metric | Baseline | Kinetics weight 10 |
| --- | ---: | ---: |
| Maximum concentration error | 5.519567e-6 | 3.440773e-6 |
| Maximum current error [A/m2] | 1.422361e-3 | 5.748795e-5 |
| Maximum flux residual [A/m2] | 9.782961e-4 | 6.932697e-5 |
| Maximum BV residual [A/m2] | 1.422630e-3 | 5.748795e-5 |
| Maximum inventory error | 3.573574e-6 | 1.118117e-6 |
| Physical normalized PDE RMS | 8.249065e-3 | 2.752158e-3 |

Status: SAMPLED_TARGETS_PASS. Initial profile error remains 1.110223e-16;
minimum current is 0.09994251 A/m2 and minimum outward flux is 0.09993426 A/m2.
The peak sampled PDE residual is 0.01614297 (reported, no maximum threshold).
No extra seeds, budget extension or full test suite were run. This is one
controlled manufactured case, not seed robustness, continuous-time certification
or full DFN validation. Training loss totals with different weights should not
be compared as a physical accuracy metric. No automatic further experiment is
authorized by this result.

Reproduction (training is a new full run, not needed to inspect saved results):

```bat
python scripts/train_coupled_kinetics10.py --full
python scripts/audit_coupled_particle.py --run results/coupled_particle_kinetics10_full_20260922T235420827096Z
```

### Frozen-Model Temporal Follow-Up

The owner requested the next step. The existing peak diagnostic was applied
to the weight-10 checkpoint, without retraining or additional seeds. Uniform
1001/2001-point time grids plus 281 logarithmic startup times both found the
largest current and BV errors at the earliest sampled time, t=1e-8 s:

- Signed current error: -5.754060199e-5 A/m2.
- Signed BV residual: -5.754060199e-5 A/m2.
- Learned current: 0.09994245940 A/m2.
- BV-implied current and concentration-derived outward flux: approximately
  0.1 A/m2 at this sample.

The refined maximum is slightly larger than the original audit value
5.748795e-5, but remains below the unchanged 1e-3 A/m2 limit (about 5.75%
of the limit). This confirms the sampled current/kinetics improvement, not
an exhaustive bound on all residuals. The earlier audit retains responsibility
for concentration, PDE, inventory and flux diagnostics; this follow-up did not
repeat them. Times below 1e-8 s and continuous-time maxima remain unchecked.

```bat
python scripts/diagnose_coupled_peaks.py --run results/coupled_particle_kinetics10_full_20260922T235420827096Z
```

The weight-10 checkpoint can now serve as the working single-seed manufactured
coupling baseline, with the above limitations. No further tuning is justified
solely by these accepted sampled errors. A next scientific experiment should
change the physical problem (for example, a prescribed potential protocol with
time-varying reaction current and an independently converged numerical reference),
not claim full DFN validation or infer robustness from this one case. Such an
experiment requires a new specification before training.
