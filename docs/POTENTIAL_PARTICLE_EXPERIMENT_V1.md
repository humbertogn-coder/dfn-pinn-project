# Potential-Driven PINN V1

Status: model and loss setup implemented, not trained. Configuration is
configs/potential_particle_v1.json. Reference problem and convergence checks
are documented in POTENTIAL_PARTICLE_REFERENCE.md. No old checkpoint or
manufactured reference is reused as training data.

## Model and Training Contract

Learn normalized concentration and signed current with two independent 32x32
tanh networks, CPU float64, seed 42. Let s=t/10 and rho=r/R:

    c_hat = sigmoid(logit(0.5) + s^2*N_c(rho^2,s))
    j = 0.1*s^2*N_j(s)       [A/m2]

This imposes uniform initial concentration, center symmetry, initial current
zero and initial concentration time derivative zero. It does not impose the
sign of current for t>0 or guarantee boundary flux accuracy. Saturation at
exact concentration endpoints is an error, not silently clipped. Time in the
existing PDE residual remains tau=t/3600; s=360*tau retains its graph.

Potential follows the fixed half-cosine 5 mV protocol, with ce=1000 mol/m3,
T=298.15 K and phi_e=0 V. The potential is NOT computed from learned surface
concentration, and no reference-current labels enter training. Mode='raw'
is required to match the finite-volume reference. Changing that mode raises
an error rather than silently changing the problem.

Reuse the already tested diffusion, flux, BV and inventory losses. Inventory
is a soft residual based on the same learned current, with time/radial GL
orders 32/64. Loss weights: PDE=1, flux=1, kinetics=10, inventory=1. This
choice carries forward the controlled manufactured result; it is not evidence
that this new case will converge. Architecture, budget and thresholds are
fixed before training, with no automatic retuning.

Planned budget: Adam 2000 steps at 1e-3, then LBFGS max_iter=300 and
max_eval=375 with strong Wolfe. Seeded fixed samples: 512 interior points,
128 positive boundary times, 32 inventory times. The existing sampler is
reused; there are no reference labels. Only focused setup tests have run.
Dedicated training and reference-aware audit entry points are now connected.
The manufactured-case auditor must
NOT be used unchanged because its reference and sign rules differ.

## Prespecified Acceptance

All criteria must pass; a smoke run can never qualify. Use the saved tight
320-cell reference with verified settings and artifact hashes. Compare PINN
concentrations at native reference centers and surface separately; do not
extrapolate center values to infer the surface. Include every saved output
time (uniform and logarithmic startup). Any additional interpolated comparison
must document interpolation and remain distinct from native-sample results.

| Metric | Limit |
| --- | ---: |
| Maximum internal concentration error | 1e-5 |
| Maximum surface concentration error | 1e-5 |
| Maximum current error [A/m2] | 1e-3 |
| Maximum surface-flux residual [A/m2] | 1e-3 |
| Maximum BV residual [A/m2] | 1e-3 |
| Maximum inventory error | 1e-6 |
| Initial profile error | 1e-12 |
| Physical normalized PDE RMS | 1e-2 |
| Independent mean-quadrature gap | 1e-7 |

These concentration limits are tighter than the manufactured benchmark because
the physical response is small (about 3e-4 surface stoichiometry variation).
The finest-mesh surface difference is about 4.3e-7, smaller than the proposed
1e-5 limit, but is not a rigorous reference error bound.

Check both predicted current and concentration-derived outward flux >=-1e-6
A/m2 at positive sampled times. This sign tolerance is an absolute bound,
not a substitute for the separate current/flux accuracy criteria. Strict
positivity and relative error at equilibrium are inappropriate: j(0)=0.
Report negative excursions without clipping. Report instantaneous relative
current error only where abs(j_reference)>1e-4 A/m2, as an auxiliary diagnostic.

Use independent radial orders 128/256 and current integral orders 64/128 for
inventory; require each resulting mean difference <=1e-7. Physical-volume/time
PDE RMS uses GL orders 64/128 in both coordinates with gap <=1e-4. Report
sampled PDE maximum, extrema locations, concentration bounds, current range,
and flux range. Refine residual time peaks independently of reference labels;
additional reference-error refinement needs a converged reference evaluation,
not a claim based solely on interpolating saved samples. All fields must be
finite and stoichiometry must stay strictly inside (0,1).

## Verification and Next Step

Run `python scripts/check_potential_particle.py` for four focused setup tests:
hard initial/center conditions, independent potential protocol agreement,
gradients without reference access, and rejection of the wrong kinetic mode.
No optimizer step or new reference simulation is performed by this command.

## End-to-End Smoke Verification

Entry points: scripts/train_potential_particle.py and scripts/audit_potential_particle.py.
The original manufactured trainers remain unchanged to preserve their recorded
source hashes. The new trainer uses the same optimization flow with the new
model/configuration and pins the reference report/array hashes before training.
Reference arrays are not consumed by any loss. The auditor verifies those
hashes, source hashes, checkpoint configuration and replayed training losses.
It also compares the applied potential with the stored reference protocol.

One smoke run completed: results/potential_particle_smoke_20260923T205630735781Z.
It used 3 Adam steps, LBFGS max_iter=2/max_eval=4, 16 interior points, 8 boundary
times and 4 inventory times, with unchanged architecture/quadrature orders.
Saved concentration and current predictions replayed exactly. Its audit is
audit_20260923T205701902376Z/audit.json within that run.

Status: SMOKE_DIAGNOSTIC_ONLY. Maximum internal concentration error 1.077830e-2,
surface error 1.079841e-2, current error 7.024263e-2 A/m2, flux residual
2.991234e-1 A/m2, BV residual 6.933023e-2 A/m2, inventory error 6.213891e-3,
and normalized physical PDE RMS 3.553139. Initial error is zero, but negative
current/flux excursions occur. This is not a physical acceptance result.

Only two new auditor tests were run, covering equilibrium sign tolerance,
smoke exclusion and invalid/missing metrics. No full suite, new finite-volume
reference solves or full-budget training were run in this step.

```bat
python scripts/train_potential_particle.py --smoke
python scripts/audit_potential_particle.py --run results/potential_particle_smoke_20260923T205630735781Z
```

The next bounded step is one full-budget trial (`--full`) followed by the same
auditor. Do not add open-ended seed studies. After the variable-current check,
move to electrolyte transport and charge equations, the next full-DFN components.

## First Full-Budget Result

The owner executed the unchanged full budget. Run:
results/potential_particle_full_20260923T224612698817Z.
Audit: audit_20260923T225339851931Z/audit.json within that run.
Source/reference hashes and final training losses were verified. No additional
training, reference solves or test suite were run during this audit.

Status: SAMPLED_TARGETS_FAIL.

| Metric | Result | Limit | Outcome |
| --- | ---: | ---: | --- |
| Internal concentration error | 5.903977e-5 | 1e-5 | FAIL |
| Surface concentration error | 5.920227e-5 | 1e-5 | FAIL |
| Current error [A/m2] | 1.768280e-4 | 1e-3 | PASS |
| Surface flux residual [A/m2] | 6.851240e-4 | 1e-3 | PASS |
| BV residual [A/m2] | 1.781925e-4 | 1e-3 | PASS |
| Inventory error | 3.862775e-6 | 1e-6 | FAIL |
| Physical normalized PDE RMS | 2.268134e-2 | 1e-2 | FAIL |

Initial error is zero; sign, admissibility and quadrature checks pass. Internal
concentration error peaks at rho=0.9984375, t=6.24 s; surface error peaks at the
same time. Current error peaks at 2.45 s, BV residual at 2.46225 s and flux
residual at 10 s. The maximum sampled PDE residual is 0.1178467.

The maximum predicted concentration is 0.5000275947, slightly above its initial
value 0.5. This overshoot is retained, not hidden by the weaker (0,1)
admissibility check. Current accuracy alone does not establish diffusion or
inventory accuracy. No threshold, budget or loss weight was changed.

Before any further optimization, examine the spatial/time distribution of
the diffusion residual and inventory error on this frozen checkpoint. This
is a diagnostic suggestion, not authorization for another run or a claim
that the variable-current case is validated.

### Frozen Diffusion Diagnosis

The existing full checkpoint was inspected without training or reference solves.
Script: scripts/diagnose_potential_diffusion.py.
Report: diffusion_diagnosis_20260923T230629070717Z/report.json inside the run.
A 201x401 radius/time grid and local 41x81 patch found a signed normalized
PDE residual of -0.1178475201 at rho=1, t=4.108125 s. At that point the time
derivative is -0.1620174035 and diffusion term is -0.04416988336.

Independent 128x128 GL volume/time quadrature attributes the integrated squared
residual as follows: rho in [0,0.8): 14.69%; [0.8,0.95): 20.33%; [0.95,1]:
64.98%. The outer 5% of radius occupies about 14.26% of spherical volume.
These are quadrature estimates of squared-residual shares, not concentration
error shares or proof of the optimizer's failure mechanism.

The inventory error peaks at 6.49 s. Predicted mean minus the learned-current
inventory target is -3.86277547e-6. At the same instant, predicted mean minus
the FV reference mean is -3.91056439e-6, whereas the current-derived target
minus reference mean is only -4.77889225e-8. This points primarily to the
concentration field for that peak; it does not prove this attribution at all
times. Training inventory MSE is 5.075166e-12 versus PDE MSE 2.178809e-4,
so a unit inventory loss weight does not imply comparable optimization influence.

A bounded next hypothesis is to enrich interior PDE samples near the surface
while keeping the total point count and all weights/budgets fixed. That changes
the sampling measure, so any comparison must use the unchanged independent
physical-volume/time auditor. Do not simultaneously retune inventory weights.
No new training is authorized or performed by this diagnostic record.

### Authorized Surface-Sampling Trial

The owner subsequently authorized one controlled trial. The dedicated frozen
trainer scripts/train_potential_surface.py maps the last 128 of 512 original
radial coordinates by rho_new=0.95+0.05*rho_old. The first 384 samples and all
times, boundary points, inventory points, configuration, loss weights and
optimizer budgets remain identical. Both saved checkpoints were compared to
verify that change exactly. The initializer and seed are unchanged.

A syntax error in the new output-directory string was corrected before any
optimization began. Then one full-budget run completed; no smoke run, seed
sweep or full test suite was performed. A focused sampling-contract check
passed before execution.

Run: results/potential_particle_surface_full_20260923T231451837683Z.
Audit: audit_20260923T231602654995Z/audit.json in that run.
Status: SAMPLED_TARGETS_FAIL.

| Metric | Original sampling | Surface-enriched sampling |
| --- | ---: | ---: |
| Internal concentration error | 5.903977e-5 | 3.993327e-5 |
| Surface concentration error | 5.920227e-5 | 2.429558e-5 |
| Current error [A/m2] | 1.768280e-4 | 1.428202e-4 |
| Flux residual [A/m2] | 6.851240e-4 | 1.301788e-3 |
| BV residual [A/m2] | 1.781925e-4 | 1.433461e-4 |
| Inventory error | 3.862775e-6 | 1.669821e-5 |
| Physical normalized PDE RMS | 2.268134e-2 | 2.109932e-2 |

Surface concentration and current improved, but inventory degraded by about
4.3x and flux now exceeds its 1e-3 A/m2 limit. Concentration and PDE targets
still fail. Sign and initial-state checks pass. This trial is not an accepted
replacement for the original baseline. Both checkpoints are retained.
This negative/mixed result does not justify further unscheduled tuning or
relaxing thresholds. No additional run was launched after the audit.

### Loss-Scale Diagnosis Without Optimization

scripts/diagnose_potential_loss_scales.py was run on both frozen checkpoints.
It reproduces their training losses, computes each weighted term's parameter
gradient separately, and verifies that no model state changed. No optimizer
step, additional simulation or broad test suite was run.

| Concentration-network gradient norm | Original | Surface sampling |
| --- | ---: | ---: |
| PDE | 4.200650e-2 | 8.438441e-2 |
| Surface flux | 4.319181e-2 | 1.388210e-2 |
| Kinetics (weight 10) | 1.669403e-4 | 9.770559e-5 |
| Inventory (weight 1) | 8.558278e-7 | 5.955210e-6 |

Inventory gradients are about 49083x and 14170x smaller than PDE gradients
on the concentration network. Inventory gradients on the current network
are also small: 2.048222e-10 and 1.324313e-9, respectively. These are local
Euclidean parameter-gradient norms at the final models, not optimizer update
sizes, gradient directions, training-history measurements or proof of causality.

Reports are loss_scale_diagnosis_20260923T232143768291Z/report.json inside the
original run and loss_scale_diagnosis_20260923T232205188874Z/report.json inside
the surface-sampling run. Threshold-scaled MSE values in those reports are
illustrative only: training MSE and independently sampled maxima are different
statistics. They must not be used as acceptance checks or automatic weights.

A possible single-variable experiment is to return to ORIGINAL uniform sampling
and change only the inventory MSE coefficient from 1 to 1e4, preserving the
kinetics coefficient 10, same seed and full budget. At the frozen original
checkpoint this would scale the inventory concentration gradient to about
8.56e-3, still below the PDE gradient; that is a starting hypothesis, not a
guarantee elsewhere in training. Do not also change sampling or thresholds,
and do not adopt the surface-sampling run as a new accepted baseline.
This experiment has NOT been executed or scheduled by the diagnosis.

### Authorized Inventory-Weight Trial

The owner subsequently authorized exactly one uniform-sampling run with
inventory weight 10000. Script: scripts/train_potential_inventory.py. Original
trainer and configuration files were preserved; the override is saved in the
new checkpoint/report. Compared checkpoints confirm all training points are
identical and the inventory coefficient is the only configuration change.

Run: results/potential_particle_inventory10000_full_20260923T233227007613Z.
Audit: audit_20260923T233342430862Z/audit.json inside that run.
Status: SAMPLED_TARGETS_FAIL. Same seed, budget and acceptance thresholds.

| Metric | Original weight 1 | Inventory weight 10000 |
| --- | ---: | ---: |
| Internal concentration error | 5.903977e-5 | 5.826769e-5 |
| Surface concentration error | 5.920227e-5 | 5.842950e-5 |
| Current error [A/m2] | 1.768280e-4 | 1.786752e-4 |
| Flux residual [A/m2] | 6.851240e-4 | 7.171596e-4 |
| BV residual [A/m2] | 1.781925e-4 | 1.799639e-4 |
| Inventory error | 3.862775e-6 | 3.543020e-6 |
| Physical normalized PDE RMS | 2.268134e-2 | 2.271672e-2 |

Inventory improved only about 8.3%; it still exceeds 1e-6. Concentration and
PDE also still fail, while current/flux/BV remain within their limits. This
does not validate the hypothesis that weak inventory weighting was the main
bottleneck. Frozen-checkpoint gradient norms were not predictive of a large
training improvement. No additional weights, seeds or budget extensions were
tried; no full test suite was run. Retain all baselines and negative results.

Do not continue an open-ended weight search. Any next step should explicitly
review the concentration representation/optimization assumptions or separate
development of the remaining DFN operators from claims of particle validation.
The variable-current PINN remains unvalidated against its stated criteria.
