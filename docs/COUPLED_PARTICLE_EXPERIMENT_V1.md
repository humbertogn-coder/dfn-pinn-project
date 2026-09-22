# First Coupled Particle Experiment

Status: specified, not trained. Configuration: configs/coupled_particle_v1.json.
This is a new manufactured interface benchmark, not a reopening of the closed
pilot or a realistic uniform-initial-state discharge. No Grace job is needed.

## Unknowns and Prescribed Data

One negative-electrode particle, constant Chen2020 D and radius, t in [0,10] s.
Learn c_hat(rho,tau) and signed j(tau) jointly using separate pointwise networks.
There is no x dependence; adapters may receive x/L=0 but networks must ignore x.
Fix c_e=1000 mol/m3, T=298.15 K, phi_e=0 V (gauge), and phi_s(t) below.
Do not learn both potentials without charge equations. No electrolyte, cell
voltage, applied-cell-current, parameter inference or full DFN claim is made.

## Manufactured Reference and Driving Potential

Let j_star=0.1 A/m2, c_ref=c_max, m0=0.5, tau=t/3600 and

    b = -j_star*R/(2*F*D*c_ref)
    k = -3*3600*j_star/(F*R*c_ref)
    c_star(rho,tau) = m0 + k*tau + b*(rho^2 - 3/5)
    phi_s(t) = U(c_star(1,tau))
               + inverse_BV(j_star, j0(c_e,c_ref*c_star(1,tau),T), T)

This compatible NONUNIFORM initial profile avoids a current-step corner.
It satisfies spherical diffusion, center symmetry, flux and mean inventory.
Construct phi_s exclusively from this fixed reference, never from the learned
surface concentration: doing that would make BV a tautology. Reference current
is used for constructing the problem and validation, NOT as a supervised
current target. No concentration reference samples enter the training loss.

## Proposed Model and Loss

Two 32x32 tanh networks, float64 CPU, seed 42. Use inputs rho^2 and s=t/10
for concentration, s alone for current. Derivatives in residuals remain with
respect to rho and tau=t/3600; retain the s=360*tau chain rule.

Planned hard initial ansatz:

    c_hat = sigmoid(logit(c_star(rho,0)) + s*N_c(rho^2,s))
    j = j_ref*N_j(s)

This enforces the initial profile and center symmetry and permits both current
signs. Do not clip concentrations or current. Numerical sigmoid saturation at
an endpoint must fail explicitly under the existing strict kinetics checks.
The networks are implemented in dfn_pinn.coupled_training.

Use mean-square PDE, surface flux, BV and inventory residuals with fixed unit
weights. PDE normalization is c_ref/t_ref (existing diffusion_residual);
flux and BV divide by j_ref=0.1 A/m2; inventory is dimensionless c_hat.
Inventory is a SOFT constraint here: radial mean minus mean_target obtained
from the same learned current via CurrentIntegral. No recursive BV-defined
current and no synthetic 1-3*t inventory. Exact inventory projection is outside
this first experiment. Report all unweighted physical residuals as well.

Freeze seeded training points: 512 uniform (rho,s) interior samples, 128
uniform positive boundary times including s=1, and 32 positive inventory
times. Initial profile is imposed by construction, not sampled as a loss.
Gauss-Legendre orders: 32 for current time integral, 64 for radial mean.
Adam 2000 steps at 1e-3, then LBFGS max_iter=300, max_eval=375, strong Wolfe.
LBFGS evaluations are not equivalent to Adam steps. No automatic budget
extensions, seed sweep or threshold relaxation after observing results.

## Acceptance Before Training

Thresholds in the JSON are exploratory engineering targets fixed before this
run; they are not universal physical accuracy guarantees. Every criterion
must pass, not just the total loss. Check a held-out 201x401 uniform radius/time
grid including endpoints; evaluate surface/BV only for t>0. Refine locally
around concentration/current/flux peaks and retain the larger sampled error.
Evaluate the initial profile separately. Report sampled concentration bounds
and reject any nonfinite field or surface stoichiometry outside (0,1).

Concentration max error <=1e-3; current error and physical flux/BV residuals
<=1e-3 A/m2; mean error <=6e-5; initial profile error <=1e-12. Require positive
predicted current and positive concentration-derived outward flux at every
positive-time validation sample. The constant reference current has no zero
crossing, so instantaneous relative errors are well defined here.

Physical-volume/time RMS of the normalized PDE residual must be <=0.01,
computed using 3*rho^2 drho dt / duration, with independent GL orders 64 and
128 in both coordinates; require their absolute RMS gap <=1e-4. Check mean
and current-integral quadrature at orders 128/256 and 64/128 respectively;
each resulting dimensionless mean change must be <=1e-7. These are sensitivity
checks, not continuous bounds. Residual maxima must be reported even where
no acceptance limit was specified. Never infer local flux accuracy from mass.

Save source/config hashes, model and optimizer settings, frozen points, loss
components, wall time, independent metrics and predictions in a new run folder.
Reload the checkpoint and reproduce predictions before accepting a run. A
smoke run is implementation evidence only, never an accepted scientific run.

## Current Deliverable

The trainer is implemented in scripts/train_coupled_particle.py. It requires
an explicit --smoke or --full flag. Smoke uses 3 Adam steps, LBFGS max_iter=2
and max_eval=4, with 16 interior points, 8 boundary times and 4 inventory times.
Quadrature orders and architecture remain unchanged. Full uses the specified
budget. Both save frozen points, configuration, checkpoint, source hashes,
loss history and exact concentration/current checkpoint replay verification.
One CPU thread is used. LBFGS line-search closures are recorded in history;
max_eval is the optimizer setting, not a strict externally enforced call cap.

Run `python scripts/train_coupled_particle.py --smoke` for implementation checks.
The first smoke run completed with finite gradients and exact checkpoint replay.
It is NOT scientific acceptance. Full training has not been run. The JSON
status describes the frozen specification, not the status of individual runs.
Neither training mode automatically performs independent acceptance auditing.
A full run is explicitly marked trained_pending_independent_audit.

## Independent Auditor

Implemented in scripts/audit_coupled_particle.py. Select the run explicitly:

```bat
python scripts/audit_coupled_particle.py --run results/coupled_particle_smoke_20260922T203110583669Z
```

The auditor verifies recorded source hashes, matches checkpoint/report config,
and reproduces saved final training losses before evaluation. It uses a 201x401
grid, a 41x81 local concentration-error patch, refined boundary times near
current/flux/kinetics error peaks, physical PDE quadrature orders 64/128, radial
mean orders 128/256, and current-integral orders 64/128. Initial state is checked
separately; flux/kinetics and sign checks exclude t=0. Reported peaks remain
sampled estimates, not exhaustive or continuous bounds. The reference and
physical residual operators are shared with the previously audited code;
independence here refers to evaluation samples and quadratures, not a separate
implementation of the governing equations.

The smoke run above was audited without retraining. Maximum concentration
error was 6.399330e-3; current error 1.043890e-1 A/m2; inventory error
5.472492e-3; physical normalized PDE RMS 3.097992. Minimum current was
-4.388996e-3 A/m2 and minimum concentration-derived outward flux was
-5.426551e-2 A/m2. The initial profile error was 1.110223e-16. Several targets
fail; this is a smoke diagnostic, not an accepted physical solution.

Only two focused auditor tests were run for this change: an exact manufactured
solution plus invalid/sign/smoke safeguards, and threshold-failure handling.
No full test suite, training, or seed study was run. The first audit report is
outside the repository in the sibling Practica/coupled_audit_results directory;
normal invocations save a new timestamped audit folder inside the selected run.
Results/checkpoints still require separate backup; Git does not preserve them.
