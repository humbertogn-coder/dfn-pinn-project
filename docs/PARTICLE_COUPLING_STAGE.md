# Particle-to-Kinetics Coupling Stage

This new stage follows the owner's explicit authorization after uploading the
closed pilot (Git commit e205650). It does not reopen or erase pilot findings.
No long training, Grace job or coupled DFN accuracy result is part of this step.

## Implemented First Step

`dfn_pinn.particle_balance` maps signed interfacial current and accumulated
charge into the local particle mean. The existing surface condition is
-D dc/dr = j/F, with j in A/m2 of active particle surface. Volume integration
gives d(mean(c))/dt = -3*j/(F*R). For c_hat=c/c_ref and tau=t/t_ref:

    d(mean(c_hat))/d(tau) = -3*t_ref*j/(F*R*c_ref)
    mean(c_hat) = initial_mean - 3*Q/(F*R*c_ref)
    Q(x,t) = integral_0^t j(x,s) ds       [C/m2, physical seconds]

Positive j removes solid lithium and negative j inserts it. Diffusivity cancels
from the integrated balance but remains in the diffusion PDE and surface
gradient. Neither cell current I [A] nor volumetric a*j [A/m3] is a valid
substitute for j. Local inventory depends on x-dependent accumulated current,
not solely the electrode-integrated current projection.

The functions preserve autograd, impose no clipping, and do not integrate j or
enforce concentration admissibility. Tests cover both Chen2020 particle scales,
zero/both current signs, the normalized-time chain rule, physical charge units,
parameter gradients, and consistency with the existing surface residual.
Run `python scripts/check_particle_balance.py`.

Verification: all 12 focused cases and all 185 repository tests passed in the
local dfn-pinn environment. This verifies the inventory helper and its unit/sign
contract, not a coupled model or training result. The new-stage files are local
changes; the preceding pilot closeout was uploaded separately as e205650.

## Remaining Before Training

The differentiable current-integral component is now implemented; see
`CURRENT_TIME_INTEGRAL.md`. It is tested on smooth prescribed/parameterized
currents. The shared-current interfaces below connect it to Butler-Volmer,
but a coupled learned-field training remains untested.

`dfn_pinn.particle_current.ParticleCurrent` now connects the inventory target
and the existing surface-flux residual to one registered current model.
`mean_target` accepts (x/L,tau) and a scalar or pointwise initial mean;
`flux_residual` accepts (rho=1,x/L,tau>0). The time scale comes from the same
ParticleScales object used for the surface residual. Units remain A/m2 of
active area for current and physical C/m2 for its integral.

Run `python scripts/check_particle_current.py`. Twelve focused cases cover
both Chen2020 scales, both signs, constant/ramp currents, independent radial
integration, time/parameter gradients, and initial-time handling. A quadratic
manufactured concentration satisfies mean and flux; only its constant-current
case is asserted to solve the diffusion PDE. The ramp construction is NOT a
diffusion solution. A deliberately uniform field verifies that exact inventory
can coexist with a nonzero surface-flux residual.

This adapter supplies a target, not a hard concentration projection, trained
model, or electrochemical feedback loop. Smooth pointwise current and resolved
time quadrature remain requirements. If current is spatially projected, its
callable must recompute that correction at each queried time. Initial mean
does not impose the entire initial concentration profile or positivity.

The SAME registered current is now also used by `kinetics_residual`:

    residual = (j - 2*j0*sinh(F*(phi_s-phi_e-U)/(2*R*T)))/j_ref

Inputs are (N,1) fields on (N,2) [x/L,tau] points: surface stoichiometry
c_s_surface/c_s_max, electrolyte concentration in mol/m3, potentials in V,
and temperature in K. Stoichiometry must be strictly inside (0,1); no clipping
or endpoint extension is added. The exchange-current mode must be explicitly
chosen as `raw` or `pybamm_26_8`. Potentials use a common gauge. The adapter
does not infer surface values from a concentration network: callers must supply
the actual surface evaluation, with its graph intact, not an independent
unconstrained surface field. Arbitrary particle c_ref is not c_s_max.

Run `python scripts/check_particle_kinetics.py`. Fifteen focused tests cover
both electrodes, both explicit fit modes, positive/negative/zero current,
manufactured inverse-BV consistency, potential perturbations, independent
field first/second derivative checks, and current-parameter gradients shared
with the inventory target. These use existing audited constitutive functions;
they are interface checks, not a new independent physical reference.

This is a residual formulation with current as a shared unknown, not an
implicit algebraic solve. BV-derived current must not recursively depend on
its own inventory projection. Electrode-integrated current enforcement and
the complete learned concentration/potential feedback are still unassembled.
The surface-concentration/kinetics interface is now assembled by
`dfn_pinn.particle_interface.particle_interface`. It evaluates the actual
concentration model at rho=1, converts c_hat using c_ref/c_s_max, and passes
that surface value to kinetics without detaching. A pointwise field callable
supplies SI c_e, phi_s, phi_e and T. It returns surface concentration,
stoichiometry, flux residual and kinetics residual. Models must be deterministic
and pointwise; repeated concentration evaluations must represent the same field.

Run `python scripts/check_particle_interface.py`. Eleven focused cases test
both electrodes/signs, c_ref equal to or different from c_s_max, and a
manufactured quadratic radial profile with constant-in-time current. That
profile satisfies diffusion, center symmetry, flux and the integrated mean;
potentials are manufactured using inverse BV. Its initial profile is NOT
uniform, and it is not a Chen2020 discharge simulation. Independent radial
quadrature checks the mean. Frozen potentials and deliberately perturbed
surface concentrations test nonzero feedback gradients without an artificial
inverse-BV cancellation. These checks do not establish training convergence.

The assembly does not impose inventory, initial data, positivity or charge
transport. A coupled training formulation still needs these constraints and
a well-posed choice of prescribed versus learned fields and gauge. No training
or Grace job was run in this step.

The previous targets 1-3*t and 1-75*t^2 must not be reused
as inventories for an unknown learned electrochemical current.

Preserve the pilot's early wrong-sign boundary flux as a regression concern.
Exact global inventory alone is not local flux enforcement or positivity.
Only after those interface checks should a small coupled training experiment
be considered. Electrolyte transport, charge equations, identifiability and
full DFN validation remain separate unfinished work.

## First Training Specification

The first bounded experiment is now specified in
`COUPLED_PARTICLE_EXPERIMENT_V1.md` and `configs/coupled_particle_v1.json`.
It learns concentration and current for a single manufactured negative particle
under a fixed potential protocol. Acceptance targets are fixed before training.
The analytic preflight can be run with `python scripts/check_coupled_particle_spec.py`.
The trainer is now implemented, and a bounded smoke run completed with finite
gradients and exact saved-model replay. Run `python scripts/train_coupled_particle.py --smoke`.
The independent auditor is now implemented and was run on the existing smoke
checkpoint. It found failed physical targets and retained SMOKE_DIAGNOSTIC_ONLY.
No full-budget run has been completed. See COUPLED_PARTICLE_EXPERIMENT_V1.md
for results, audit scope and the explicit reproduction command.
The earlier closed pilot limitations remain unchanged.
