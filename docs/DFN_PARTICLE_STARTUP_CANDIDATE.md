# Dimensionally scaled particle startup candidate

## Scope and evidence

Command: `python scripts/check_particle_startup_candidate.py`.
Result: **8 passed in 9.00 s**. No training, simulation, historical checkpoint
edit or full test suite ran. The candidate is not connected to a DFN trainer
or checkpoint factory. The earlier synthetic pilot remains closed.

The implementation reuses the square-root-time and surface-layer idea in
`scripts/train_constant_flux_particle.py`, but is a new separate module with
the DFN global x coordinate and each particle's physical scales. Historical
startup-pilot failures remain evidence against assuming that this idea alone
solves PDE residual or conservation problems.

## Construction and physical meaning

Inputs are rho=r/R, global X=x/L and tau=t/t_ref. Define

`Fo = D*t/R^2 = diffusion_number*tau`

`beta = j_ref*R/(F*D*c_ref)`

`layer = exp(-(1-rho^2)/(4*sqrt(Fo)))`

For positive time, the candidate is

`c_hat = c_hat_initial + beta*sqrt(Fo)*raw(rho^2, X, sqrt(Fo/Fo_end), layer)`.

At exactly zero time its value is explicitly c_hat_initial. This branch
specifies a value, not a right time derivative. Diffusion and surface-flux
derivatives are evaluated only for t>0. The rho-squared features enforce
zero radial gradient at the center for smooth pointwise raw fields.

The factor beta is the inverse of the existing normalized surface-flux
radial-derivative coefficient. The candidate therefore incorporates physical
current/diffusion scales rather than an arbitrary shared output amplitude.
It changes the feature set, time behavior and output scaling together as one
declared representation intervention, not an amplitude-only ablation.

## Why finite startup flux is representable

For the test fixture `raw = -2*s*layer`, where s=+1 or -1, the surface
derivative of c_hat is exactly -s*beta for every positive time. Consequently
`F*N_out = s*j_ref`, while the surface concentration change is
`-2*s*beta*sqrt(Fo)` and tends to zero as time tends to zero.

This demonstrates compatible limiting concentration and positive-time flux.
It does not impose this fixture on the neural network, nor prove the spherical
diffusion equation: the test deliberately verifies that this simple fixture
has a nonzero diffusion residual. A uniform initial profile has zero radial
derivative at t=0, so do not demand its classical derivative and nonzero
switched-on flux simultaneously at that corner.

## Focused checks

- Both Chen2020 particle scales and both outward/inward flux signs.
- Exact initial values at sampled radii and center symmetry.
- Correct physical flux for the analytic surface-layer fixture at times
  1e-6 through 1 second, including its parameter derivative.
- Neural first, second and mixed derivatives finite at positive sampled times,
  including the center, interior and surface, with usable parameter gradients.
- Near-zero-time concentration limit and invalid scale rejection.

Finite derivatives at the sampled times do not imply uniform bounds as t->0.
The surface layer may produce large startup PDE residuals, as earlier pilot
experiments already illustrated. The candidate has no hard surface flux,
inventory projection, concentration positivity/bounds, or guaranteed solution.
Any training implementation must monitor concentration range and inventory;
silently clipping concentrations would change the diagnostic.

## Next bounded step

Create explicit candidate checkpoint metadata and minimal neural training/replay
verification for the frozen-current particle subproblem. Keep the existing
current function and physical residual definitions unchanged. Before a matched
200-step comparison, specify initialization fairly: the candidate has four
inputs rather than three, so historical particle weights cannot be loaded
unchanged. A fresh paired baseline/candidate experiment is distinct from the
previous warm-start diagnostic. Preserve that failed result and report all
representation and initialization differences. Do not extend full DFN training.
