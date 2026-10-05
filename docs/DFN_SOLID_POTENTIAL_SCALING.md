# Solid potential amplitude check

## Scope and evidence

An isolated candidate changes only the output amplitude of each solid-potential
Field. It is not installed in a DFN trainer or checkpoint loader. Existing
architectures, global X=x/L coordinates, time features, base potentials,
kinetics, loss weights and historical runs remain unchanged.

Command: `python scripts/check_solid_potential_scaling.py`

Observed result: **8 passed in 10.94 s**. No training or reference solve ran.

## Theory in simple terms

Solid current is i_s = -sigma_eff * d(phi_s)/dx. With a normalized potential
phi_s/phi_ref = base + A*f(X,tau), its normalized current is

`i_s/I_ref = -[sigma_eff*phi_ref/(L*I_ref)] * A * df/dX`.

Choose `A = I_ref*L/(sigma_eff*phi_ref)` using the positive reference-current
magnitude and the total through-cell length L, not the electrode length.
Then `i_s/I_ref = -df/dX`. This removes the fixed conductivity multiplier
from this particular map between the raw network slope and current.
It does not force any current, boundary condition, gauge or conservation law.

For the current settings, A is approximately 0.00152299 for the negative
electrode and 1.81912730 for the positive electrode, instead of 0.01 for both.
The old raw slopes required for unit normalized current are approximately
-0.152299 and -181.912730, respectively; the candidate requires -1 in both.

## What was checked

- Manufactured quadratic potentials yield the expected linearly varying
  currents, zero separator currents and unit collector currents in both
  electrodes, with the appropriate opposite reaction-source signs.
- Their solid charge residuals vanish to numerical tolerance, and their
  collector-current derivative with respect to source strength is one.
- Paired fresh networks with identical state dictionaries differ only by
  the prescribed amplitude ratio in potential, current and parameter gradients.
- First and second time derivatives remain finite; invalid conductivities fail.

The manufactured polynomials are test fixtures, not built-in constraints or
claims that the neural network has learned these profiles.

## Limitations and next decision

This rescaling also changes potential values away from the base and therefore
the initial kinetics residuals. It does not repair saturated hidden layers in
old checkpoints, prove better optimization, or validate a coupled DFN solution.
The tests do not show that amplitude was the sole cause of prior failures.

Field amplitude is constructor metadata, not a state_dict tensor. Any future
training variant must explicitly persist and verify its scale configuration;
loading old weights with a different amplitude is not historical replay.

Next bounded step: a matched short fresh-initialization probe comparing the
existing and scaled solid-potential amplitudes, keeping raw weights, current
variant, samples, budget, kinetics and loss weights fixed. Report all initial
and final residuals, including possible kinetics regressions. Do not start a
full-budget run or infer success from analytic checks alone.
