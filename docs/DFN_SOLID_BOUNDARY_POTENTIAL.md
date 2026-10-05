# Joint solid-current boundary representation

## Scope and evidence

Command: `python scripts/check_solid_boundary_potential.py`.
Observed: **10 passed in 7.31 s**. Analytic fixtures only; no optimizer,
simulation, checkpoint change or full DFN validation.

`SolidBoundaryPotential` is a new isolated candidate. It is not part of a
DFN model factory, trainer or historical checkpoint loader. It is a separate
representation intervention, not a continuation of the amplitude-only test.

## Construction

Let X=x/L be the global coordinate, w the electrode width divided by L,
and xi=(X-left)/w the local coordinate. Let A=I_ref*L/(sigma_eff*phi_ref),
with positive reference current and constant effective solid conductivity.
The known potential contribution is A*w*q(xi), with

- Negative electrode: q=-xi+xi^2/2.
- Positive electrode: q=-xi^2/2.

Since i_s/I_ref = -d(q)/d(xi), the negative current decreases from one at
the collector to zero at the separator. Positive current increases from zero
at the separator to one at the collector. The same positive through-cell
current convention applies to both electrodes.

The learned correction evaluates a raw smooth field at
`X_warped = left + w*(3*xi^2 - 2*xi^3)`.
The derivative of this coordinate map vanishes at both endpoints. Thus the
correction can change the potential and its interior curvature without adding
endpoint current. The negative correction subtracts its value at the left
collector, enforcing the single zero-potential gauge there. The positive
potential offset remains learnable; no second gauge is imposed.

For the negative electrode:
`phi_hat = A*w*(q + raw(X_warped,tau) - raw(left,tau))`.

For the positive electrode:
`phi_hat = base + A*w*(q + raw(X_warped,tau))`.

The raw field must accept global [X,tau] and produce an unscaled scalar
correction. Do not accidentally reuse a Field with an already scaled output
and physical base without explicitly accounting for those transformations.

## Checks and limitations

- Both electrode signs, both endpoints, multiple times including zero and
  multiple raw parameter values preserve the prescribed currents.
- Boundary-current parameter derivatives vanish to numerical tolerance.
- The negative gauge stays zero; a positive potential offset remains free.
- A constant raw correction gives zero solid-charge residual for a uniform
  compatible volumetric reaction source.
- A nonuniform correction deliberately produces a nonzero local charge
  residual, with a nonzero parameter gradient and finite time derivatives.
- Invalid geometry, electrode labels and negative-gauge base are rejected.

These tests use polynomial raw fields, not a trained neural network. Smooth
coordinate warping can affect conditioning and approximation near endpoints;
no claim of universal approximation, improved optimization or correct initial
algebraic state is established. The initial concentration constraint is not
addressed by this module. This version handles only fixed positive current;
it must not be used silently for rests, ramps or current reversal.

Endpoint currents constrain the integrated solid-current divergence. Local
charge conservation still requires a compatible reaction source throughout
the electrode; a learned source is not forced to satisfy that requirement by
this representation. Electrolyte equations, kinetics, particle flux and all
other DFN conditions remain unresolved.

## Next bounded step

Wire an explicit experimental variant with raw neural corrections, constructor
metadata and strict checkpoint replay. Verify neural parameter gradients and
the unchanged physical loss inventory with a minimal smoke run before any
matched 200-step comparison. Preserve the original and amplitude-only controls;
do not silently change their checkpoints, kinetics, sampling or loss weights.
