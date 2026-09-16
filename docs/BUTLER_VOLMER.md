# Symmetric Butler-Volmer Primitives

Implemented in `src/dfn_pinn/kinetics.py`, using the constants already audited
for Chen2020. This is the one-electron, symmetric, full-utilization relation:

    j = 2*j0*sinh(F*eta/(2*R*T))
    eta = (2*R*T/F)*asinh(j/(2*j0))

Units: eta in V, j and j0 in A/m2 of active area, temperature in K.
Positive eta means positive oxidation current. Both electrodes use the same
formula; no electrode-specific sign flip is introduced.

The direct expression was checked against installed PyBaMM's
`interface/kinetics/butler_volmer.py`, `SymmetricButlerVolmer._get_kinetics`
with electron count and utilization equal to one. The inverse implemented here
is the algebraic inverse, not a port of any regularized PyBaMM inverse submodel.

Both functions require finite floating tensors, positive j0 and temperature,
and matching dtype/device. Broadcasting is allowed and preserves autograd.
No j0 floor, clipping, film correction or asymmetric transfer coefficient is
introduced. Choose the constitutive j0 mode explicitly upstream when matching
the working HDF5 reference.

## Numerical Scope

Direct sinh grows exponentially and can overflow; the ordinary inverse ratio
can also overflow for extreme j/j0. Nonfinite intermediates or outputs raise
FloatingPointError instead of silently clipping. This deliberately conservative
implementation is not a log-domain extension to every representable input.
Underflow and finite outputs with nonfinite derivatives remain possible at
extreme scales. Monitor outputs and gradients during future training.

Inverse BV is not automatically well conditioned: at j=0 its slope with respect
to j is R*T/(F*j0), which diverges as j0 approaches zero. Numerical equivalence
does not establish a training advantage.

Raw direct and inverse residuals have different units. Before comparing losses,
divide direct current residuals by j_ref and inverse voltage residuals by phi_ref,
as specified in the mathematical contract. Equal roots do not imply equal losses
or gradients away from the solution. Residual assembly and training remain pending.

## Reproduction

Run `python scripts/check_kinetics.py`. Tests include positive, negative and zero
current, both round trips, comparison to an independent PyBaMM expression,
analytic slopes, float64 gradcheck/gradgradcheck for all inputs, broadcasting,
float32 smoke checks, invalid inputs, a small-j0 example and explicit overflow.
No DFN simulation, training, data rewrite or Git operation is performed.
