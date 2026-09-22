# Differentiable Interfacial Charge

`CurrentIntegral` maps queries (x/L,tau) to physical Q in C/m2 active surface,
where tau=t/t_ref and its current model returns j in A/m2 active surface:

    Q(x,tau) = t_ref*tau*sum_i w_i*j(x,tau*s_i)

Fixed Gauss-Legendre nodes s_i and weights w_i cover [0,1]. The default is 32
nodes. Both the moving nodes tau*s_i and the prefactor tau stay in autograd.
Do not detach the integral or the upper limit. The time scale is a fixed
buffer, not a learned parameter. Initial charge is zero for finite current at
zero; the right derivative is retained. Model output is (N,1) and points are
(N,2). Use pointwise networks, not batch normalization or attention across
unrelated queries. Float64 CPU is the tested research configuration.

Feed this charge to `particle_mean_from_charge`. For a sufficiently resolved
smooth current, the resulting mean's tau derivative matches
`particle_mean_rate` applied to the same instantaneous current. Spatial and
network-parameter derivatives also flow through the integral.

This is numerical quadrature, not exact integration. Tests deliberately show
that a one-node rule can produce wrong charge AND wrong endpoint derivative
for a quartic current despite a valid autograd graph. Test refinement before
using learned currents with sharp temporal features. Discontinuous protocols
require a separately designed split integration scheme; it is not implemented.
The current must be finite at zero; non-smooth/singular derivatives there are
not automatically regularized. No clipping or sign constraints are applied.

Run `python scripts/check_current_integral.py`. Audits cover positive/negative
and zero constants, quadratic and exponential time dependence, x dependence,
zero-time limits, first/second input derivatives, parameter gradients and the
inventory chain rule for both particle scales. This connects charge to the
inventory helper, not yet to Butler-Volmer or a coupled trained network.

The same current function must eventually drive this integral, the particle
surface boundary condition and the kinetics residual. A projected current
requires a callable that recomputes its spatial correction at every time
quadrature node; independent pointwise raw values are not a substitute.
