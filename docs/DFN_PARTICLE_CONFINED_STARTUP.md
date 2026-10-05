# Surface-confined startup and linear-time bulk

## Bounded step completed

Command: `python scripts/check_particle_confined_startup.py`.
Result: **6 passed in 9.74 s**. No training, simulation or checkpoint changes.
Historical startup and DFN source files remain unchanged.

The new `ParticleConfinedStartup` reuses the physical scales and the four-input
raw network interface from the previous candidate. It evaluates the same raw
network twice, so no additional network parameters are introduced.

## Representation

Use Fo=D*t/R^2, beta=j_ref*R/(F*D*c_ref), and
`layer=exp(-(1-rho^2)/(4*sqrt(Fo)))` for positive time.

`bulk_raw = raw(rho^2, X, Fo/Fo_end, 0)`

`surface_raw = raw(rho^2, X, sqrt(Fo/Fo_end), layer)`

`c_hat = initial + beta*Fo*bulk_raw + beta*sqrt(Fo)*layer*surface_raw`.

At t=0, the surface contribution is defined as zero and Fo is zero, preserving
the exact initial value. Physics derivatives are evaluated at positive time.
For a smooth bounded raw network, the bulk first time derivative has a finite
right limit. The square-root contribution is spatially suppressed away from
the surface as time tends to zero. This does not eliminate possible surface
startup residuals or imply uniformly bounded derivatives at the corner.

Changing the bulk time input from square-root time to linear time is explicit:
the former unconfined candidate is preserved as a separate representation.
The two evaluations share weights; they are not independent bulk/layer networks.

## Necessary verification

- Both particle scales, initial profiles and center symmetry.
- A constant raw fixture -2*sign gives the correct signed surface current
  for positive time, while concentration changes vanish at startup.
- For that fixture, the bulk time derivative is constant, not proportional
  to inverse square-root time; its surface correction is negligible deep
  inside the particle at early sampled times.
- A neural raw map has the expected finite bulk derivative limit, finite
  positive-time diffusion/mixed derivatives and nonzero finite parameter
  gradients on the tested points.
- Invalid radius and negative-time inputs are rejected.
- The signed-flux fixture deliberately retains a nonzero diffusion residual:
  satisfying a surface condition is not solving the particle PDE.

There is no hard surface flux, inventory projection, positivity bound or
conservation guarantee. The linear-time bulk can still be wrong in magnitude.
No trained improvement has been demonstrated by these checks.

## Next experiment and stopping rule

Proceed next to the previously agreed short frozen-current comparison, with
checkpoint wiring and replay checks included in that task rather than adding
another stand-alone smoke stage. A paired fresh initialization of the former
four-input startup map and the confined map can use identical raw weights;
the frozen DFN current and other fields, points, optimizer, budget and loss
weights must be identical. This is a representation comparison, not a claim
of identical initial physical concentration fields at positive times.

Use 200 Adam steps per arm with no automatic extension and report diffusion,
surface flux, inventory and concentration ranges on evaluation points. Guard
against nonfinite or invalid concentrations without silent clipping. Preserve
failure results. Advancement requires joint improvement, not just lower total
loss or removal of the bulk singular term. If flux still fails, stop this
local representation sequence and review the formulation/training strategy
with the owner rather than adding more ad hoc variants.
