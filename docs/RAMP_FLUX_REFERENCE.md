# Linear-Ramp Particle Reference

This is the next reference benchmark, not a trained PINN. Run
`python scripts/check_ramp_flux_sphere.py`.

Dimensionless problem: c_t = r^-2 (r^2 c_r)_r, c(r,0)=1,
c_r(0,t)=0, c_r(1,t)=-q(t), with q(t)=k*t, k=50 and 0<=t<=0.02.
The exact mean inventory is 1-3*k*t^2/2, ending at 0.97. Unlike the previous
unit step, q(0)=0 is compatible with the uniform initial radial gradient.
This does not remove every possible higher-order startup regularity issue.

Let g(r,t) be the existing unit-step response minus one. Linearity gives
c(r,t)=1+k*integral_0^t g(r,s) ds. Integrating that series yields

    c = 1 + k*(-3*t^2/2 + (3/10-r^2/2)*t
        + sum[2*sinc(alpha*r/pi)*(1-exp(-alpha^2*t))/(alpha^4*cos(alpha))])

where alpha*cos(alpha)-sin(alpha)=0. The code evaluates 1-exp(-z) with
-expm1(-z). The target mean is derived from the imposed boundary flux, not
from fitting concentration observations. This derivation reuses the existing
audited step series; the independent numerical comparison uses PyBaMM finite
volumes with a time-dependent Neumann boundary condition.

Audits cover initial values, the derivative identity c_t=k*g, independent
volume integration, surface derivative sign and invalid domains. The script
compares 80/160/320 native FV meshes against 4096 modes, with a 2048/4096
truncation check at every comparison location and time, including the center
and surface. A low-mode truncated series can slightly overshoot near the center;
finite truncations do not inherit an exact maximum principle. FV mass uses spherical
shell volumes. Solver tolerances remain rtol=1e-11, atol=1e-13. Maxima are
sampled, not continuous bounds. Native-center comparison includes representation
error; surface values include numerical boundary reconstruction error.

No current constant-flux training settings, seed-study source files or
checkpoints are modified. Extending inventory projection and PINN training to
this protocol is a separate next step after examining this reference.

## Observed Reference Check

Final run: `results/ramp_flux_sphere_20260922T043818513769Z`.
Nine focused tests and all 163 repository tests passed. No training was run.

| Radial cells | Native concentration max error | Surface max error | Mean balance max error |
| --- | ---: | ---: | ---: |
| 80 | 3.656188e-5 | 3.763574e-4 | 7.797873e-12 |
| 160 | 9.143421e-6 | 9.636637e-5 | 9.415579e-12 |
| 320 | 2.286049e-6 | 2.437834e-5 | 1.813993e-12 |

Maximum 2048/4096-mode difference is 1.640e-10 including the center. Mesh
differences are consistent with approximately second-order reduction for the
reported maxima, not a general accuracy certification. The earlier 2048-mode
run omitted the center from its truncation check and is superseded by this
expanded audit; retain it as development evidence.

The next training change must replace BOTH the surface target by -50*t and
the inventory target by 1-75*t^2. The current constant-flux projection must not
be reused unchanged. New validation must use this ramp series, not the step
series. No arbitrary time-dependent protocol support is claimed.
