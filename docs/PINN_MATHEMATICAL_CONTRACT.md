# DFN PINN Mathematical Contract

Status: implementation specification v1, not an implemented or validated PINN.
The HDF5 reader, concentration normalization and tested constitutive primitives
exist. The network, residuals, projection and training below are planned.
This contract covers the forward Chen2020 1C case only.

## 1. Reference Configuration

Checked against installed PyBaMM 26.8.0.0 and the project's `run_discharge`:
isothermal DFN, spherical single-size particles, one phase per electrode,
Fickian diffusion, symmetric Butler-Volmer, Bruggeman transport, uniform current
collectors, prescribed current, no convection, SEI, plating or contact resistance.
Temperature is 298.15 K. The working reference is x80/r320, not an exact solution.
Current is I = +5 A during discharge. Stop at the first 2.5 V cutoff.

No inverse parameters, new protocols or experimental observations are included
in this first forward implementation. Record the full parameter snapshot and
function provenance before training; matching a parameter-set name alone is
insufficient for reproducibility.

## 2. Domains and Signs

Use a global x coordinate increasing from negative to positive current collector:

- Negative electrode: 0 < x < L_n, L_n = 85.2e-6 m.
- Separator: L_n < x < L_n + L_s, L_s = 12e-6 m.
- Positive electrode: L_n + L_s < x < L, L_p = 75.6e-6 m.
- L = 172.8e-6 m; particles have 0 <= r <= R_k for k = n,p.
- R_n = 5.86e-6 m; R_p = 5.22e-6 m.

A = electrode height * electrode width * parallel electrode count = 0.1027 m2.
Define i_app = I/A (approximately 48.6855 A/m2). Currents i_s and i_e use
geometric area. Interfacial j uses active particle area; positive j means
oxidation and lithium leaving the solid. Thus discharge has net j_n > 0 and
net j_p < 0. Never substitute j [A/m2] for a*j [A/m3].

For spherical particles, a_k = 3*epsilon_s,k/R_k. Separator a*j = 0.
Local j need not be forced to have a prescribed sign in the architecture.

## 3. Network Interface

Use separate electrolyte branches for n, separator and p: coefficients jump
at interfaces, so continuity of a global neural derivative is not required.
All branches use the same global normalized x, not undocumented local coordinates.

| Branch | Input order | Independent outputs |
|---|---|---|
| Negative particle | r/R_n, x/L, t/t_ref | c_s_n/c_s_n_max |
| Positive particle | r/R_p, x/L, t/t_ref | c_s_p/c_s_p_max |
| Negative electrode | x/L, t/t_ref | phi_s_n, phi_e_n, c_e_n, j_n in mixed variants |
| Separator | x/L, t/t_ref | phi_e_s, c_e_s |
| Positive electrode | x/L, t/t_ref | phi_s_p, phi_e_p, c_e_p, j_p in mixed variants |

Output scaling is specified below. The direct-BV baseline has no learned j.
Surface concentration is the particle network evaluated at r/R_k = 1,
not a separate network and not the last native HDF5 radial center.
Derive eta, U, j0, a*j, i_s and i_e from the outputs and constitutive laws.
Native particle data order r,x,t differs from batched neural point layout;
the loader must build coordinate tuples explicitly and test that association.

## 4. Dimensional Equations

Write residuals first in SI units. Let F be Faraday's constant and R_g the gas
constant, using the same constants as the exporting PyBaMM version.
Let theta_k = c_s_k(R_k,x,t)/c_s_k_max.

### Solid Diffusion

For each electrode and r > 0:

    d_t c_s_k = (1/r^2) d_r(r^2 D_s_k d_r c_s_k)

Chen2020 D_s_n = 3.3e-14 and D_s_p = 4e-15 m2/s at the specified temperature.
There is no x diffusion term in the particle equation.
At r=0 enforce d_r c_s=0. Do not evaluate 1/r at zero: sample interior residuals
at r>0, or implement and test the regular limit 3*D_s*d_rr c_s at the center.
At the particle surface:

    -D_s_k d_r c_s_k(R_k,x,t) = j_k(x,t)/F

### Charge Conservation

In each electrode:

    i_s_k = -sigma_eff,k d_x phi_s_k
    d_x i_s_k + a_k*j_k = 0

In all three electrolyte regions:

    i_e = kappa_eff(c_e,T) * (2*R_g*T/F * (1-t_plus) * chi * d_x(log c_e)
                            - d_x phi_e)
    d_x i_e - a*j = 0

Here chi is the thermodynamic factor, not a second copy of (1-t_plus).
For the current Chen2020 configuration t_plus = 0.2594 and chi = 1.
D_e_eff = epsilon_e^b_e D_e; kappa_eff = epsilon_e^b_e kappa_e.
sigma_eff = epsilon_s^b_s sigma. Use the actual region parameters: this case
has b_e=1.5 and electrode b_s=0, not an assumed solid exponent of 1.5.

### Electrolyte Mass

For this constant-porosity, constant-t_plus, no-convection configuration:

    epsilon_e d_t c_e = d_x(D_e_eff d_x c_e) + (1-t_plus)*a*j/F

Use the Chen2020 electrolyte transport functions, not constant approximations.
The equation above must be revised if t_plus becomes concentration-dependent
or convection/porosity evolution is introduced.

### Interfacial Kinetics

    eta_k = phi_s_k - phi_e_k - U_k(theta_k,T)
    j_k = 2*j0_k(c_e,c_s_surf,T)*sinh(F*eta_k/(2*R_g*T))

Equivalent inverse relation for j0 > 0:

    eta_k = (2*R_g*T/F)*asinh(j_k/(2*j0_k))

Implementation audit: PyBaMM 26.8.0.0 applies scaled RegPower to exchange-current
concentration factors, with delta=0.001. Use the explicit `pybamm_26_8` mode
when matching the HDF5 reference; retain `raw` only as a separately labelled
model choice. See `CONSTITUTIVE_FUNCTIONS.md` for the formula and tested range.
The regularization is nonzero even in the interior. OCP endpoint barriers are
not ported, so raw OCP compatibility is limited to the audited interior range.

Port the actual Chen2020 OCP and exchange-current functions into differentiable
tensor operations. Validate values and derivatives against the installed source
before use; do not call NumPy/PyBaMM functions on tensors inside autograd.
Enforce admissibility c_e>0 and 0<c_s_surf<c_s_max. Do not silently clip j0 or
concentrations to make residuals finite: any regularization changes the model
and must have an explicit threshold, diagnostics and an ablation.

## 5. Boundary and Initial Conditions

Solid electrical conditions matching the reference:

    phi_s_n(0,t) = 0                       # fixes the potential gauge
    i_s_n(L_n,t) = 0
    i_s_p(L_n+L_s,t) = 0
    i_s_p(L,t) = i_app(t)

Electrolyte outer boundaries:

    i_e(0,t) = i_e(L,t) = 0
    d_x c_e(0,t) = d_x c_e(L,t) = 0

At both electrode/separator interfaces impose continuity of c_e, phi_e,
i_e and -D_e_eff*d_x c_e, with one-sided branch evaluations. Do not impose
continuity of concentration gradients when coefficients differ.
The implied i_s_n(0,t)=i_app and i_s+i_e=i_app are diagnostics; avoid treating
the implied negative current condition as an additional independent boundary
condition alongside all equations and the gauge. Do not also set phi_e(0,t)=0.

Concentration initial conditions come from the exported initial_soc=1 state:
c_s_n and c_s_p are uniform in r and x initially; c_e=1000 mol/m3.
Use the verified HDF5 t=0 values for the solids, not unadjusted Chen2020 default
initial concentration entries. Approximate initial stoichiometries are
theta_n=0.910618 and theta_p=0.263845; do not round these for computation.

Potential and j initial values are algebraically consistent loaded values at
t=0+, not independently prescribed zero-current equilibrium conditions.
The instantaneous current step makes uniform initial solid concentration and
nonzero surface flux incompatible at the exact (r=R,t=0) corner. Enforce the
concentration IC at t=0 and surface flux for t>0; do not demand both derivatives
at that corner. Preserve fine startup sampling and report its errors separately.

    V(t) = phi_s_p(L,t) - phi_s_n(0,t)

Do not use the final x node as a substitute for either collector boundary.

## 6. Scales and Chain Rule

Existing reader scales:

    t_ref = 3600 s; x_ref = L; r_ref,k = R_k
    c_s_n_ref = 33133 mol/m3; c_s_p_ref = 63104 mol/m3
    c_e_ref = 1000 mol/m3

Planned additional fixed scales (not yet applied by the reader):

    phi_ref = R_g*T/F
    i_ref = 5 A / A
    j_ref,k = i_ref/(a_k*L_k)

Potentials use one common gauge and phi/phi_ref, without data-fitted offsets.
OCPs must be scaled by the same phi_ref. This does not guarantee well-conditioned
potential learning; record gradient diagnostics before adopting other offsets.
All current scales remain positive and fixed even for future rest intervals;
do not divide by the instantaneous applied current.

For normalized outputs q_hat=q/q_ref:

    d_t q = q_ref/t_ref * d_tau q_hat
    d_x q = q_ref/L * d_xhat q_hat
    d_r q = q_ref/R_k * d_rhat q_hat

Second derivatives require squared coordinate scales. With normalized x, every
physical electrode integral includes dx=L*d_xhat. With normalized radius,
particle integrals include r^2 dr=R_k^3*rhat^2*d_rhat.

Make residuals dimensionless with these fixed divisors:

| Residual | Divisor |
|---|---|
| Solid diffusion equation | c_s_k_ref/t_ref |
| Electrolyte mass equation | c_e_ref/t_ref |
| Charge divergence equation | i_ref/L |
| Particle surface flux | j_ref,k/F |
| Direct BV, learned j | j_ref,k |
| Inverse BV | phi_ref |
| Concentration IC or interface continuity | corresponding concentration scale |
| Potential boundary or interface continuity | phi_ref |
| Electrical current boundary or interface continuity | i_ref |
| Electrolyte diffusive flux continuity | c_e_ref*L/t_ref |
| Zero particle concentration gradient at center | c_s_k_ref/R_k |
| Zero electrolyte concentration gradient at collector | c_e_ref/L |

These are unit-consistent starting scales, not tuned loss weights. Keep loss
weights separate, log each term and require chain-rule tests before training.

## 7. Ablations and Integral Projection

Use identical data, scales, sampling budgets and seeds across four variants:

1. Direct BV: derive j from concentrations and potentials; no learned-j head.
2. Mixed direct BV: learn j and enforce its direct-BV residual.
3. Mixed inverse BV: learn j and enforce the inverse-BV residual.
4. Projected inverse BV: project raw learned j, then enforce inverse BV.

For variant 4, at each time and separately in each electrode define
J_n=i_app and J_p=-i_app, and use:

    j_projected(x,t) = j_raw(x,t)
                     + (J_k(t) - integral_electrode a_k*j_raw dx)
                       / integral_electrode a_k dx

Use projected j consistently in kinetics, particle flux, charge and electrolyte
mass equations. This guarantees the integrated constraint only under the chosen
quadrature; it does not guarantee local charge conservation or exact continuous
integrals. Use fixed differentiable quadrature nodes, preserve gradients through
the integral, include the physical dx Jacobian, and check on independent nodes.
For constant a this is a uniform additive correction, not a positivity constraint.
Projection is undefined for zero active area; reject that configuration.

## 8. Conservation and Data Use

Diagnostics independent of the training loss:

    integral_n a_n*j_n dx = i_app
    integral_p a_p*j_p dx = -i_app
    N_s,k = A * integral_k epsilon_s,k * (3/R_k^3)
                       * integral_0^R_k c_s_k*r^2 dr dx
    N_e = A * integral_0^L epsilon_e*c_e dx
    d_t N_s,n = -I/F; d_t N_s,p = +I/F
    d_t(N_s,n + N_s,p + N_e) = 0

Particle volume weighting is essential; an unweighted average over r is wrong.
For this constant-t_plus closed-cell case N_e is also conserved globally.

Use the reader's 956 nonoverlapping timestamps for reference inspection, not as
a claim of 956 independent experimental observations. Physics collocation points
are separate from reference data points. Define losses per region and per time
window; do not let 401 fine-startup samples dominate the full discharge metric.
Use physical integration weights for integrated errors and report startup maxima
separately. No random split of overlapping segment samples is an independent test.
Label supervised pretraining as supervised, not as a physics-only baseline.

## 9. Implementation Gates

Before any full-DFN training:

1. Snapshot physical parameters, constitutive function provenance and the initial
   concentrations actually used by the reference.
2. Test tensor constitutive functions and their derivatives, normalization
   inverses, axis mapping and chain-rule factors in float64.
3. Test direct/inverse BV equivalence on admissible inputs of both signs.
4. Test projection, quadrature Jacobians and gradients, including zero current.
5. Test spherical diffusion, flux signs and volume conservation on the existing
   analytic benchmarks before coupling the full DFN.
6. Check gauge, interface traces and concentration admissibility; then run one
   small forward training experiment and inspect all residual components.

No PINN accuracy, convergence, speedup or inverse identifiability claim is made
by this document. The next coding ticket is the differentiable constitutive
layer with tests, not a large full-DFN training run.

## Sources and Audit Trail

- Project `scripts/export_pybamm_reference.py` defines the reference protocol.
- Installed PyBaMM 26.8.0.0 `basic_dfn.py` was inspected for signs, fluxes,
  boundary conditions and constitutive conventions; actual DFN options and
  Chen2020 values were also queried directly on 2026-09-15.
- [PyBaMM DFN documentation](https://docs.pybamm.org/en/pybamm-v26.7.1.0/source/examples/notebooks/models/DFN.html)
  is a public conceptual reference, not a substitute for the installed version.
- `WORKING_REFERENCE.md` and `PINN_DATA_READER.md` document numerical and data
  limitations. This contract defines design choices beyond those implemented files.
