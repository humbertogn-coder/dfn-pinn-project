# DFN Electrolyte Operators

The owner authorized development of the remaining DFN operators separately
from the still-unvalidated potential-driven particle PINN. The original and
two modified particle trials retain their failed acceptance status; this
operator work does not certify or supersede them. No coupled DFN training
has been performed.

## Implemented Salt Balance

Module: dfn_pinn.electrolyte_mass. Use ElectrolyteScales for global length,
time and electrolyte concentration scales. Default values are L=172.8e-6 m,
t_ref=3600 s and c_ref=1000 mol/m3, as in the mathematical contract.

The model is pointwise and returns c_e/c_ref for (N,2) [X=x/L,tau=t/t_ref].
Points require gradients. Explicit region constants are porosity, Bruggeman
exponent and constant transference number. The operator returns:

    N = -epsilon^b * D_e(c_e) * d_x c_e       [mol/m2/s]
    R = epsilon*d_t c_e + d_x N - (1-t_plus)*aj/F [mol/m3/s]
    R_hat = R/(c_ref/t_ref)

Here N is the diffusive flux used in the reduced salt equation; it is NOT a
complete species flux including migration. Source aj has units A/m3, not
particle surface j [A/m2] or cell current I [A]. Positive aj creates a positive
source; in the separator pass zero. Scalars or explicit (N,1) source tensors
are accepted, with no silent vector broadcasting. Positivity/finite-value
checks reject invalid concentration and effective diffusivity without clipping.

Autograd differentiates the full nonlinear diffusive flux. Replacing this by
D(c)*d_xx c alone would omit the D'(c)*(d_x c)^2 term. No NumPy or PyBaMM
calls appear in the differentiable path. Constant porosity/transference,
isothermal Chen2020 transport and no convection are scope restrictions.

Evaluate each material branch separately. X remains the global coordinate
even inside a thin separator. The caller must restrict collocation points to
the intended physical region; this local operator does not infer region
ownership or apply interface conditions. Do not differentiate coefficient
jumps or require equal concentration gradients across them.

## Focused Verification

Run `python scripts/check_electrolyte_mass.py`.
Eight tests passed: nonlinear polynomial product rule and SI chain factors
for three porosities, zero-source separator/uniform state, signed reaction
sources, first/second parameter gradients, and invalid inputs. No training,
new reference solves or full test suite were run. Manufactured-source tests
verify operator algebra, not accuracy of a learned solution.

## Remaining Assembly

Charge conservation and constitutive electrical currents are now implemented
locally as described below. Next implement one-sided concentration/potential/current/diffusive-flux interface
conditions and collector boundaries. Gauge and global conservation checks
must be audited before a coupled network is trained. Particle validation
remains a separate unresolved gate and must not be hidden by passing unit tests.

## Implemented Charge Operators

Module: dfn_pinn.charge_conservation. ChargeScales specifies global L, c_ref,
phi_ref and positive geometric current scale i_ref. Defaults follow the
mathematical contract: L=172.8e-6 m, c_ref=1000 mol/m3,
phi_ref=R_g*298.15/F, i_ref=5/0.1027 A/m2. Potential models return phi/phi_ref,
NOT SI volts; concentration returns c_e/c_ref. Inputs are [global x/L,t/t_ref].

    i_s = -sigma_eff*d_x(phi_s)
    R_s = d_x(i_s) + aj
    i_e = epsilon^b*kappa(c_e)*(B*d_x(log(c_e)) - d_x(phi_e))
    B = 2*R_g*T/F*(1-t_plus)*chi
    R_e = d_x(i_e) - aj

Currents have units A/m2 of geometric area; aj and physical charge residuals
have units A/m3. Normalized residuals divide by i_ref/L. Charge divergence
retains all conductivity and concentration derivative terms. The thermodynamic
factor chi is separate from (1-t_plus), which occurs only once in B.

Supply effective solid conductivity explicitly (Chen2020's electrode solid
Bruggeman exponent is zero in the current contract). Electrolyte porosity,
Bruggeman exponent, transference, thermodynamic factor and temperature are
constant within a region. The fixed-temperature transport fit is scoped to the
isothermal reference, not a general thermal model. Reject nonpositive effective
conductivity, invalid shapes/scales and nonfinite values without clipping.
Use aj=0 in the separator and evaluate each region independently.

Run `python scripts/check_charge_conservation.py`. Eleven focused tests passed:
solid conductivity/chain rule for both electrode values and signs; nonlinear
electrolyte conductivity product rule for three porosities; zero-current
diffusion potential and additive gauge invariance; opposite source signs and
source gradients; first/second parameter derivatives; invalid inputs.
No training, new reference simulations or full test suite was run.

Invariance to a spatially constant potential shift is NOT a gauge choice.
The eventual assembly must fix phi_s_n(0,t)=0 once and must not independently
fix phi_e(0,t). Electrical and diffusive flux continuity are now implemented
as described below. Passing these local tests does not establish global conservation
or successful coupled DFN training.

## Interfaces and Collectors

Module: dfn_pinn.dfn_boundaries. ElectrolyteBranch groups normalized pointwise
concentration/potential callables and region constants. Boundary coordinates
are built from (N,1) normalized times, preserving time/parameter gradients.
Mass/charge scales must share global length and concentration scales.

electrolyte_interface evaluates independent branches at the same global X,
L_n/L or (L_n+L_s)/L. Outputs are left-minus-right jumps in c/c_ref,
phi/phi_ref, i_e/i_ref and N/(c_ref*L/t_ref). Concentration gradients are not
equated. The caller supplies correct adjacent branches and interface positions;
region ownership is not inferred. Solid potential is not continuous across
the separator, which has no solid conduction branch.

collector_conditions returns an explicit conditions dictionary:

- Both collectors: zero electrolyte current and diffusive salt flux.
- Negative collector X=0: phi_s/phi_ref=0, the single potential gauge.
- Positive collector X=1: (i_s-i_app)/i_ref=0.

Applied current is signed geometric density [A/m2], not cell current [A].
The separately returned total_current_diagnostic is not an independent BC
and must not be added to the loss alongside all equations and the gauge.
solid_separator_condition enforces zero solid current at an inner electrode
boundary. Electrolyte potential is never pinned separately.

Existing local operators extract traces; their divergence outputs with dummy
zero sources are discarded. This does not impose zero electrode reaction.
The implementation currently computes second derivatives during extraction
and requires smooth pointwise models. Traces use exact boundary positions,
not the last interior mesh node.

Run `python scripts/check_dfn_boundaries.py`. Eight focused tests passed:
continuous flux with discontinuous gradients at both interface positions;
potential jumps and time/parameter gradients; collector signs at positive,
zero and negative current; single-gauge structure; deliberately wrong negative
current kept as a diagnostic; insulating solid trace; invalid time/scales.
No training, reference simulation or full test suite was run. These manufactured
traces are not a complete solution of the governing PDEs.

Next audit global current and salt balances on a manufactured three-region
assembly before connecting a trainable whole-cell model. Particle validation
remains pending and is not superseded by these interface checks.

## Three-Region Integral Audit

Run `python scripts/check_dfn_global_balance.py`. Five focused tests passed;
the previous suites and training runs were not repeated. The assembly uses
physical Gauss-Legendre dx weights over the actual electrode/separator widths.

For positive, zero and negative applied geometric current, manufactured
electrolyte potentials generate a rising i_e in the negative electrode,
constant i_e in the separator and falling i_e in the positive electrode.
The complementary solid currents satisfy i_s+i_e=i_app in the electrodes.
Local charge residuals, both interface traces, insulating solid boundaries,
collector conditions and the single negative solid gauge are checked together.
The electrode integrals of aj are +i_app and -i_app with preserved parameter
gradients. No extra negative-collector current condition is imposed.

Uniform electrolyte concentration gives zero global salt residual when the
signed electrode sources cancel. For nonzero current, its LOCAL salt residuals
do not vanish: the tests explicitly retain that fact. Thus the construction
is a charge/interface solution and a salt integral identity, NOT a complete
manufactured DFN solution. A deliberate wrong positive-electrode source is
detected by a -2*i_app charge-integral residual. Another test verifies that
the salt integral includes both internal flux jumps; ignoring them fails.

Next is implementation of a trainable three-region/two-particle assembly and
one bounded end-to-end smoke run, not another series of isolated operator
experiments. That is the second of the two near-term verification blocks
described to the owner; it is not a promise that only two tests establish
scientific validity. Passing a smoke run will establish connectivity, shapes,
finite derivatives and saved-model replay only. Particle accuracy remains an
open gate, and full DFN convergence and validation are separate future work.

The first assembly smoke has now been implemented and executed. See
DFN_ASSEMBLY_SMOKE.md for the explicit synthetic initial state, 34 connected
residuals, finite gradients in 12 branches, exact checkpoint replay and retained
order-one current-balance errors. This supersedes the planning note above,
not the unresolved physical validation gates.
