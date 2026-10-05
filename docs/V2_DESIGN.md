# DFN PINN v2: design, rationale and verification

Date: 2026-10-03. Code: `src/dfn_pinn/v2/`, scripts `scripts/v2_*.py`,
config `configs/v2_forward_1C.json`, tests `tests/test_v2_model.py`.
v1 code, results and documents are untouched.

## 1. Why a v2

A review of the v1 assembly (`src/dfn_pinn/dfn_smoke.py`) and of the
handover report identified three structural causes for the failed full-DFN
training, each sufficient on its own to prevent convergence:

1. **Output scaling.** Every v1 field is `base + 0.01 * NN(x, t)`. Potentials
   are in units of V_T = 25.7 mV, so one network unit is 0.26 mV. At 5 A the
   negative overpotential is about 89 mV (3.5 V_T): the network has to output
   ~350 to represent it, the positive overpotential ~50, the 36 % spatial
   variation of j ~36. The diagnosed "86.6 mV negative eta deficit" at 1 s
   (`docs/DFN_CHARGE_KINETIC_OFFSETS.md`) is essentially the whole
   overpotential: the potentials never left equilibrium. The flat learned j and
   the low separator electrolyte current (12.9 instead of 48.7 A/m2) follow.
2. **Benchmark conditioning.** The 1 s step from rest gives
   D t / R^2 = 1e-3 (n) and 1.5e-4 (p): the particle diffusion layer is 1-3 %
   of the radius, with a sqrt(t) singularity at the surface, and log-time
   sampling down to 1e-6 s concentrates the training on that corner.
3. **Capacity and budget.** 2 x 12 tanh units per field, 128 fixed
   collocation points per region (never resampled), 2000 Adam steps, 34 loss
   terms with weight 1.

A fourth issue appeared while building v2: with a free particle network,
`theta = theta0` satisfies the diffusion PDE exactly and violates only the
surface-flux condition. Training falls into that trivial attractor (positive
flux loss stuck at ~0.3). v2 removes it with a hard-constrained particle
representation (section 3).

## 2. Benchmark

PyBaMM DFN, Chen2020 (LG M50), SOC 1 (theta_n = 0.9014, theta_p = 0.2700),
c_e0 = 1000 mol/m3, 298.15 K, I(t) = 5 A tanh(t / 30 s), 0-3000 s
(V from 4.181 to 3.233 V). The tanh ramp removes the incompatible t = 0
corner and is applied identically in PyBaMM (Casadi DAE solver,
rtol 1e-8, atol 1e-10; IDAKLU in PyBaMM 26.9 fails at t = 0 with this
forcing). Evaluation reference: x 80/40/80, r 120; x40/r60 differs by at most
0.74 mV in voltage. The reference is used only for evaluation.

## 3. Formulation

Coordinates: t_hat = t / t_end; global X = x / L for electrolyte fields; local
x_k in [0, 1] for electrode fields; s = (r / R)^2 for particles.

| Field | Representation | Exact by construction |
| --- | --- | --- |
| c_e / c_e0 | 1 + (1 - exp(-t / 60 s)) NN(X, kinks, t, g) | IC; continuity at both interfaces |
| phi_e | -U_n(theta_n0) + 0.2 V NN(X, kinks, t, g) | continuity at both interfaces |
| phi_s,n | x_n (i L_n / sigma_n) NN | gauge phi_s,n(0) = 0 |
| phi_s,p | V(t) + (1 - x_p)(i L_p / sigma_p) NN | phi_s,p(L) = V(t) |
| V | U_p(theta_p0) - U_n(theta_n0) + 0.5 V NN(t, g) | |
| j_k | g(t) [ +/- j_ref + j_ref (NN - quadrature mean of NN) ] | integral of a j = +/- i_app(t) (hard projection); j(t=0) = 0 |
| theta_bar_k | theta0 +/- t_hat rate_k (1 + NN) | IC |
| theta_k | theta_bar - j R/(2 F D c_max) (s - 3/5) + g W [h - mean(h) - h_s(1)(s - 3/5)] | volume mean = theta_bar; -D dc/dr(R) = j/F; centre symmetry; IC |

Time features (2026-10-04 update): every network receives [2t-1, 2g-1,
sin(2 pi k t), cos(2 pi k t), k = 1..4] (`fourier_t = 4`). With only
(2t-1, 2g-1) the networks could not represent the time dependence to better
than ~8 mV in V even when fitted directly to the PyBaMM solution, and three
seeds of the PINN converged to the same biased solution (2.6-3.2 mV rmse).
The Fourier features removed that bias (docs/V2_RESULTS.md, sections 2d-2e).

g(t) = tanh(t / ramp) is also a network input (it lets every field follow
the current ramp without a sharp feature in t). Kink features
(X - X_i) H_i, with side indicators H_i, allow slope jumps at the
electrode/separator interfaces while keeping c_e and phi_e continuous.

Residuals (all O(1) per unit physical error):

* particle diffusion: (d theta/dt_hat - Q lap_s theta) / rate, with
  lap_s = 6 theta_s + 4 s theta_ss (no 1/r division; regular at the centre);
* particle inventory: (d theta_bar/dt_hat + 3 j t_end / (F R c_max)) / rate;
* electrolyte salt: (eps dc/dt_hat - Lambda d/dX(eps^b D_hat dc/dX) - source) / S_e;
* electrolyte charge: d i_hat/dX - a j L / i_ref, with
  i_hat = -K_hat (d phi_e/dX / V_T - 2 (1 - t+) d ln c / dX);
* solid charge in first-order form: d phi_s/dx_k / (L_k i_ref / sigma_k) + (g - i_hat);
  together with the hard projection this implies i_s = 0 at the separator;
* kinetics: inverse BV, (eta - 2 V_T asinh(j / 2 j0)) / V_T (direct form
  available with `kinetics = "direct"`);
* boundaries/interfaces: zero salt flux and i_e at both collectors, flux and
  current continuity at both interfaces.

Training: Adam with exponential learning-rate decay (2e-3 to 2e-5), fresh
random collocation points every step (electrodes 512 each, separator 128,
particles 128 (x, t) pairs x 8 radii each, boundaries 128 times; 30 % of
times in the first 300 s), group weights balanced every 250 steps by
gradient norms. Particle residuals share each (x, t) pair across radii and
differentiate the per-pair parts explicitly (exactly equal to the plain
autograd residual, about 8x cheaper).

## 4. Verification of the equations (independent of training)

`scripts/v2_check_equations_fd.py` evaluates the v2 conventions on the
PyBaMM fields with finite differences:

* OCPs identical to PyBaMM (1e-14 V), j0 within 3e-5 relative, inverse BV
  reproduces PyBaMM's overpotential within 0.001 mV;
* electrolyte current formula vs PyBaMM i_e: 5e-5 relative on interior faces;
* charge balance d i_e/dx = a j: 1e-4 relative with the true finite-volume
  widths;
* solid first-order form: 3.5e-6 (n) and 2.4e-7 (p) relative;
* salt balance closes in the electrode interior (example: 4.1020 vs 4.1003
  mol/m3/s); finite-difference errors dominate at the interfaces;
* particle surface-gradient signs match the sign of j.

`tests/test_v2_model.py` (7 tests): constitutive equality with v1 in the
interior, inverse/direct BV consistency, exact projection, exact particle
flux/IC, grouped particle residual equal to the plain one, interface
continuity, IC and gauge.

## 5. Results

See `docs/V2_RESULTS.md` (generated from `results/v2_runs/*`).

## 6. Known limitations / next steps

* Options implemented and kept for the planned comparisons (all tested, none
  adopted as default; see V2_RESULTS.md 2b-2c): `kinetics = direct | hard`,
  `inventory = derived`, `causal = true`, `x_edge_fraction`, fixed group
  weights, mini-batch L-BFGS (`lbfgs_iters`, `lbfgs_resample_every`), soft
  electrode-current term for `projection = false`.
* Never run L-BFGS (or Adam) on a single fixed collocation batch: the loss
  decreases while the physical error grows (V2_RESULTS.md 2b, run C2).
* The direct-vs-inverse and soft-vs-hard factorial (C-F) can now be run on a
  model that actually converges: `--set kinetics=direct projection=false`.
* Inverse problem: add parameters as trainable tensors (e.g. D_p, k_p,
  sigma_p) and a voltage-data loss; start with synthetic data from the same
  PyBaMM reference.
