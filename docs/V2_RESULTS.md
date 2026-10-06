# DFN PINN v2: results

Benchmark and formulation: `docs/V2_DESIGN.md`. All numbers are against the
PyBaMM x80/r120 reference on its native grid (436 output times, 0-3000 s);
the reference never enters the forward loss. Hardware: Claude cloud sandbox,
2 CPU cores, float32, ~0.27 s per Adam step.

## 0. Summary (2026-10-04)

Final forward configuration (`configs/v2_forward_1C.json`, `fourier_t = 4`,
20 000 Adam steps, ~2 h on one CPU core): over three seeds the voltage error
against PyBaMM is 0.51 +/- 0.04 mV rmse and 2.5 +/- 0.4 mV max over the
3000 s discharge; electrolyte concentration 8.5 mol/m3 rmse (max 77, at the
negative-collector peak); reaction current 0.7 % (n) and 2.4 % (p) of j_ref
rmse; surface stoichiometry 6e-4 / 8e-4 rmse. The route to this result,
including the negative results, is in sections 2b-2f.

Inverse problem (section 3): from voltage data with 1 mV noise the PINN
recovers D_p and k_n to about 1 % and D_n to about 10 % (single rate,
3.2). The parameter error is set by the systematic voltage error of the
forward PINN divided by the voltage sensitivity of the parameter, not by
the information in the data (Cramer-Rao bounds are 5-10x smaller): a
0.5 mV forward error is worth 10 % in D_n but 0.7 % in D_p at 1C, and a
test started at the true parameters with noise-free data confirms a
biased minimum (D_n -4 %, 3.3). Adding C/2 and 2C data (3.3) did not help
within a 10 000-step budget because the forward error at 2C is larger;
multi-rate estimation needs the forward problem converged at every rate
first (hierarchical release, now supported by `param_release_steps` and
`--init`). Hunting the bias (sections 3.3, 6, 7) found a real defect of
the formulation, the soft zero-flux condition at the collectors
(violated at the graphite-staging current peaks); `collector_bc = "hard"`
(cos(pi X) features) fixes it and halves the electrolyte error (c_e rmse
7.0 -> 3.4 mol/m3, V 0.54 -> 0.45 mV); width 96 x 40k steps gives
0.29 mV. The D_n-equivalent bias of about -4 % survives both: it is a
late-time (t > 2300 s) voltage offset of +0.3 mV plus an end-point jump
that every variant shares, located where the graphite OCP steepens.

Final state of the DFN part (2026-10-06, sections 9 and 8): hard collectors
x width 96 x 40 000 steps give V 0.25 mV rms / 1.15 mV max and c_e max 6.6
mol/m3 (section 9), the predicted D_n bias falls to -0.9 %, and the
hierarchical inverse on that model recovers D_p, k_n, D_n to +0.1, +0.4,
-0.6 % from 2.5x / 0.4x / 0.6x with 1 mV noise. For aged cells (section 8)
the pipeline (synthetic degradation data, aging parameters theta_n0,
theta_p0, eps_am_n, eps_am_p, R0, two-stage inverse) recovers the
negative-electrode lithium content to ~1 % but cannot split LLI from LAM,
because the 5-parameter aging DFN itself misses the aged discharge by
30-45 mV (PyBaMM check). The cause (end of section 8) is PyBaMM's
stress-induced diffusion, switched on by default with the particle-mechanics
submodel of the degradation model: it multiplies the NMC diffusivity by
100-400, so the data have no positive-particle diffusion polarization while
the PINN's DFN has Chen2020's D_p. Not an aging effect: a 2-cycle cell shows
the same 75 mV. Fix: generate the synthetic cells without it, and add
D_k(c) = D_k0 (1 + theta_M,k c) to the PINN (or release D_p) for the real
LG M50 data set, which has the same feature.

## 1. Forward problem, first run (plain time inputs)

Run `results/v2_runs/run20k_salt_20261004T015631Z` (config
`configs/v2_forward_1C.json`, seed 0, 20 000 Adam steps, 90 min).

| Adam steps | V RMSE [mV] | V max [mV] | c_e RMSE / max [mol/m3] | j_n RMSE / j_ref | j_p RMSE / j_ref | theta_surf,n RMSE | theta_surf,p RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 500 | 5.15 | 17.0 | 44.5 / 287 | 0.119 | 0.050 | 0.0095 | 0.0023 |
| 5 000 | 5.68 | 15.9 | 40.5 / 273 | 0.060 | 0.043 | 0.0072 | 0.0015 |
| 10 000 | 3.21 | 10.7 | 30.1 / 227 | 0.067 | 0.038 | 0.0082 | 0.0014 |
| 15 000 | 2.80 | 9.9 | 28.8 / 228 | 0.070 | 0.034 | 0.0083 | 0.0013 |
| 20 000 | **2.56** | **9.07** | 28.4 / 227 | 0.072 | 0.033 | 0.0085 | 0.0012 |

(Intermediate rows use 120 of the 436 reference times; the last row uses all.)
Electrolyte potential: RMSE 1.7 mV, max 8.2 mV. Reference mesh uncertainty
(x40/r60 vs x80/r120): V 0.74 mV, c_e 1 mol/m3, theta_surf 0.002, j 0.024 A/m2.

Figure: `results/v2_runs/run20k_salt_20261004T015631Z/diagnostic.png`.

Screen against the v1 limits (`configs/dfn_baseline_v1.json`): none of the
max-norm limits pass yet (V max 9.1 vs 5 mV; c_e max 227 vs 10 mol/m3; j max
23-26 % vs 1 % of j_ref; theta_surf max 0.031/0.009 vs 0.001). The largest
errors sit in the last ~5 % of the electrode next to the separator (steep j
and theta_surf gradients) and, for c_e, near the negative collector late in
the discharge. These v1 limits were written for a 1 s problem; for a full
discharge they are very strict (c_e spans 500-2250 mol/m3, so 10 mol/m3 is
0.6 % of the range). Proposed working targets for the next stage: V max
5 mV, c_e RMSE 1 % of range, j RMSE 3 % of j_ref, theta_surf RMSE 0.005.

For context, v1's best result was 23.2 mV voltage error on the 1 s problem
and 132.8 mV for the first full-budget C0 run.

## 2. Ablation evidence collected while building v2

| Change | Evidence |
| --- | --- |
| Free particle network (no hard flux/inventory) | `smoke2k_20261004T004508Z`: positive flux residual stuck at MSE 0.31, particle 0.17; positive surface stoichiometry at 3000 s 0.37 vs 0.81 (trivial attractor theta = theta0). Fixed by the hard particle representation. |
| No global salt term | `run20k_20261004T011916Z` (stopped at 6000 steps): mean c_e drifted -53, -148, -246 mol/m3 at t = 300, 1200, 3000 s although local salt residuals were 1e-3-1e-2; c_e max error 800 mol/m3 at 5000 steps vs 273 with the term. |
| Hard particle + no salt term, 3000 steps | `smoke_hardp_20261004T010400Z`: V RMSE 11.5 mV. |

## 2b. Refinement experiments on the 20k checkpoint (2026-10-04)

All warm starts from `run20k_salt_20261004T015631Z/final.pt`, one CPU thread
each, Adam lr 1e-4 -> 5e-6 (a fresh Adam state at the config's lr = 2e-3
destroys the warm start within two steps: V error 56 mV). Evaluations on 120
reference times.

| Experiment | steps | V rmse / max [mV] | c_e max | phi_e max [mV] | j_n / j_p rmse/jref | th_s,n / th_s,p max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline end point | 20k | 2.56 / 9.07 | 227 | 8.2 | 0.072 / 0.033 | 0.031 / 0.0088 |
| A: + separator-side sampling (30 % of x within 0.15 of the separator), adaptive weights | +7.5k | 3.36 / 7.76 | 214 | 7.2 | 0.082 / 0.032 | 0.032 / 0.0091 |
| D: fixed weights, kinetics x10 (= nearly uniform physical weights) | +7.5k | 3.14 / 11.24 | 259 | 10.2 | 0.038 / 0.021 | 0.023 / 0.0059 |
| C: L-BFGS float64, fixed 4x batch | 60 it | 2.53 / 8.25 | 235 | 7.4 | 0.071 / 0.033 | 0.030 / 0.0089 |
| C2: L-BFGS float64, SAME fixed batch | 2000 it | 10.10 / 26.2 | 1183 | 23.2 | 0.066 / 0.027 | 0.032 / 0.0087 |

C2 is a textbook collocation overfit: the loss on the fixed batch fell from
2.95 to 1.68 (-43 %) while the c_e error against PyBaMM grew five-fold and
the voltage error four-fold. Any second-order refinement must resample
(`lbfgs_resample_every`, section 2d); this also retroactively explains part
of v1's difficulties, which trained on 128 fixed collocation points.

Reading: (i) the three continuations move along a flat valley: emphasizing
kinetics (D) halves the j and theta_surf errors but degrades c_e, phi_e and
V; (ii) L-BFGS reduces the loss by only ~5 % in 200 evaluations; (iii) the
remaining kinetic residual is smooth and low-frequency (+/-0.3 V_T across
the electrode, slow oscillation in time), not an under-resolution of fine
scales. Diagnosis: with a free j network the direction "perturb j with zero
electrode mean, let theta_bar / theta_s follow through the inventory" is a
soft mode that only the kinetics residual penalizes; residuals normalized by
rates (mass, inventory) accumulate over the 3000 s integration (a 0.5 %
rate error becomes 0.4 % of the total stoichiometry change), which is why
small PDE residuals coexist with 10 % c_e errors. These experiments
motivated the "derived current" formulation of section 2c.

## 2c. Formulation variants (fresh runs, seed 0, 2 threads)

Baseline = `run20k_salt` (learned j, inverse BV, soft inventory). Evaluations on
120 reference times at the same Adam step.

| Variant | step | V rmse / max [mV] | c_e max | phi_e max [mV] | j_n / j_p rmse/jref | th_s,n / th_s,p max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 2500 | 5.15 / 17.0 | 287 | 15.0 | 0.119 / 0.050 | 0.051 / 0.0150 |
| derived current (`inventory=derived`) | 2500 | 5.24 / 10.9 | 241 | 12.9 | 0.142 / 0.052 | 0.157 / 0.0137 |
| baseline | 5000 | 5.68 / 15.9 | 273 | 12.4 | 0.060 / 0.043 | 0.031 / 0.0105 |
| derived current | 5000 | 4.13 / 8.2 | 247 | 7.4 | 0.110 / 0.050 | 0.044 / 0.0136 |
| baseline | 7500 | 3.89 / 11.6 | 237 | 9.3 | 0.063 / 0.042 | 0.031 / 0.0110 |
| derived current | 7500 | 4.21 / 9.5 | 214 | 6.9 | 0.102 / 0.049 | 0.030 / 0.0129 |

| baseline | 2500 | 5.15 / 17.0 | 287 | 15.0 | 0.119 / 0.050 | 0.051 / 0.0150 |
| hard kinetics (`kinetics=hard`) | 2500 | 8.02 / 18.3 | 274 | 21.0 | 0.105 / 0.047 | 0.073 / 0.0097 |

| baseline | 7500 | 3.89 / 11.6 | 237 | 9.3 | 0.063 / 0.042 | 0.031 / 0.0110 |
| causal time weighting (`causal=true`, Wang et al. 2022; 16 bins, eps 1e-2) | 7500 | 3.92 / 11.1 | 344 | 9.7 | 0.115 / 0.050 | 0.048 / 0.0131 |

`causal=true` weights each time bin by exp(-eps * cumulative loss of earlier
bins) (time marching). After 7500 steps the late-time weights are still ~0.7
and the late-time fields are less converged than in the baseline; no gain at
equal cost in this problem (the baseline already samples 30 % of the points
in the first 300 s and imposes the initial conditions exactly).

`kinetics=hard` defines phi_e inside the electrodes by inverse Butler-Volmer,
phi_e = phi_s - U(theta_s) - 2 V_T asinh(j / 2 j0), so the kinetic residual
disappears and the j-shape error is penalized directly by the electrolyte
charge equation; phi_e continuity at the two interfaces becomes a residual.
Cost 2.2x per step; at equal steps it is not better than the baseline
(stopped at 3000 steps).

`inventory=derived` makes theta_bar the learned field and j = -(F R cmax/3)
d theta_bar/dt, so inventory, electrode current and j(x,0)=0 are exact and
the net_thbar networks disappear (+25 % cost per step). Voltage and
electrolyte errors improve, j errors worsen: the exact inventory transmits
the j-shape error directly into theta (max theta_s error 0.157 at 2500
steps, at the negative collector at 3000 s). Comparable overall; kept as an
option, not adopted as default.

## 2d. Seed study (final baseline configuration, 20 000 Adam steps each)

| seed | run | V rmse [mV] | V max [mV] | c_e rmse / max | phi_e max [mV] | j_n / j_p rmse/jref | th_s,n / th_s,p rmse |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | run20k_salt_20261004T015631Z | 2.56 | 9.07 | 28.4 / 227 | 8.2 | 0.072 / 0.033 | 0.0085 / 0.0012 |
| 1 | seed1_20261004T072014Z | 2.68 | 8.61 | 25.8 / 186 | 6.6 | 0.065 / 0.035 | 0.0067 / 0.0013 |
| 2 | seed2_20261004T083213Z | 3.19 | 8.86 | 24.8 / 178 | 7.0 | 0.073 / 0.037 | 0.0079 / 0.0014 |
| mean +/- std | | 2.81 +/- 0.33 | 8.85 +/- 0.23 | 26.3 / 197 | 7.2 | 0.070 / 0.035 | 0.0077 / 0.0013 |

The three seeds agree closely, and more importantly their ERROR FIELDS are
the same: correlation of the pointwise error between seeds is 0.94-0.99 for
V(t), c_e(x,t) and j_n(x,t). Decomposing each error into the seed-mean
pattern plus a seed-specific remainder: V 2.75 mV (common) vs 0.4-1.0 mV
(seed-specific); c_e 26 mol/m3 (common) vs 3-6 (seed-specific). The common
j_n error at 1500 s runs from +0.10 A/m2 at the collector to -0.22 at the
separator and reverses sign at 3000 s. The remaining error is therefore a
SYSTEMATIC approximation bias of this architecture/loss, not optimizer noise;
more seeds will not remove it, and neither will more steps at the observed
power-law rate. Section 2e tests the architecture directly.

Mini-batch L-BFGS refinement (float64, 8x batches resampled every 25
iterations, 800 iterations, ~1 h on one core) on seed 0: V 2.56 -> 2.15 mV
rmse, 9.07 -> 6.97 mV max; c_e max 227 -> 217; phi_e max 8.2 -> 6.4 mV; j and
theta unchanged. Salt conservation is preserved (drift < 0.3 mol/m3). An
earlier version of this refinement that evaluated the salt term only at
early times (indexing bug, fixed) reached 4.2 mV max by LOSING 9 % of the
salt (drift -88 mol/m3): a reminder that voltage accuracy alone is not a
convergence criterion.

## 2e. Architecture test by direct regression, and the Fourier-time result

To separate "the networks cannot represent the solution" from "the PINN
loss cannot find it", `scripts/v2_check_residuals.py --no-theta` fits the
networks to the PyBaMM fields by plain regression (no physics; 3000 Adam
steps, 1024 points per field per step, lr 2e-3 -> 1e-5):

| architecture | V rmse [mV] | phi_e rmse [mV] | phi_s,p rmse [mV] | c_e rmse | j_n rmse/jref |
| --- | ---: | ---: | ---: | ---: | ---: |
| width 64, tanh, inputs (x, 2t-1, 2g-1) = v2 baseline | 8.19 | 4.46 | 6.12 | 21.4 | 0.0051 |
| same + Fourier time features sin/cos(2 pi k t), k = 1..4 (`fourier_t=4`) | 3.36 | 1.21 | 2.47 | 5.7 | 0.0026 |

Even with the exact solution as data, the baseline networks cannot
represent V(t) better than 8 mV rmse in this budget: the time dependence is
the representational bottleneck (spectral bias; the only time inputs were
2t-1 and the saturated 2g-1). Adding four Fourier frequencies in time
improves every field 2-4x in the regression and, in the PINN, changes the
convergence qualitatively (same loss, same sampling, seed 0, one thread):

| Adam steps | baseline V rmse / max [mV] | `fourier_t=4` V rmse / max [mV] | baseline j_n rmse/jref | f4 j_n rmse/jref | baseline th_s,n max | f4 th_s,n max | baseline c_e max | f4 c_e max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 500 | 5.15 / 17.0 | 3.87 / 16.4 | 0.119 | 0.055 | 0.051 | 0.028 | 287 | 366 |
| 5 000 | 5.68 / 15.9 | 1.90 / 8.3 | 0.060 | 0.018 | 0.031 | 0.011 | 273 | 207 |
| 7 500 | 3.89 / 11.6 | 1.16 / 4.98 | 0.063 | 0.012 | 0.031 | 0.0074 | 237 | 138 |
| 20 000 | 2.56 / 9.1 | **0.57 / 2.46** | 0.072 | **0.007** | 0.031 | **0.0047** | 227 | **62** |

Final f4 seed 0 (`f4_seed0_20261004T120413Z`, all 436 reference times):
V rmse 0.54 mV, max 2.46 mV; phi_e rmse 0.34 / max 2.6 mV; phi_s,p max 2.7 mV;
c_e rmse 7.0 / max 62 mol/m3; j_n rmse 0.7 % (max 7.6 %) and j_p rmse 2.7 % (max 23 %,
localized at the separator side of the positive electrode during the first
~100 s); theta_surf rmse 0.0005 / 0.0010, max 0.0047 (n) / 0.0075 (p). The three 5 mV voltage/
potential screens of the v1 acceptance policy PASS; the field screens
(c_e 10 mol/m3, j 1 % of j_ref, theta 0.001) do not yet. Figure:
`results/v2_runs/f4_seed0_20261004T120413Z/diagnostic.png`.

Where the remaining f4 error sits (seed 0, all 436 times): j_p max 0.39 A/m2
(23 % of j_ref) at the separator side of the positive electrode during the
30-150 s transient (after 150 s: max 9 %, rms 1.2 %); j_n max 0.11 A/m2
(7.6 %) at the separator side around 800 s (rms 1.0 %); c_e max 62 mol/m3 at
the negative collector near 2500 s (rms 7.0); V max 2.5 mV at the very end
of the window (rms 0.40 mV for t > 150 s); theta_surf rms 5e-4 (n) and
1e-3 (p). The early positive-electrode transient (Q_p = 0.44, OCP feature at
theta = 0.31 crossed at ~100-200 s) is the next thing to resolve, e.g. with
more early-time samples or an extra short-time feature.

Attempt at the early transient (run `f4s_seed0_20261004T175436Z`): two
short-time features 2 exp(-t/tau) - 1 with tau = 50 and 200 s plus 15 % of
the time samples inside the first 60 s. In the regression test this
architecture fits the fields better than f4 alone (V 2.90 vs 3.36 mV,
phi_s,p 1.99 vs 2.47 mV, j_p 0.07 vs 0.11 %), and width 96 does too (V 2.42,
c_e 5.0, 2x cost per step); eight Fourier frequencies instead of four are
worse (V 4.45 mV). As a PINN, however, the short-time variant converges more
slowly and ends worse than f4 at 20 000 steps (V 0.81 / 5.09 mV, c_e max
141) except for the transient quantities it targets (j_p rmse 2.1 vs 2.7 %,
theta_surf,p max 0.0060 vs 0.0075). Not adopted; the recommended next try
for the transient is width 96 with 40 000 steps on a machine with more cores.

## 2f. Seed study with the final architecture (`fourier_t = 4`, 20 000 steps)

| seed | run | V rmse [mV] | V max [mV] | c_e rmse / max | phi_e max [mV] | j_n / j_p rmse/jref | th_s,n / th_s,p max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | f4_seed0_20261004T120413Z | 0.54 | 2.46 | 7.0 / 62 | 2.6 | 0.007 / 0.027 | 0.0047 / 0.0075 |
| 1 | f4_seed1_20261004T125139Z | 0.46 | 2.88 | 9.0 / 79 | 2.3 | 0.008 / 0.023 | 0.0048 / 0.0067 |
| 2 | f4_seed2_20261004T141440Z | 0.52 | 2.10 | 9.7 / 90 | 1.5 | 0.007 / 0.020 | 0.0046 / 0.0057 |
| mean +/- std | | 0.51 +/- 0.04 | 2.48 +/- 0.39 | 8.5 / 77 | 2.1 | 0.0072 / 0.0235 | 0.0047 / 0.0066 |

Compared with the plain architecture (section 2d): voltage rmse 5.5x lower,
voltage max 3.6x lower, c_e 3x, j_n 10x, theta_surf,n 6x. All three seeds
pass the three 5 mV screens (V, phi_e, phi_s,p). Seed-to-seed error-field
correlations fell from 0.94-0.99 to 0.72-0.94: a smaller systematic
component remains (mainly the 30-150 s positive-electrode transient and the
c_e peak at the negative collector), on top of which seed noise is now
visible. Figures: `diagnostic.png` in each run folder.

Mini-batch L-BFGS (600 iterations, float64, 4x batches) on f4 seed 0 changes
nothing measurable (V 0.57 / 2.35 mV, c_e max 61): at this point the error
is no longer limited by the optimizer but by the remaining representation
error of the early positive-electrode transient and the c_e collector peak.

## 3. Inverse problem

### 3.0 Local identifiability from voltage data (PyBaMM sensitivities)

`scripts/v2_identifiability.py` computes dV/dlog p by central finite
differences of the PyBaMM DFN (same cell and tanh ramp), the Fisher
information for Gaussian voltage noise of 1 mV sampled every 10 s, the
Cramer-Rao bound on the relative standard deviation of each parameter, the
parameter correlations and the singular values of the sensitivity matrix.
Local, linearized statements around the true parameters: they identify weak
and correlated directions, not global uniqueness.

| parameter | CRLB rel. std, C/2 alone | 1C alone | 2C alone | C/2 + 1C + 2C | max dV/dlog p at 1C [mV] |
| --- | ---: | ---: | ---: | ---: | ---: |
| D_n (neg. solid diffusivity) | 3.8 % | 4.5 % | 3.8 % | 1.6 % | 15 |
| D_p (pos. solid diffusivity) | 0.40 % | 0.55 % | 0.75 % | 0.18 % | 112 |
| k_n (neg. exchange-current prefactor) | 3.7 % | 3.1 % | 2.8 % | 0.32 % | 49 |
| k_p (pos. exchange-current prefactor) | 26 % | 19 % | 15 % | 5.1 % | 20 |
| sigma_p (pos. solid conductivity) | 41 % | 37 % | 26 % | 11 % | 8 |
| D_e (electrolyte diffusivity scale) | 5.2 % | 2.9 % | 1.1 % | 0.25 % | 55 |
| kappa_e (electrolyte conductivity scale) | 5.7 % | 4.0 % | 2.3 % | 0.92 % | 30 |

Condition number of the noise-normalized sensitivity matrix: 607 (C/2), 682
(1C), 925 (2C), 441 (combined). The weakest direction is always the pair
(k_p, sigma_p) (correlation -0.81 at 1C, -0.97 combined): the kinetic and the
ohmic resistance of the positive electrode are nearly interchangeable in the
voltage. D_p is by far the best-identified parameter; D_n, k_n, D_e and
kappa_e are identifiable at the few-percent level from a single 1C curve
and below 1-2 % from three rates. Raw sensitivities in
`results/v2_identifiability/`. Consequence for the PINN inverse study: use
{D_p, k_n, D_n} (or add D_e, kappa_e) for single-rate data and keep k_p and
sigma_p fixed unless several C-rates (or EIS / pulse data) are available.
The PINN now carries multipliers for all seven (`DFNPINN.PARAMETERS`).

### 3.1 Synthetic inverse problem, plain architecture (first attempt)

Run `results/v2_runs/inverse_scratch_20261004T033045Z` (config
`configs/v2_inverse_synthetic.json`, 15 000 Adam steps, 67 min). Fields and
parameters are trained jointly from scratch (no warm start). Data: the
PyBaMM voltage at the 436 reference times plus 1 mV Gaussian noise. Unknowns:
log-multipliers of D_p (positive-particle diffusivity) and k_n
(negative exchange-current prefactor), started at 2.5x and 0.4x the true
values. Physics loss unchanged; the data term is (V_pred - V_data)/1 mV.

| Adam steps | D_p / true | k_n / true | voltage misfit RMS [mV] |
| ---: | ---: | ---: | ---: |
| 0 | 2.50 | 0.40 | 532 |
| 1 000 | 2.23 | 0.53 | 5.5 |
| 2 500 | 1.54 | 0.68 | 3.4 |
| 5 000 | 1.16 | 0.82 | 3.0 |
| 10 000 | 1.044 | 0.906 | 2.3 |
| 15 000 | **1.038** | **0.918** | 2.1 |

Final errors: D_p +3.8 %, k_n -8.2 %, still drifting slowly toward the true
values when the learning rate had decayed. Forward field accuracy of the
same run (against PyBaMM): V RMSE 1.83 mV, max 7.6 mV. Figures:
`inverse_parameters.png`, `diagnostic.png` in the run folder.

Caveats: single seed; the two parameters are only weakly correlated for this
protocol (D_p acts late in the discharge, k_n mainly on the early
overpotential), which is why they separate well. k_n is the harder one:
its voltage signature (a few mV of kinetic overpotential) is comparable to
the residual model error, so a 8 % bias is consistent with the present
forward accuracy. Identifiability analysis (sensitivities, Fisher
information) and multi-start are the next steps before experimental data.

### 3.2 Synthetic inverse problem with the final architecture (`fourier_t = 4`)

Parameters chosen from 3.0: D_p, k_n, D_n (CRLB at 1C with 436 samples and
1 mV noise: about 0.5 %, 2.6 %, 3.7 %). Data: the 436 reference voltages
plus 1 mV Gaussian noise. True multipliers 1; starts 2.5, 0.4, 0.6.

| run | start | steps | D_p | k_n | D_n | V misfit to data [mV] | V error vs noiseless reference [mV] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f4_inverse3_20261004T170030Z | from scratch | 15 000 | +1.7 % | -1.1 % | -13.5 % | 0.98 (noise 1.0) | 0.37 rmse / 1.74 max |
| f4_inverse3_warm_20261004T184453Z | forward f4 model, fresh Adam, lr 2e-4 | 5 000 | +1.5 % | -1.0 % | -10.1 % | 1.00 | 0.43 / 2.16 |
| f4_inverse3_cont_20261004T192342Z (continuation of the first, Adam state restored, param lr 0.02) | | +10 000 | +0.8 % | -0.2 % | -8.8 % | 0.98 | 0.35 / 1.68 |

Trajectories (`inverse_parameters.png` in each folder): D_p and k_n converge
within 1-2 % in a few thousand steps, well inside the CRLB x3; D_n is the
slow direction. In the from-scratch run it first reaches 1.0 at step 500,
is pushed down to 0.76 while D_p and k_n are still far from their values
(compensation), and then recovers at +0.015 per 1000 steps once the others
have converged; the warm start shows the same slow recovery. The voltage
misfit is at the noise floor from step 5000 on, so the data no longer
discriminate; the recovery of D_n is driven by the physics residuals. With
the plain architecture (3.1) the same two parameters came out at +3.8 % and
-8.2 %: the forward accuracy gained in step 1 translates directly into
parameter accuracy.

### 3.3 Multi-C-rate synthetic inverse problem (`scripts/v2_inverse_multirate.py`)

Setup: three protocols with the same 30 s tanh ramp, C/2 (2.5 A, 0-6500 s,
V 4.18 -> 3.19), 1C (5 A, 0-3000 s) and 2C (10 A, 0-1400 s, V -> 3.05; the
electrolyte is nearly depleted at the positive collector, c_e down to
82 mol/m3). One DFNPINN per protocol (own networks, scales and adaptive
weights, `fourier_t = 4`), physical log-multipliers shared, voltage data of
the three protocols at their reference times with 1 mV Gaussian noise;
about 1 s per step for the three models on two cores.

**First attempt, five parameters (D_p, k_n, D_n, D_e, kappa_e), aborted.**
With the parameters released from step 1 and `param_lr = 0.02`, D_e ran to
24x and kappa_e to 0.37x within 500 steps while the fields were still
random (`f4_multirate5_aborted`, first launch). With a 2000-step parameter
warm-up (fields trained first with the wrong parameters frozen) and
`param_lr = 0.01`, the voltage data were already fitted to 1.3 mV at the
end of the warm-up although the parameters were wrong, and after release
the electrolyte pair drifted away (D_e 1.5 -> 2.5, kappa_e 0.7 -> 0.32 by
step 3500) while k_n overshot to 1.15; the 2C electrolyte concentration
field was still inaccurate at that point (max error 1300 mol/m3 at step
2500). Two lessons: the voltage data can be fitted by the fields almost
independently of the parameters, so the parameter information lies in the
tension between the physics residuals and the data, and the electrolyte
parameters compensate forward-model errors in c_e. Electrolyte parameters
should only be released once the electrolyte fields are converged
(hierarchical estimation), which is consistent with the identifiability
analysis (D_e and kappa_e are well identified only because of the 2C
curve, i.e. precisely where the forward problem is hardest).

**Second run, three parameters (D_p, k_n, D_n), electrolyte fixed**
(`f4_multirate3_20261004T224451Z`; from 2.5x / 0.4x / 0.6x, `param_lr`
0.01, no warm-up, bound factor 0.1-10, 10 000 Adam steps, 2.8 h on two
cores; interrupted twice by container restarts and continued with
`--resume`). Caveat found afterwards: the first version of `--resume`
re-stepped a fresh scheduler on top of the stored, already-decayed
learning rate, so from step 1000 on this run trained with lr and
`param_lr` 0.4x the nominal schedule (final lr 1.3e-5 instead of 2e-5);
fixed on 2026-10-05 (closed-form lr on resume). This slows the field
convergence somewhat but does not change the conclusions below.
Multipliers (true value 1) and voltage misfit per protocol:

| step | D_p | k_n | D_n | misfit C/2 / 1C / 2C [mV] |
| ---: | ---: | ---: | ---: | --- |
| 0 | 2.50 | 0.40 | 0.60 | - |
| 1000 | 1.53 | 0.54 | 0.80 | 1.74 / 1.86 / 2.43 |
| 2500 | 1.10 | 0.81 | 1.21 | 1.30 / 1.28 / 1.32 |
| 5000 | 0.966 | 0.979 | 0.848 | 1.09 / 0.99 / 1.00 |
| 7500 | 0.961 | 1.007 | 0.741 | 1.04 / 0.93 / 0.97 |
| 10000 | **0.963** | **1.011** | **0.718** | 1.03 / 0.91 / 0.96 |

Field errors against the PyBaMM references (60 times each):

| step | protocol | V rmse / max [mV] | c_e rmse / max | j_n / j_p rmse/jref | th_s,n / th_s,p max |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2500 | C/2 | 1.18 / 7.0 | 13.6 / 63 | 0.083 / 0.033 | 0.032 / 0.016 |
| 2500 | 1C | 1.30 / 6.1 | 46.6 / 318 | 0.103 / 0.055 | 0.060 / 0.018 |
| 2500 | 2C | 1.21 / 7.3 | 131 / 601 | 0.134 / 0.036 | 0.124 / 0.019 |
| 5000 | C/2 | 0.57 / 2.6 | 12.3 / 67 | 0.066 / 0.030 | 0.024 / 0.010 |
| 5000 | 1C | 0.80 / 2.8 | 29.4 / 214 | 0.072 / 0.051 | 0.031 / 0.014 |
| 5000 | 2C | 0.50 / 2.1 | 91.5 / 481 | 0.089 / 0.034 | 0.045 / 0.025 |
| 10000 | C/2 | 0.40 / 1.3 | 8.8 / 50 | 0.052 / 0.028 | 0.015 / 0.010 |
| 10000 | 1C | 0.59 / 2.4 | 23.9 / 176 | 0.049 / 0.048 | 0.024 / 0.013 |
| 10000 | 2C | 0.37 / 1.2 | 67.3 / 374 | 0.073 / 0.032 | 0.041 / 0.022 |

Result: D_p -3.7 %, k_n +1.1 %, **D_n -28 %**, all three misfits at the
1 mV noise floor from step 5000 on. D_n crosses the true value at step
~4000 and then drifts monotonically downwards while the voltage fit no
longer changes (figure `inverse_parameters.png` in the run folder). This
is worse than the single-rate result of 3.2 (D_p +0.8 %, k_n -0.2 %,
D_n -8.8 % after 25 000 steps), although the Cramer-Rao bounds for the
three-parameter subset are 0.25 / 0.41 / 1.5 % with the 1C data alone and
0.12 / 0.18 / 0.78 % with the three protocols (1 mV noise, same
sensitivities as 3.0). The information content of the data is not the
limit.

**Interpretation: forward-model error over sensitivity.** The fields of
the three models are far from converged at 10 000 steps (c_e max error
176 mol/m3 at 1C and 374 mol/m3 at 2C against 61 for the forward run of 2f
after 20 000 steps; j_n rmse 5-7 % against 0.7 %; th_s,n max error 0.04
against 0.005), yet the voltage is fitted to the noise. The data are
therefore fitted by fields that are wrong in a way that is cheap in the
physics loss, and the parameter with the lowest voltage sensitivity
absorbs the residual systematic voltage error. The rms sensitivities
dV/d ln p of the reference model (mV per e-fold, from the FD sensitivities
of 3.0) are:

| protocol | D_n | D_p | k_n | k_p | sigma_p | D_e | kappa_e |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C/2 | 2.5 | 41 | 38 | 7.3 | 3.4 | 15 | 11 |
| 1C | 5.0 | 73 | 45 | 16 | 6.8 | 41 | 22 |
| 2C | 11.4 | 118 | 49 | 38 | 14 | 215 | 55 |

A systematic voltage error of 0.5 mV (the converged forward PINN of 2f)
is therefore equivalent to a bias of 10 % in D_n but only 0.7 % in D_p and
1.1 % in k_n at 1C: exactly the pattern of the single-rate result (-8.8,
+0.8, -0.2 %). With fields at the 1 mV level, as here, the equivalent D_n
bias is 20-30 %. The single-rate continuation of 3.2 moved D_n from 0.865
(15k) to 0.912 (25k) while c_e max error fell from 90 to 76 mol/m3, which
is the same mechanism in the other direction: the bias shrinks as the
fields converge.

Consequences: (i) for parameters with weak voltage sensitivity (D_n, k_p,
sigma_p) the accuracy of the forward PINN, not the data, sets the error,
so the forward problem must be converged to well below the noise level
before those parameters are released (hierarchical estimation: fields
first, then the sensitive parameters, then the weak ones, implemented
with `param_release_steps` / `--init`); (ii) more C-rates help only once
the forward error at the highest rate is below the noise, and the 2C
protocol is precisely where the forward problem is hardest (c_e down to 82
mol/m3); (iii) a multi-rate run needs the per-protocol step budget of the
forward problem (20 000+ steps per model, i.e. 8+ h on two cores), which
should be run on the owner's machine or Grace; (iv) the single-rate
numbers of 3.2 should be reported with this error model: D_p and k_n are
recovered to about 1 %, D_n to about 10 %, both consistent with a 0.5 mV
forward error.

A direct check of the mechanism (`f4_biastest_true0`, below) starts the
three parameters at their true values from the converged forward model of
2f with noise-free data: if the PINN minimum is biased, D_n drifts away
from 1 although nothing else changes.

**Bias test** (`f4_biastest_true0_*`: warm start from `f4_seed0` (2f),
parameters started at their true values, noise-free 1C data, `lr` 2e-4,
`param_lr` 0.01, 4000 steps, 25 min on one core). With the true
parameters the converged forward model misfits the data by 0.54 mV rms
(its own error). The optimizer removes part of that misfit by moving the
parameters: within 1000 steps D_n settles at 0.959 (-4.1 %), D_p at 1.005
and k_n at 0.998 and they no longer move (steps 1000-4000: D_n 3.17 +/-
0.01 e-14), while the data misfit falls to 0.26 mV and the error against
the reference to 0.30 mV rms. So the minimum of the inverse problem is
biased by the forward-model error, in the direction of the least
sensitive parameter and by the amount the sensitivity rule predicts
(-4 % in D_n for the part of the 0.5 mV error that the D_n direction can
absorb); the larger errors of the from-scratch runs (-8.8 % single-rate,
-28 % multi-rate) are this bias plus the extra forward error of
less-converged fields. The practical rule for the project: the achievable
parameter accuracy is (systematic forward voltage error) / (rms voltage
sensitivity), and the forward PINN must be driven below the noise level
before weakly sensitive parameters are released.

**Predicting the bias from the forward run alone**
(`scripts/v2_bias_prediction.py`). To first order the inverse optimizer
absorbs the part of the forward error e(t) = V_pinn - V_ref that lies in
the span of the sensitivities: S delta = -e in the least-squares sense,
bias_i = exp(delta_i) - 1. With the FD sensitivities of 3.0 (1C, 301
times) and the forward runs of 2f:

| forward run | forward error rms / max [mV] | residual after fit [mV] | predicted D_p | k_n | D_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| f4_seed0 | 0.42 / 2.46 | 0.39 | +0.25 % | -0.11 % | **-3.7 %** |
| f4_seed1 | 0.50 / 2.88 | 0.46 | +0.30 % | -0.13 % | -5.2 % |
| f4_seed2 | 0.51 / 2.10 | 0.42 | +0.38 % | -0.47 % | -6.2 % |
| fac_D (soft current, section 5) | 0.64 / 4.80 | 0.50 | +0.46 % | -0.74 % | -7.5 % |

The prediction for f4_seed0 (+0.25 / -0.11 / -3.7 %) matches the bias
test started from that model (+0.5 / -0.2 / -4.1 %) in sign and size, and
the three seeds agree on a negative D_n bias of 4-6 %: the systematic
part of the forward error (the seed-correlated error fields of 2d/2f)
projects onto the D_n direction. Only ~10 % of the forward error (0.42 ->
0.39 mV) is absorbable by the three parameters; the rest stays as data
misfit. Use of the script: evaluate any forward model before an inverse
run and report the predicted bias with the estimate; a target forward
accuracy for a given parameter follows from the same algebra (D_n to 1 %
needs the projection of the forward error on the D_n direction below
~0.1 mV).

The reference mesh is not the source: the x40/r60 solution differs from
x80/r120 by 0.18 mV rms, equivalent to D_p +0.10 %, k_n -0.56 %, D_n
+0.41 %, so the discretization error of the x80/r120 reference itself is
worth well under 0.2 % in D_n.

**Where the systematic error lives** (decomposition of V_pinn - V_ref into
U_p + eta_p + [phi_e(L) - phi_e(0)] - U_n - eta_n at the collectors, 150
times). For the width-96 / 40k model of section 6 (0.30 mV rms) the
components are U_n 0.17, eta_n 0.35, U_p 0.27, eta_p 0.33, dphi_e 0.43 mV
rms, and in the last 600 s the negative-collector terms carry a persistent
offset: U_n -0.39 mV (surface stoichiometry at x = 0 too high by 8e-4),
eta_n +0.73 mV, dphi_e +0.61 mV - numbers that are the same for the
width-64 / 20k model (-0.40, +0.76, +0.75 mV) although every other error
halved. Their origin is the electrolyte concentration at the negative
collector: c_e(x=0) is under-predicted by 26-36 mol/m3 at t ~ 1200 s and
t ~ 2400 s (0 to 7 mol/m3 elsewhere in time, and only 2 mol/m3 at
x = L_n/2), i.e. a boundary layer at the zero-flux collector that the
networks smooth out exactly when the reference c_e shows the bumps
produced by the current redistribution of the graphite staging plateaus.
The c_e error enters eta_n through j0 ~ sqrt(c_e) (1.5 % of c_e -> 0.7 mV)
and dphi_e through the concentration overpotential, and because D_n acts
on the voltage through the same graphite-staging features, this error
projects onto the D_n direction (-3.6 % equivalent for both models).
Remedies to test: denser collocation at the collectors (`x_collector_fraction`,
added to the sampler), a hard zero-flux parametrization of the collector
boundaries (an even input feature such as cos(pi X)), or more time
frequencies.

*Collector sampling does not help* (`f4_coll_seed0_20261005T085815Z`,
`x_collector_fraction = 0.25`, `x_collector_width = 0.1`, otherwise the 2f
config, 20 000 steps): V 0.81 / 2.59 mV, c_e max 97, j_p max 32 % - worse
than the baseline everywhere (the points were taken from the interior),
and the collector error itself grew: c_e(x=0) -49 and -66 mol/m3 at 1200
and 2400 s (baseline -34, -36); predicted bias D_n -2.8 %, D_p +0.9 %,
k_n -1.6 %. The reference shows what the networks are missing: during the
graphite staging transitions the reaction current at the negative
collector rises to 1.09x (t ~ 1200-1300 s) and 1.32x (t ~ 2500 s) the
electrode mean and the surface stoichiometry there sweeps the plateau
edges, and the PINN under-predicts j_n(x=0) by 2-3 % around t = 2440 s in
both the width-64 and the width-96 model (0-1 % elsewhere). The c_e
bumps at the collector are the time integral of those current peaks, so
the missing feature is temporal (events ~300 s wide at t ~ 1200 and
2400 s), which points to the time features rather than to the spatial
sampling: `fourier_t = 8` was the next test.

*More time frequencies do not help either* (`f8_seed0_20261005T114019Z`,
`fourier_t = 8`, 20 000 steps, interrupted once and resumed): V 0.65 /
4.90 mV, c_e max 98, c_e(x=0) -57 / -65 mol/m3 at 1200 / 2400 s and a
predicted D_n bias of -9.2 %; worse than `fourier_t = 4` at equal budget.

*The actual mechanism: the soft zero-flux condition at the collector is
not enforced.* Evaluating the trained models at the bump times gives a
wall slope d c_e/dx(0) of +6e5 (1200 s) and +9e5 mol/m4 (2400 s) for both
the width-64 and the width-96 model, against zero in the reference, while
the normalized boundary residual `bc_flux_0` is only 4-6e-3 (its scale is
the cell-level flux, so a slope error of 7 % of the typical gradient
costs nothing in the loss); the interior mass residual near x = 0 is at
the 1e-2 level, i.e. the networks satisfy the PDE with a slightly wrong
boundary condition, and a flux of salt leaks out of the collector exactly
when the current peak there pushes c_e up. The PyBaMM fields satisfy the
v2 salt equation at the collector cells to rounding (checked at 600-2900
s), so the equations are the same. Fix implemented: `collector_bc = "hard"`
replaces the electrolyte x-features (2X-1, kinks) by cos(pi X) and
sin(pi/2 kink) features, all with zero x-derivative at X = 0 and X = 1, so
d c_e/dx = d phi_e/dx = 0 at both collectors for any network (zero salt
flux and zero electrolyte current exact; unit test
`test_hard_collector_bc_zero_slope`). Run `f4_hardbc_seed0` (section 7).

## 4. Next experiments (ordered)

1. Forward accuracy is the bottleneck of the inverse problem (3.3). Done
   in the sandbox: width 96 x 40k (0.29 mV, section 6) and hard collectors
   (section 7). Next on the owner's machine: hard collectors + width 96 x
   40k (expected ~0.2 mV), seed study of the hard-collector configuration,
   and the end-of-discharge fixes of section 7 (training domain beyond the
   data window, end-of-discharge time feature) judged with
   `v2_bias_prediction.py`.
2. Inverse, hierarchical: forward fields first (per rate), then D_p / k_n
   (sensitive), then D_n, k_p, sigma_p and the electrolyte pair, each
   released only when the forward error at the relevant rate is below the
   noise (`param_release_steps`, `--init`). Report every estimate with the
   error model of 3.3 (forward error / sensitivity) next to the CRLB.
3. C-F factorial (direct/inverse kinetics x soft/hard current), section 5
   (in progress).
4. Other C-rates and a pulse protocol; then LG M50 measured data.

## 5. C-F factorial: kinetics (direct / inverse) x electrode current (soft / hard)

Question from the handover: does the formulation matter, or would the
classic PINN (direct Butler-Volmer residual, soft current conservation)
have converged with enough steps? All four runs use the final architecture
(`configs/v2_forward_1C.json`, `fourier_t = 4`, width 64 x 4, seed 0,
20 000 Adam steps, one thread each, adaptive group weights):

| variant | kinetics residual | electrode current | run |
| --- | --- | --- | --- |
| C | direct: (j - 2 j0 sinh(eta / 2 V_T)) / j_ref | soft term (`projection=false`, group "current") | `fac_C_direct_soft_stopped10k` |
| D | inverse: (eta - 2 V_T asinh(j / 2 j0)) / V_T | soft term | `fac_D_inverse_soft_*` |
| E | inverse | hard projection of the learned j (baseline, 2f) | `f4_seed0_20261004T120413Z` |
| F | direct | hard projection | `fac_F_direct_hard_*` |

Voltage rmse against PyBaMM [mV] along training (evaluations every 2000
steps for C, D, F and every 2500 for E):

| variant | 2000-2500 | 4000-5000 | 6000-7500 | 8000 | 10000 | 12000-12500 | 15000-16000 | 20000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C direct / soft | 464 | 463 | 462 | 461 | 459 | stopped | | |
| D inverse / soft | 32.1 | 6.02 | 3.39 | 2.06 | 1.39 | 0.99 | 0.85 | **0.70** |
| E inverse / hard (baseline) | 3.87 | 1.90 | 1.16 | | 0.88 | 0.74 | 0.64 | **0.57** |
| F direct / hard | 463 | 461 | 460 | 458 | 455 | stopped | | |

Final field errors (D and E at 20 000 steps; C and F at 10 000):

| variant | V rmse / max [mV] | c_e rmse / max | phi_e max [mV] | j_n / j_p rmse/jref | th_s,n / th_s,p max |
| --- | ---: | ---: | ---: | ---: | ---: |
| C direct / soft | 459 / 816 | 425 / 1313 | 353 | 0.554 / 0.495 | 0.437 / 0.234 |
| D inverse / soft | 0.70 / 4.80 | 7.6 / 75 | 4.2 | 0.008 / 0.029 | 0.0053 / 0.0083 |
| E inverse / hard | 0.57 / 2.46 | 6.9 / 61 | 2.6 | 0.007 / 0.027 | 0.0047 / 0.0075 |
| F direct / hard | 455 / 810 | 415 / 1284 | 363 | 0.170 / 0.064 | 0.135 / 0.202 |

Reading of the factorial: the kinetics residual is the decisive factor
(direct: no convergence in 10 000 steps whatever the current treatment;
inverse: convergence in both cases). The hard current projection is a
second-order gain on top of the inverse residual: E is ahead of D by about
a factor 2 at every step up to 10 000 (3.9 vs 32 mV at 2000-2500, 0.88 vs
1.39 mV at 10 000) and ends 20 % better in voltage rmse, with half the
maximum error (2.5 vs 4.8 mV, the D maximum sits in the 30-150 s transient
where the soft current term is weakest) and the electrolyte and j errors
10-20 % lower. With the hard projection the j errors of F are 3-7x smaller
than those of C even though both are stuck (the projection fixes the
electrode-integrated current exactly), which does not help the potentials.
A probe of C with unit group weights (`probe_C_unitw`, `adaptive=false`,
4000 steps) is stuck in the same state (V rmse 466 mV at 2000 steps), so
the trap is the direct residual itself, not the gradient balancing.

**C (direct, soft) does not converge.** The kinetic loss starts at 1e10
(sinh of the random initial overpotentials), the gradient-norm balancing
drives the weight of the kinetics group to 0.1-0.5 while the other groups
reach 1e2-1e3, and the positive-electrode kinetic residual stays at ~3e3
(j_ref units squared): the potentials never enter the regime where the
Butler-Volmer relation holds, and the voltage error is 464 mV at step
2000 and 459 mV at step 10 000 (c_e max error 1300 mol/m3, theta_surf,n
max error 0.44). Stopped by hand after the step-10 000 evaluation. This
is the v1 failure mode reproduced inside the v2 code: with the direct
residual, an error of 0.2 V in eta is amplified to a residual of ~50 j_ref
by the exponential, so the optimizer cannot use the kinetic term to
correct the potentials, and nothing else in the loss defines them.

## 6. Forward problem, width 96 x 40 000 steps (`f4_w96_40k_20261005T040031Z`)

Next experiment 1 of section 4, run in the sandbox (2 threads, 0.39 s per
step, 4.7 h); same config as 2f except `width = 96`, `adam_steps = 40000`
(the exponential lr schedule therefore decays twice as slowly), seed 0.

| step | V rmse / max [mV] | c_e rmse / max | phi_e max [mV] | j_n / j_p rmse/jref | j_p max/jref | th_s,n / th_s,p max | th_s,n / th_s,p rmse |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5000 | 2.23 / 8.89 | 34.8 / 302 | 6.1 | 0.019 / 0.046 | 0.357 | 0.0150 / 0.0126 | 2.9e-3 / 1.6e-3 |
| 10000 | 2.00 / 6.04 | 18.5 / 186 | 3.2 | 0.011 / 0.044 | 0.347 | 0.0066 / 0.0136 | 1.3e-3 / 1.9e-3 |
| 20000 | 0.49 / 2.45 | 9.2 / 104 | 1.7 | 0.007 / 0.024 | 0.214 | 0.0032 / 0.0069 | 6.6e-4 / 8.2e-4 |
| 30000 | 0.35 / 1.72 | 5.8 / 68 | 1.2 | 0.006 / 0.015 | 0.149 | 0.0024 / 0.0044 | 5.1e-4 / 4.7e-4 |
| 40000 | **0.31 / 1.34** | **4.4 / 52** | **1.0** | **0.005 / 0.012** | **0.125** | **0.0020 / 0.0032** | **4.5e-4 / 3.5e-4** |
| 2f baseline (64 x 20k) | 0.54 / 2.46 | 7.0 / 62 | 2.6 | 0.007 / 0.027 | 0.230 | 0.0047 / 0.0075 | 5e-4 / 1e-3 |

(Final row of the run from `final_metrics.json`, 436 reference times:
V 0.285 mV rmse.) Every error is reduced by a factor 1.5-2.5 with respect
to the baseline: voltage 0.54 -> 0.29 mV rms and 2.46 -> 1.34 mV max,
phi_e 2.6 -> 1.0 mV max, positive reaction current 2.7 -> 1.2 % rms and
23 -> 12.5 % max (the 30-150 s transient), surface stoichiometries 2-3x.
Along training the width-96 run is behind the baseline at equal steps up
to 15 000 (slower lr decay) and overtakes it at 20 000 (0.49 vs 0.57 mV)
with the same wall time per 20 000 steps x 1.4. The one error that does
not scale down is the c_e boundary layer at the negative collector
(max 62 -> 52 mol/m3, the bumps at 1200 and 2400 s discussed in 3.3),
and with it the D_n-equivalent bias: `v2_bias_prediction.py` gives
D_p -0.00 %, k_n +0.34 %, D_n -3.6 % for this model (forward error
0.27 mV rms, 0.23 after the fit) against -3.7 % for the baseline.

**Bias test from this model** (`w96_biastest_true0_*`, same protocol as
in 3.3: true parameters, noise-free 1C data, 4000 steps): the data misfit
drops from 0.285 mV (the forward error) to 0.082 mV and the parameters
settle within 1000 steps at D_p +0.3 %, k_n +0.0 %, **D_n -3.9 %** (3.17e-14,
constant from step 1000 to 4000), against -4.1 % from the width-64 model
and the prediction of -3.6 %. Halving the forward error did not reduce the
D_n bias at all, which confirms that the bias is carried by the one error
component that did not shrink (the collector boundary layer of c_e), not
by the overall forward accuracy. The remedy must therefore target that
component (next: `x_collector_fraction`, run `f4_coll_seed0`), and the
estimate of D_n should be reported as "-4 % systematic" until it does.

## 7. Hard zero-flux collectors (`collector_bc = "hard"`, run `f4_hardbc_seed0_20261005T140419Z`)

Same config as 2f (width 64, 20 000 steps, `fourier_t = 4`, seed 0),
electrolyte x-features cos(pi X) and sin(pi/2 kink) so that d c_e/dx and
d phi_e/dx vanish at both collectors by construction (3.3 for the
diagnosis). 1.9 h on two threads.

| step | V rmse / max [mV] | c_e rmse / max | phi_e max [mV] | j_n / j_p rmse/jref | th_s,n / th_s,p max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 2.94 / 13.4 | 38.2 / 314 | 11.1 | 0.047 / 0.042 | 0.0257 / 0.0118 |
| 10000 | 0.68 / 4.22 | 6.7 / 70 | 3.4 | 0.017 / 0.026 | 0.0079 / 0.0070 |
| 20000 | **0.45 / 2.72** | **3.4 / 32** | 2.5 | 0.012 / 0.019 | 0.0054 / 0.0049 |
| 2f baseline (soft BCs) | 0.54 / 2.46 | 7.0 / 62 | 2.6 | 0.007 / 0.027 | 0.0047 / 0.0075 |

(20 000-step row from `final_metrics.json`, 436 times.) The electrolyte
error halves (c_e rmse 7.0 -> 3.4, max 62 -> 32 mol/m3) and the collector
bumps are now resolved (c_e(x=0) error -6 / -13 mol/m3 at 1200 / 2400 s
against -34 / -36), the voltage improves (0.54 -> 0.45 mV rms) and the
late-time eta_n offset of 3.3 disappears (+0.73 -> +0.00 mV). The hard
features cost the negative-electrode current some accuracy (j_n rmse 0.7
-> 1.2 %, th_s,n rmse 5e-4 -> 1.2e-3; the positive electrode improves:
j_p 2.7 -> 1.9 %, th_s,p max 0.0075 -> 0.0049).

The D_n bias, however, is unchanged: `v2_bias_prediction.py` gives
D_p +0.26 %, k_n +0.13 %, D_n -4.7 % (forward error 0.37 mV rms, 0.30
after the fit). The late-time voltage offset survives at the same level
(+0.27 mV mean for t > 2400 s, against +0.38 for the baseline and +0.27
for width 96) and is now carried by other components (U_n +0.25, dphi_e
+0.58, eta_p -0.23, U_p +0.18 mV): removing the collector error moved the
compensation elsewhere but did not remove the late-time offset. The
offset is therefore not a property of one field; a candidate common to
all runs is the periodic time features (sin/cos(2 pi k t / t_end) take
the same values at t = 0 and t = t_end, so the representation near the
end of the discharge is aliased with the hard-IC region). Test:
`fourier_period = 2` (features sin/cos(pi k t / t_end), no aliasing),
with the hard collectors (`f4_hardbc_p2_seed0`).

*Result of `fourier_period = 2`* (`f4_hardbc_p2_seed0_20261005T160150Z`,
hard collectors, half-period time features, 20 000 steps): V 0.60 / 3.18
mV, c_e 3.4 / 34, j_n 1.0 %, j_p 1.6 %, th_s,n / th_s,p max 0.0044 /
0.0042 - the fields are as good as with period 1 but the voltage is
worse, and the late-time offset is unchanged (+0.27 mV for t > 2400 s,
predicted D_n bias -4.4 %). The aliasing hypothesis is rejected.

**What the late-time error looks like.** For all four forward models
(baseline, width 96, hard collectors, hard + period 2) the voltage error
has the same shape: +0.3 to +0.8 mV between 2300 and 2700 s, a negative
excursion of -0.4 to -1.2 mV at 2900-2950 s and a positive jump of +1.3
to +3.2 mV at t = t_end = 3000 s exactly (this end point is the "max V
error" quoted for every run). The reference is smooth there (dV/dt
-0.31 -> -0.35 mV/s, no solver artefact, identical on the x40/r60 mesh),
and the window is where the graphite surface stoichiometry at the
collector reaches 0.21, i.e. the steep end of the graphite OCP, where
dV/dt accelerates and where the D_n sensitivity is largest (6-15 mV per
e-fold against 1-3 earlier). The networks under-resolve this end-of-
discharge acceleration, with the voltage network itself showing the
end-point jump. Restricting the fitted window does not help (the bias
prediction with t <= 2400 s or 1500 s moves between -1 and -8 % because
the D_n sensitivity there is small). Candidate remedies for the owner's
machine: extend the training domain beyond the data window (t_end 3300 s
so that the data end at an interior point), an end-of-discharge time
feature (analogous to the short-time features), and the seed study of
the hard-collector configuration.

*Extended domain* (`f4_hardbc_t3300_seed0_20261005T175123Z`,
`configs/v2_forward_1C_t3300.json`: hard collectors, protocol and
reference to 3300 s where V reaches 3.02 V, 20 000 steps). On the 3000 s
window the end-point jump is gone (V 0.37 mV rms, max 1.27 mV against
2.72 for the same model trained to 3000 s) and the whole-domain metrics
are V 0.74 / 5.1 mV, c_e max 61, because the last 300 s add the steep
part of the discharge. The late-time offset, however, is unchanged
(+0.38 mV mean for t > 2400 s; +0.5 to +0.6 mV at 2100-2600 s) and the
predicted D_n bias is -6.3 %: the end-point jump was not the carrier of
the bias, the 2100-2700 s offset is.

*Where this leaves the D_n question.* Five forward variants (baseline,
width 96 x 40k, hard collectors, hard + half-period features, hard +
extended domain) share a positive voltage error of 0.3-0.6 mV during
2100-2700 s, the window in which the negative-collector reaction current
rises from 0.93x to 1.32x the electrode mean (graphite plateau transition,
surface stoichiometry at the collector 0.46 -> 0.29) and in which the
PINN under-predicts j_n(x=0) by 2-3 % in every variant. The PINN
therefore under-resolves the current redistribution of the plateau
transition, which costs it part of the local overpotential and makes V
too high exactly where the D_n sensitivity is largest; the electrolyte
part of that error was removed by the hard collectors, the solid-phase
part (surface stoichiometry and j_n at the collector) remains and is the
next target: denser particle/electrode sampling in time around the
transition (a residual-based or OCP-slope-based time sampler), a time
feature built from the reference OCP slope, or simply more capacity in
the negative-electrode networks. Until then every D_n estimate from this
model carries a systematic -4 to -6 %, and D_p / k_n are good to ~0.3 %.


## 8. Step C: aged cells and the commercial-cell data (2026-10-05; root cause of the mismatch found 2026-10-06, end of section)

**Why synthetic aging data first.** The LG M50 data set shared by the owner
is itself synthetic (PyBaMM DFN with the Chen2020 parameters of our forward
model plus the four O'Kane 2022 degradation mechanisms, 12 cells, 1C/1C
CC-CV cycling at 25 C, 648-1200 cycles, capacity retention 69-99 %). Until
its per-cycle files arrive, `scripts/v2_make_aging_dataset.py` generates
the same kind of data in the sandbox (same model options and the OKane2022
parameter set, which contains Chen2020), so the whole pipeline can be built
and tested against a known truth.

**Aging parameters in the PINN.** Five new multipliers/parameters
(`DFNPINN.PARAMETERS`): `theta_n0`, `theta_p0` (the stoichiometries at the
start of the discharge: loss of lithium inventory shifts them; they define
the hard initial condition and the voltage offsets), `eps_am_n`,
`eps_am_p` (active-material fractions: loss of active material; they scale
the specific area in the electrolyte and charge sources, in the soft
current term and in the hard current projection, so the electrode current
stays exactly the applied one with the reduced area), and `R0` (lumped
series resistance, applied only to the comparison with the measured
terminal voltage; it represents the SEI/plating film resistance that the
DFN does not contain). Unit test `test_aging_parameters_enter_consistently`.
Ground truth for the synthetic cells: theta_n0/p0 from the x- and
r-averaged particle concentrations at the start of each discharge,
eps_am from the x-averaged active-material fractions, LLI/LAM/SEI
thickness from PyBaMM's summary variables (for the first cycles of the
fresh cell theta_n0 = 0.9036 and theta_p0 = 0.2667 against Chen2020's
0.9014 / 0.2700: the state after a CC-CV charge to 4.2 V is itself a
parameter to estimate).

**Pipeline** (`dfn_pinn.v2.aging`, `scripts/v2_inverse_aging.py`,
`scripts/v2_import_lgm50.py`): CC segment of the discharge -> Protocol
(current, duration) -> voltage data for t >= 100 s -> inverse run
warm-started from a fresh forward model -> `aging_result.json` with
estimates, truth and misfit. Smoke-tested end to end on a 2-cycle set; the
first real test (cycle 200 of the 200-cycle synthetic cell, LLI ~ 4 %) is
queued behind the forward run of section 9.

**Synthetic cells generated** (`results/aging_synthetic/`, coarse mesh, 200 cycles,
every 25th discharge saved; ~8 s per cycle):

| cell | degradation parameters | cycle 200: Q_CC [Ah] | t_CC [s] | LLI | LAM_n / LAM_p | theta_n0 / theta_p0 | SEI [nm] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cellA | OKane2022 as published | 4.90 (fresh 4.96) | 3529 | 0.93 % | 0.31 / 0.09 % | 0.8956 / 0.2665 | 17 |
| cellB | cracking x20, dead Li x5, LAM x20, SEI diffusivity x20 | 4.53 | 3262 | 6.7 % | 6.4 / 2.7 % | 0.8847 / 0.2666 | 78 |

(fresh cell after the first CC-CV charge: theta_n0 0.9046-0.9049, theta_p0 0.2668;
Chen2020 nominal 0.9014 / 0.2700.) cellB is the test case for the first
aging inverse (multipliers to recover at cycle 200: theta_n0 0.981,
theta_p0 0.988, eps_am_n 0.936, eps_am_p 0.973, plus an R0 for the 78 nm
SEI film); cellA is the slow-aging control.

**Identifiability of the aging parameters** (`scripts/v2_identifiability.py
--params theta_n0 theta_p0 eps_am_n eps_am_p --add-R0 --t-end 3262 --t-min 100
--dt 60`, 1C, 1 mV noise, 54 samples as in the synthetic cycle): the voltage
sensitivities are huge (rms 480, 330, 410, 370 mV per e-fold for theta_n0,
theta_p0, eps_am_n, eps_am_p; 5 mV per mOhm for R0), so the Cramer-Rao
bounds are 0.36, 0.15, 0.39, 0.24 % and 0.17 mOhm - but theta_n0 and
eps_am_n are correlated at -0.992 (one discharge mostly sees their
product, the lithium content of the negative electrode), eps_am_p/R0 at
+0.81, and the condition number is 890. With such sensitivities the
forward-error rule of 3.3 predicts biases of only ~0.1 % for a 0.5 mV
forward error; the difficulty is the near-degenerate (theta_n0, eps_am_n)
direction, which is resolved only by the small OCV features (graphite
staging) that the PINN resolves worst.

**First aging inverse** (`C_aging_cellB_c200_w64_20261005T224933Z`, cellB
cycle 200, width-64 hard-collector fresh model as start, two stages: 3000
forward steps with the fresh parameters to re-adapt the fields to the
3262 s discharge, then 8000 steps with the data, 1000-step parameter
warm-up, `param_log_bound` 0.3, R0 in [0, 1.5 mOhm]; 50 min on one thread
while the forward run occupied the other). A first attempt without stage 1
was stopped: the parameters ran away while the fields were still adapting
(data misfit 170 -> 4 mV, eps_am_n to 0.62, R0 negative).

| parameter | truth (multiplier) | estimate | error |
| --- | ---: | ---: | ---: |
| theta_n0 | 0.9815 | 1.078 | +9.8 % |
| theta_p0 | 0.9874 | 0.911 | -7.8 % |
| eps_am_n | 0.9364 | 0.843 | -10.0 % |
| eps_am_p | 0.9730 | 0.982 | +0.9 % |
| R0 | (78 nm SEI) | 1.50 mOhm (at the bound) | - |
| theta_n0 x eps_am_n (negative-electrode lithium) | 0.919 | 0.908 | -1.2 % |

Final misfit 1.48 mV rms, still decreasing; the estimates were still moving
along the degenerate valley (theta_n0 up, eps_am_n down) when the run
ended. Reading: the identifiable combination (negative-electrode lithium
content, i.e. the capacity) is recovered to ~1 % from a single 1C
discharge of a cell with 6.7 % LLI and 6.4 % LAM_n, but the LLI/LAM split
is not, and the fields were not converged enough for the small OCV
features that would split it (misfit 1.5 mV against the 0.5 mV the model
reaches on the fresh cell). Next: (i) longer stage-1 and stage-2 budgets
(the fresh model needed 20 000 steps for 0.5 mV), (ii) denser voltage
sampling (10 s instead of 60 s), (iii) release R0 and eps_am_p first, then
theta_p0, then the (theta_n0, eps_am_n) pair, (iv) add the C/100 CV tail
or a low-rate discharge of the same cycle, which separates LLI from LAM
through the OCV, as the real data set allows (it has the CC-CV discharge of
every cycle), (v) the width-96 fresh model of section 9 as start. The
hierarchical run of the chain (section 9) uses 3000 + 8000 steps with the
width-96 model.

**Second and third attempts** (`C_aging_cellB_c200_20261006T053349Z`: same
protocol with the width-96 model of section 9; `C2_aging_cellB_c200_nowarm_*`:
no parameter warm-up in stage 2, data scale 5 mV, bounds 0.5 / R0 <= 2.5
mOhm). Both land in the same valley: theta_n0 +11.5 / +11.5 %, theta_p0
-10.7 / -14.3 %, eps_am_n -11.6 / -11.3 %, eps_am_p -1.6 / -2.7 %, R0 at
its bound, negative-electrode lithium content -1.2 / -1.1 %; misfits 0.98
and 4.5 mV rms. The better forward model does not help, and neither does
removing the warm-up.

**The cause is the model form, not the PINN.** PyBaMM's own plain DFN,
run with the TRUE aged state of cycle 200 (theta_n0, theta_p0, eps_am_n,
eps_am_p from the degradation simulation, and even the true negative
porosity), misses the aged discharge by 31 mV rms / 106 mV max for cellB
and 45 / 79 mV for cellC (a cell generated WITHOUT cracking so that the
area and the active-material fraction coincide), against 100 mV rms for
the fresh parameters; the aged cell sits ABOVE the 5-parameter DFN
through most of the discharge (the best lumped resistance is negative,
-3 to -7 mOhm). So the degradation model contains 1C physics that the
(theta0, eps_am, R0) parametrization cannot represent - candidates are the
stripping of reversibly plated lithium during the discharge (it holds the
negative potential near 0 V), the SEI film resistance distribution and
the porosity change - and any estimator built on the 5-parameter DFN
(PINN or not) must compensate along the degenerate (theta_n0, eps_am_n)
direction. What is recovered robustly is the identifiable combination,
the negative-electrode lithium content (capacity), to about 1 %.

Consequences for the real LG M50 data set: (i) report capacity-type
quantities, and LLI/LAM only with a model that includes the mechanisms
that act at 1C (plated-lithium stripping, film resistance, porosity), or
from data where they are negligible (the C/100 CV tail and the rests of
every cycle, which the data set has, or low-rate check-ups); (ii) use the
degradation-pathway files as truth for the full state, not only
LLI/LAM; (iii) the identifiability analysis says the parameters are
observable to < 0.5 % once the model is right, so the limiting factor is
the physics in the model, which is the kind of question the PINN
framework was built to answer (each mechanism is one more residual).

**Root cause found (2026-10-06, `scripts/v2_aging_mismatch_diagnosis.py`,
logs `results/aging_mismatch_diagnosis*.log`).** The mechanism-by-mechanism
diagnosis (60 cycles, each O'Kane mechanism switched off in turn) gave the
SAME 56-63 mV rms mismatch with a best lumped resistance of -10 to -12
mOhm for every variant, including "no LAM" and "no lithium plating" - so
no aging mechanism is responsible. Repeating the comparison after only
2 cycles (LLI 0.08 %, eps_am 0.9997, i.e. a fresh cell) still gives 73-75
mV rms (R0 -14 mOhm) for every variant that has particle mechanics on,
and 6.5-8.6 mV (R0 +1 mOhm, the residual being the end-of-discharge mesh
and initial-state differences) for every variant without mechanics;
switching only `"stress-induced diffusion": "false"` while keeping the
swelling model brings the full model to 7.4 mV rms (R0 +1.4 mOhm). The
culprit is PyBaMM's **stress-induced diffusion** (Ai et al. 2019, eq. 12),
which it enables by default whenever `particle mechanics` is not "none":
the particle diffusivities are multiplied by 1 + theta_M (c - c_0) with
theta_M = Omega/(RT) x 2 Omega E/(9(1 - nu)) and c_0 = 0 (OKane2022). With
the OKane2022 mechanical parameters theta_M is 1.9e-5 m3/mol for graphite
(factor 1.2-1.5) and 6.6e-3 m3/mol for NMC (factor 110 at c = 17 000 and
370 at the end of discharge): the positive-particle diffusivity is
effectively 100-400 times Chen2020's 4e-15 m2/s, the NMC diffusion
polarization disappears, and the discharge sits 60-70 mV above the plain
DFN with a current-proportional signature that a fit reads as a NEGATIVE
resistance. The earlier reading (plated-lithium stripping, SEI film,
porosity) is withdrawn.

Consequences. (i) The three aging inverses of this section were fitting
data whose solid-diffusion physics differs from the PINN's DFN; the
degenerate drift along (theta_n0, eps_am_n) and the saturated R0 are the
compensation for a D_p that is 100 times too small, not an aging
identifiability limit. (ii) The real LG M50 data set was generated with
cracking, hence with mechanics, hence with stress-induced diffusion: the
same applies to it. (iii) Two ways forward, both cheap with the existing
code: generate the synthetic cells with `"stress-induced diffusion":
"false"` (then the 5-parameter aging DFN is adequate to ~7 mV and the
LLI/LAM study can proceed), and - for the real data set - add the
stress-enhanced diffusivity D_k(c) = D_k0 (1 + theta_M,k c) to the PINN's
particle residual (one line in `residuals.py`, parameters from
OKane2022), or at minimum release D_p (already an inverse parameter) in
a first hierarchical stage on the fresh cycle. The 30 s tanh current ramp
of the protocol also lags the data's current step by 20.8 s of charge
(18.5 mV rms / 128 mV max against the fresh cell at 1C; 7 mV with a 1 s
ramp): use `ramp_s` <= 1 s, or ramp the data, for the aging fits.

## 9. Best forward model: hard collectors x width 96 x 40 000 steps (`f4_hardbc_w96_40k_20261005T194145Z`)

Step A of the owner's plan, run in the sandbox (two threads, 7 h because
the cores were shared with the step-C work; three container restarts,
continued with `--resume`). Configuration: `configs/v2_forward_1C.json`
(collector_bc = hard, fourier_t = 4) with `width = 96`, `adam_steps = 40000`.

| step | V rmse / max [mV] | c_e rmse / max | phi_e max [mV] | j_n / j_p rmse/jref | th_s,n / th_s,p max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 1.03 / 4.65 | 6.5 / 55 | 3.5 | 0.012 / 0.030 | 0.0081 / 0.0083 |
| 20000 | 0.52 / 2.51 | 1.4 / 15.5 | 1.7 | 0.005 / 0.013 | 0.0033 / 0.0038 |
| 30000 | 0.24 / 1.37 | 0.8 / 8.3 | 0.9 | 0.004 / 0.010 | 0.0021 / 0.0026 |
| 40000 | **0.25 / 1.15** | **0.6 / 6.6** | **0.75** | **0.004 / 0.008** | **0.0019 / 0.0020** |
| section 6 (soft collectors, 96 x 40k) | 0.29 / 1.34 | 4.4 / 52 | 1.0 | 0.005 / 0.012 | 0.0020 / 0.0032 |
| section 7 (hard collectors, 64 x 20k) | 0.45 / 2.72 | 3.4 / 32 | 2.5 | 0.012 / 0.019 | 0.0054 / 0.0049 |
| 2f baseline (soft, 64 x 20k) | 0.54 / 2.46 | 7.0 / 62 | 2.6 | 0.007 / 0.027 | 0.0047 / 0.0075 |

(final row from `final_metrics.json`, 436 times). The two improvements
multiply: the electrolyte concentration error falls by a factor 10 with
respect to the baseline (c_e max 62 -> 6.6 mol/m3, the first run to pass
the 10 mol/m3 screen), the reaction currents by 2-3, the surface
stoichiometries by 2.5-3.7, and the voltage reaches 0.25 mV rms / 1.15 mV
max. More important for the inverse problem: `v2_bias_prediction.py` now
gives **D_p -0.01 %, k_n +0.47 %, D_n -0.93 %** (forward error 0.21 mV rms
on the 301 sensitivity times, 0.11 mV after the fit), against -3.6 to
-4.7 % for every earlier model: the late-time systematic error that
carried the D_n bias (3.3, 7) has shrunk by a factor 4 once the
electrolyte is right and the networks have the capacity and the steps to
resolve the graphite-plateau current redistribution. This model is the
recommended starting point for all inverse work (`--init
results/v2_runs/f4_hardbc_w96_40k_20261005T194145Z/final.pt`, width 96).

**Step B, hierarchical inverse from this model** (`B_hier_w96_20261006T043317Z`:
1C data with 1 mV noise, fields warm-started from the model above, fresh
optimizer with `lr` 2e-4 -> 1e-5, `param_lr` 0.01; D_p and k_n released
from step 1 at 2.5x / 0.4x, D_n released at step 2000 from 0.6x
(`param_release_steps`); 6000 steps, 1 h):

| step | D_p | k_n | D_n | note |
| ---: | ---: | ---: | ---: | --- |
| 0 | 2.50 | 0.40 | 0.60 (frozen) | |
| 1000 | 1.47 | 0.77 | 0.60 | |
| 2000 | 1.17 | 0.88 | 0.60 -> released | misfit 1.02 mV |
| 3000 | 1.07 | 0.94 | 0.73 | |
| 4000 | 1.04 | 0.97 | 0.80 | misfit at the 1 mV floor from here |
| 6000 | **1.017 (+1.7 %)** | **0.989 (-1.1 %)** | **0.872 (-12.8 %)** | lr already 1e-5 |

D_p and k_n are recovered to 1-2 % in 6000 steps; D_n is still rising
monotonically (0.60 -> 0.87, about +0.03 per 1000 steps at the end) when the
learning rate has decayed away, so its -12.8 % is a budget limit, not the
bias (predicted -0.9 % for this model). The slow D_n direction needs either
its own, larger learning rate or ~10 000 more steps at a constant lr; a
continuation from `final.pt` (`--init`, `inverse_init={}`) is the cheap
test. Note that the fields are pulled by the wrong parameters during the
run (c_e max error 6.6 -> 16 mol/m3, V 0.25 -> 0.33 mV), which is why the
hierarchical order (sensitive parameters first) matters.

**Step B, continuation** (`B2_hier_w96_cont_*`: warm start from the run
above with its estimates, fresh Adam, `lr` 1e-4 -> 5e-5, `param_lr` 0.02,
6000 more steps, 1 h):

| parameter | start (B, step 6000) | B2, step 1000 | B2, step 3000 | **B2, step 6000** | true | CRLB (1C, 1 mV) | predicted bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D_p | 1.017 | 1.009 | 1.005 | **1.001 (+0.1 %)** | 1 | 0.25 % | -0.01 % |
| k_n | 0.989 | 0.997 | 1.006 | **1.004 (+0.4 %)** | 1 | 0.41 % | +0.47 % |
| D_n | 0.872 | 0.937 | 0.982 | **0.994 (-0.6 %)** | 1 | 1.5 % | -0.93 % |

All three parameters are recovered to better than 1 % from the 2.5x / 0.4x /
0.6x start with 1 mV noise, in 12 000 inverse steps on top of the 40 000
forward steps, and the residual errors are at the level of the predicted
bias and the Cramer-Rao bound: the inverse problem for the solid-phase
parameters at 1C is solved to the accuracy the data allow. The fields stay
accurate during the inverse (eval at the end: V 0.34 mV, c_e max 7.6
mol/m3). The recipe is therefore: converged forward model (section 9) ->
release the sensitive parameters -> release the slow ones with a larger
parameter learning rate -> continue at a low field learning rate until the
estimates stop moving. Compare with the single-rate result of 3.2 on the
width-64 soft-collector model (+0.8 / -0.2 / -8.8 %): the whole gain in
D_n comes from the forward accuracy, as the error model of 3.3 predicted.


