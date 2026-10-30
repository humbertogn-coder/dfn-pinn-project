# Phase 3b: chemistry-conditioned Li-SPAN PINN

Goal: one trained network that returns the discharge of the Simanjuntak cell for any C-rate in 0.1-1 C **and** any
chemistry inside a box of parameters, so that the two-curve calibration tool fits a cell without training anything.
Phase 3a (docs/PHASE3_PARAMETRIC_PINN.md) made the C-rate an input; 3b adds the chemistry with the same machinery.

## Design (src/dfn_pinn/lispan/chem_pinn.py)

The condition vector z = (r, theta) replaces the rate argument of the Phase 3a model everywhere (the residuals,
sampler-independent physics and training loop are shared: `rate_residuals`, `train_rate` with hooks).  theta holds k
chemistry inputs in their own units; the networks see each mapped to [-1, 1] over its box, appended to the time
features like the rate.  In the kinetics the inputs act per point exactly as the scalar multipliers of the single-rate
inverse code do (`tests/test_lispan_chem_pinn.py::test_chemistry_inputs_act_like_parameter_multipliers`).

**v1 inputs** (`CHEM_INPUTS_V1`, chosen from the literature fits of Phase 2 and the identifiability results):

| Input | Meaning | Box | Why this range |
|---|---|---|---|
| dU0_2 | offset of the step-2 equilibrium potential relative to step 1 [V] | -0.12 to +0.12 | literature fits move the steps by 5-120 mV relative to each other (excluding first cycles); +-0.12 V keeps the steps in order (U0_1 - U0_2 = 0.30 V, U0_2 - U0_3 = 0.24 V nominal) |
| dU0_3 | the same for step 3 | -0.12 to +0.12 | as above |
| b | natural log of a common multiplier of the OCV slopes b_1..b_3 | x0.6 to x1.5 | fitted multipliers x0.68-0.83 (SPAN500) |
| k0_1 | natural log of the step-1 rate-constant multiplier | x0.1 to x10 | step 1 is reversible: its k0 and U0 are separately identifiable |

For the irreversible steps 2-3 only k0_m exp(F U0_m / 2RT) enters the model (identifiability study, LI_SPAN 6.4), so
dU0_2 and dU0_3 also stand for k0_2 and k0_3.

**Parameters handled analytically** (no network input; exact invariances of the model):

| Parameter | Effect | FV check (1 C, `scripts/lispan_chem_design.py --check-invariances`) |
|---|---|---|
| common shift d of U0_1..U0_3 | V + d (every rate depends on Delta phi - U0_m only) | max error 7.3e-7 V for d = 50 mV |
| contact resistance Z_CC | V - Z_CC I (a series term; no field depends on it) | max error 1.9e-16 V for +0.015 Ohm m2 |
| effective sulfur fraction w_S | capacity x w_S, current / w_S (data-axis scaling, scripts/lispan_lit_rates.py) | by construction |

So the tool fits seven parameters (dU0_2, dU0_3, b, k0_1, d, Z_CC, w_S) with a network of five conditioning inputs.
Not in v1: separate b_m (supported names `b_1..b_3`), the electrolyte transport (D_salt; the flagged literature cells
need ~10-30x slower transport and the FV solver struggles there), first-cycle shifts beyond +-0.12 V.

**Time window.**  T = t / t_end(r, theta) with ln(r t_cut) a polynomial of total degree 4 in (ln r, theta_hat) (126
coefficients for k = 4), fitted to FV cut-off times over a design of 161 chemistries - the 3-level factorial of the box
(81), its centre and two Latin hypercubes (32 + 48) - each at 0.1 / 0.2 / 0.5 / 1 C (644 discharges, all reaching the
cut-off; `configs/lispan_chem_design_v1.json`, samples cached in `.samples.jsonl`).  Fit: max error 0.90 %, rms 0.33 %;
five random 80/20 hold-out splits give at most 2.2 % (quadratic 3-3.8 %, cubic 2.6-2.8 %), so the window stops at
97 % of the fitted cut-off (`window_fraction` 0.97; the 24 evaluation cases are covered to 96.6-97.6 %).  The
network's own contact resistance is the window value Z_CC = 0.010 Ohm m2: a lower Z_CC than any to be fitted lengthens
the discharge, so the window covers the curves of the higher resistances, which are then cut at V_cut
(`discharge_curve(..., V_cut=...)`).  The cut-off time moves with the analytic parameters too (1 C: 2838 s at
0.025 Ohm m2, 3092 s at 0.010, 2924 s with U0 + 50 mV), so data beyond the window are simply not used in a fit.

**Evaluation.**  Eight chemistries (the nominal one and seven Latin-hypercube points of another seed) at 0.1, 0.3 and
1 C (0.3 C is not a window rate): 24 FV references, voltage only, in `data/lispan_chem_refs/lispan_chem_design_v1/`.
The first four chemistries are the 'select' cases (best.pt is chosen on them), the other four are test cases that
nothing in training or selection sees.

## Files

- `src/dfn_pinn/lispan/chem_pinn.py`: model (`LiSPANChemPINN`), window fit, sampler, evaluation, loading.
- `scripts/lispan_chem_design.py`: FV design, window fit (with a held-out check), evaluation references, invariance check.
- `scripts/lispan_chem_train.py`: CLI (resumes from latest.pt like the rate script; optional warm start).
- `scripts/lispan_chem_summary.py`: table of the runs (final.pt and best.pt, select / test / worst) and the gate below.
- `scripts/lispan_chem_recovery.py`: parameter-recovery test - the two-curve fit of FV curves of known chemistry with
  a trained network (noise-free and with noise; linearized standard deviations from the network Jacobian).
- `scripts/lispan_lit_v1form.py`: FV fit of a Phase 2 literature dataset with the v1 parameterization (common b, U0
  offsets relative to step 1) inside the v1 box (`--mode box`) or with wide bounds (`--mode wide`).
- `report/phase3/make_phase3b_figures.py`: figures p3b_v1_errors.png and p3b_v1_recovery.png.
- `configs/lispan_chem_design_v1.json` (+ `.samples.jsonl`), `configs/lispan_chem_v1.json`,
  `configs/lispan_chem_variants_v1.txt`; references in `data/lispan_chem_refs/lispan_chem_design_v1/` (24 x ~12 KB).
- v2: `configs/lispan_chem_design_v2.json` (+ `.samples.jsonl`), `configs/lispan_chem_v2.json`,
  `configs/lispan_chem_variants_v2.txt`, `configs/lispan_chem_eval_extra_v2.json`; references in
  `data/lispan_chem_refs/lispan_chem_design_v2/` (39 files).
- `slurm/lispan_chem_array.slrm`: one single-core job per variant; resubmit to resume after the time limit.
- `tests/test_lispan_chem_pinn.py`: spec checks, window fit, parameter mapping, equality with the rate model at the
  nominal chemistry, per-point kinetics = scalar multipliers, analytic parameters, sampler box, train/save/load/evaluate,
  warm start, vectorized window features and the t_end cache.
- Shared with Phase 3a: `rate_pinn.train_rate` (now with hooks: model, sampler, residuals, evaluation, best.pt score),
  `rate_residuals` (generic condition columns).

## Training recipe v1 and cost

Phase 3a recipe as settled by waves 3-4 (docs/PHASE3_PARAMETRIC_PINN.md, "Wave 4 results"): no weight guard; rollback
with the best-recent-level baseline from step 0 and at most 10 rollbacks per 20 000 steps; adaptive weights frozen at
step 60 000; best.pt chosen on the select cases; the Phase 3a collocation points (1536 / 1024 / 192) and 150 000 steps.
Phase 3a showed that a single run fails about half of the time (a chaotic charge-group collapse), so the procedure is
multi-start with selection: five runs (`configs/lispan_chem_variants_v1.txt`), three seeds from scratch and two warm
starts (`init_from` = the two 0.15 mV Phase 3a networks, rate_w96_120k_s0 of wave 1 and rate_v4_s0 of wave 4; lr 3e-4).
`warm_start_from_rate` copies every layer of the rate networks; the first layers get the rate weights on the shared
columns and zero on the chemistry columns, so at step 0 the chem network is the rate network for every chemistry.  A
warm start skips the early phase in which the Phase 3a collapses happened (11 000-20 000 steps from scratch).

Step time: 0.53 s on the sandbox (the Phase 3a model: 0.49 s there and 0.54 s on Grace), so ~0.6 s and ~25 h per run on
one Grace core, inside the 36 h limit; five runs ~125 SU.  Two changes made this possible (first version: 0.88 s): the
window polynomial is built degree by degree (`window_features_torch`) and `t_end` is cached per condition tensor - one
residual evaluation asks ~60 times for the window of the same 6 tensors (~2e5 rows per step).

## Acceptance (proposal)

1. Test cases: mean voltage error < 2 mV, every case < 5 mV.
2. No penalty off the selection points: test mean <= 1.5 x select mean.
3. Multi-start: at least two of three seeds pass 1-2 on final.pt or best.pt.

## v1 results (Grace, array 20278534, finished 2026-10-25)

25.5-27.7 h per run, MaxRSS ~1.0 GB, ~134 SU.  Voltage rms error vs FV [mV] over the 4 select chemistries (12 cases)
and the 4 test chemistries (12 cases, never used in training or selection); "outside cliffs" leaves out +-1 % of the
capacity around every point where the FV voltage falls faster than 5 V per discharge; cliff timing = shift of the
mid-voltage crossing of each cliff, % of the capacity (12 cliffs in the 24 cases).

| Run | final select / test / worst | best.pt (step) | outside cliffs select / test / worst | cliff timing max / median | rollbacks | frozen weights inventory / charge |
|---|---|---|---|---|---|---|
| chem_v1_s0 | 2.35 / 3.67 / 8.83 | 1.72 / 3.69 / 13.1 (45k) | 2.20 / 1.63 / 4.90 | 0.85 / 0.50 % | 1 | 136 / 115 |
| chem_v1_s1 | 1.88 / 3.39 / 8.08 | 1.87 / 3.38 / 8.11 (145k) | 1.78 / 1.37 / 3.30 | 0.94 / 0.43 % | 6 | 72 / 286 |
| **chem_v1_s2** | **0.90 / 1.82** / 7.26 | 0.89 / 1.89 / 7.05 (130k) | **0.88 / 0.76 / 1.97** | 0.61 / 0.11 % | 3 | 0.21 / 896 |
| chem_v1_warm_s0 | 1.34 / 2.55 / 6.79 | = final | 1.31 / 1.36 / 3.01 | 0.57 / 0.14 % | 0 | 0.21 / 920 |
| chem_v1_warm_s1 | 1.30 / 2.53 / 7.38 | 1.25 / 2.52 / 7.73 (125k) | 1.27 / 1.32 / 2.96 | 0.62 / 0.14 % | 0 | 0.21 / 918 |

Gate as proposed: 0 of 5 pass (every run has a test case above 5 mV, p7 at 1 C).  Findings
(`report/phase3/figures/p3b_v1_errors.png`):

1. **The error sits at voltage cliffs.**  When the reduction steps are well separated and the OCV slope is small
   (test chemistries p5: b x0.76, k0_1 x6.6, dU0_2 +0.105 V; p7: b x0.61, dU0_3 -0.104 V), the FV voltage falls by
   0.15-0.3 V within ~1 % of the capacity at the end of a plateau.  The network places these cliffs 0.1-0.6 % of the
   capacity early or late, which is a 60-100 mV error for a few points and up to 7 mV rms for the case.  Outside the
   cliffs the best run (chem_v1_s2) is at 0.88 / 0.76 mV (select / test), worst case 1.97 mV: no loss on unseen
   chemistries.
2. **Run quality follows the Phase 3a weight signature**: s0 and s1 froze with the inventory weight at 72-136 and the
   charge weight at 115-286 (1.4-2.2 mV outside cliffs); s2 and both warm starts froze at inventory 0.21 / charge ~900
   (0.8-1.4 mV).  The warm starts had no early collapse but stopped at ~1.3 mV; s2 is the best by the select AND the
   test cases.
3. **The chemistry inputs cost accuracy**: at the nominal chemistry s2 is at 1.23 / 0.30 / 0.75 mV (0.1 / 0.3 / 1 C)
   vs 0.31 / 0.08 / 0.34 mV for the Phase 3a network of the same size; the errors were still falling slowly at 150 000
   steps (s2 test 2.04 mV at 90 000 -> 1.82 at 150 000).
4. **Parameter recovery** (`scripts/lispan_chem_recovery.py`, chem_v1_s2 final.pt, `p3b_v1_recovery.png`): two-curve
   fit (0.1 + 1 C, points up to 93 % of the cut-off) of the FV curves of the 8 evaluation chemistries for dU0_2, dU0_3,
   b, k0_1, the common shift d and an extra contact resistance dZ (w_S held at 1), started at the box centre.
   Noise-free (= error caused by the network): dU0_2 / dU0_3 0.1-7.4 mV (median ~2.3), b 0.2-1.8 %, k0_1 0.4-17 %, d
   0.0-8.0 mV, dZ 0.1-3.5 Ohm cm2 (of 100).  Against the scatter of the same fit with measurement noise (linearized sd,
   confirmed by 10 noise draws): median bias / sd ~1 at 5 mV noise (0.4-1.9 by parameter), at most ~6 (the cliff
   chemistries p5, p7 and p4 with k0_1 at x0.12, near the box edge); at 2 mV noise the median ratio is 1-5.  The
   network's error therefore matters only for data cleaner than ~5 mV (noise plus model-form error); the Phase 2
   literature fits have 10-40 mV misfits.
5. **The v1 box does not cover the literature cells** (`scripts/lispan_lit_v1form.py`; FV fits, same data and model as
   Phase 2, two rates each).  Free fit = scripts/lispan_lit_rates.py with separate b_m:

   | Dataset, rates | free fit | v1 form in the v1 box (at a bound) | v1 form, wide bounds |
   |---|---|---|---|
   | Hao 2025 SPAN, 0.1 + 2 C | 35.1 mV | 37.5 mV (dU0_2 = +0.12) | 33.6 mV: dU0_2 +0.265 V, dU0_3 +0.057, b x1.39, k0_1 x33 |
   | Chen 2019 ether, 0.4 + 2 A/g | 11.1 mV | 23.4 mV (dU0_3, b, k0_1) | 15.2 mV: dU0_2 +0.51 V, dU0_3 +0.34, b x2.0, d +0.125 V |
   | Chen 2019 carbonate, 0.4 + 2 A/g | 20.0 mV | 34.4 mV (dU0_3, b, k0_1, Z_CC) | ~25 mV, not converged at the end of the session: dU0_2 +0.5...+0.7 V, b x1.9, Z_CC at its lower bound |

   Of the 22 Phase 2 / SPAN-chemistry free fits only 3 fall inside the v1 box (per-step b_3 x1.4-3.4 in the Chen and
   Hao fits, x1.2-1.5 in Ma, dU0_2 up to +0.27 V, k0_1 up to x400 in flat valleys).  The contact resistances of the Chen and
   Hao two-rate fits (0-0.0053 Ohm m2) are below the network's 0.010 Ohm m2, so those discharges run past the
   network's time window (the last part of the curve cannot be fitted).

**Use of v1.**  chem_v1_s2/final.pt is the v1 network: ~1 mV outside cliffs, cliff positions within 0.6 % of the
capacity, parameter errors at the level of a 5 mV measurement.  It serves cells whose fits fall inside the box (the
synthetic tests, the Simanjuntak-type cells); it cannot serve most of the literature cells.

**v2 (proposal, not built).**  (1) Network contact resistance 0, so that the window covers every Z_CC >= 0; (2) a box
from the literature fits: dU0_2 -0.15...+0.30 V, dU0_3 -0.20...+0.30 V, k0_1 x0.1...x100, b split into b_12 (steps
1-2) and b_3 (x0.5...x3.5), i.e. five chemistry inputs (the two-rate free fits of Hao and Chen fall inside it
except that they put b_1 and b_2 0.1-0.35 apart; with one common b, Chen ether would need dU0_2 = +0.5 V); (3) the training changes that the v1 errors point to: more
steps (200 000-250 000), denser sampling where the residuals are large (the cliffs), multi-start as before.  The wider
box makes the cliffs sharper and more frequent, so v2 needs the revised acceptance below and a new FV design (window
fit and references; ~3-4 h on the sandbox).

**Acceptance, revised (proposal).**  The voltage rms at a near-vertical segment measures a time shift, not a voltage
error.  Proposed gate for v1/v2: (1) test chemistries outside cliffs: mean < 2 mV, every case < 5 mV; (2) cliff
positions within 1 % of the capacity; (3) parameter recovery: noise-free errors below the 5 mV-noise standard
deviation for the median chemistry and below 3 sd for every chemistry; (4) at least two runs pass.  By this gate
chem_v1_s2 passes (1)-(2) and the median part of (3), and fails the worst-chemistry part of (3) (p5, p7: up to 6 sd).

## v2 (built 2026-10-25, not trained yet)

What changed with respect to v1, and why (the v1 findings above):

| Item | v1 | v2 | Reason |
|---|---|---|---|
| Chemistry inputs | dU0_2, dU0_3, b, k0_1 | dU0_2 (-0.15...+0.28 V), **dU0_3rel2** (step 3 relative to step 2, -0.26...+0.20 V), **b_12** (x0.5-1.6), **b_3** (x0.4-3.5), k0_1 (x0.1-100) | box of the Phase 2 literature fits; with dU0_3rel2 the step-1/2 gap stays >= 0.02 V and the step-2/3 gap 0.04-0.50 V, so no point of the box has the steps out of order; a separate step-3 slope is what most free fits need |
| Network contact resistance | 0.010 Ohm m2 | **0** | the window then covers every Z_CC >= 0 (literature two-rate fits 0-0.005 Ohm m2) |
| Window | degree 4 in 5 variables, 644 FV runs, fraction 0.97 | degree 4 in 6 variables (210 coefficients), **1612 FV runs** (403 chemistries: 3-level factorial 243, LHS 64 + 96, x 0.1/0.2/0.5/1 C, all reach the cut-off), fraction **0.95** | fit max 3.6 %, rms 0.9 %, held-out max 4.7 %; the largest errors sit at the corner dU0_2 = -0.15, dU0_3rel2 = -0.26 at 1 C.  Degree 5 (462 coefficients) fits the design better (held-out 2.7 %) but misses an evaluation chemistry by 5 % (p6: steps bunched together), so degree 4 is kept; the 39 evaluation cases are covered to 92.7-96.7 % |
| Evaluation | 8 chemistries x 3 rates (4 select, 4 test) | **13 chemistries** x 0.1 / 0.3 / 1 C: nominal + 4 select, 5 test, and 3 **literature-like test chemistries** (the free FV fits of Hao 2025 0.1+2 C and Chen 2019 ether 0.4+2 A/g, b_12 = geometric mean of b_1 and b_2; the SPAN300 cycle-2 fit of Wang 2026 SI Fig. 21) | coverage of realistic regions; `configs/lispan_chem_eval_extra_v2.json` |
| Metrics | voltage rms | + **outside-cliff rms and cliff shift** per case (`cliff_metrics`, logged at every evaluation) | the revised gate |
| Sampling | uniform | **residual-based** (`ChemRARSampler`): every 250 steps the kinetics residuals (ode1-3 + charge_s) are evaluated on 8192 uniform cathode points; 25 % of every cathode batch is drawn from the 1024 worst, with a small jitter (T 0.003, Y 0.05, ln r 0.02, 1 % of each box width; reflected at the ends - T = 0 exactly gives non-finite time derivatives, found in the smoke test) | the v1 error is at the cliffs, where these residuals peak |
| Steps | 150 000 | **200 000** | v1 errors still falling at 150 000 |
| Initial weights | torch's default seed for every run (the model was built before train_rate seeded torch) | follow the run's seed | real multi-start |

Unchanged: rollback from step 0 (10 per 20 000 steps), weights frozen at 60 000, inventory term from 40 000, best.pt on
the select cases, 1536 / 1024 / 192 points.

Variants (`configs/lispan_chem_variants_v2.txt`): chem_v2_s0/s1/s2 (residual-based sampling), chem_v2_norar_s0 (same
seed without it), chem_v2_warm_s0 (from the wave-4 Phase 3a network rate_v4_s0, lr 3e-4).

Cost: 0.61 s per step on the sandbox with residual-based sampling (0.56 without; one pool refresh 2.3 s = 0.009 s per
step), so ~0.65-0.7 s and ~36-39 h per run on one Grace core; slurm limit raised to 48 h; five runs ~190 SU.

Smoke tests (sandbox, 40-100 steps): from scratch with residual-based sampling 0 skipped batches, evaluation 321 / 255
mV (select / test) at step 100 (the pool's worst-to-median ratio is only ~1.7 that early); warm start 112 / 153 mV at
step 40 (pool ratio 35: the warm network's residuals are localized).

Acceptance (revised gate, `scripts/lispan_chem_summary.py`): test chemistries outside cliffs mean < 2 mV and every case
< 5 mV; cliff positions within 1 % of the capacity; outside-cliff test mean <= 1.5 x select; at least two runs pass;
then the parameter recovery of the best run (`scripts/lispan_chem_recovery.py --design configs/lispan_chem_design_v2.json`)
compared with the 5 mV-noise standard deviation.

## v2 results (Grace, array 20288770, finished 2026-10-27)

34.8-37.9 h per run, MaxRSS 0.94-1.11 GB, ~181 SU.  Voltage rms error vs FV [mV] over the 5 select chemistries (15
cases) and the 8 test chemistries (24 cases, 9 of them the literature-like ones); "out" = outside the voltage cliffs.

| Run | final select / test / worst | out select / test / worst test | cliff shift max | rollbacks | frozen inventory / charge | revised gate |
|---|---|---|---|---|---|---|
| **chem_v2_norar_s0** | 2.00 / 1.78 / 7.13 | **1.02 / 1.15 / 2.41** | **0.32 %** | 2 | 0.21 / 776 | PASS |
| chem_v2_s0 | 3.10 / 2.16 / 8.98 | 1.51 / 1.47 / 2.75 | 0.45 % | 2 | 0.21 / 726 | PASS |
| chem_v2_warm_s0 | 2.12 / 2.32 / 7.08 | 1.48 / 1.59 / 4.70 | 0.39 % | 0 | 0.21 / 738 | PASS |
| chem_v2_s1 | 204 / 153 / 334 (collapsed) | 206 / 153 | - | 8 | 817 / 2.3 | FAIL |
| chem_v2_s2 | 315 / 239 / 442 (collapsed) | 318 / 239 | - | 15 | 469 / 5.3 | FAIL |

Revised gate (voltage part): 3 of 5 runs pass -> PASS.  Strict gate: 0 of 5 (the cliffs).  Findings
(`report/phase3/figures/p3b_v2_errors.png`, `p3b_v2_recovery.png`):

1. **The wider box costs little.**  The best run (chem_v2_norar_s0) is at 1.02 / 1.15 mV outside the cliffs (v1, on a
   much smaller box: 0.88 / 0.76), worst test case 2.41 mV, and places every cliff within 0.32 % of the capacity (v1:
   0.61 %).  The three literature-like chemistries are at 0.4-1.4 mV outside the cliffs; the nominal one at 0.38 / 0.39
   / 0.79 mV (0.1 / 0.3 / 1 C).
2. **Residual-based sampling did not help.**  Same seed and initial weights: without it 1.02 / 1.15 mV and 0.32 %, with
   it 1.51 / 1.47 mV and 0.45 %; two of the four runs with it collapsed, the run without it did not (one pair and five
   runs, so no proof of harm, but no sign of benefit either).  Its pool was only ~4-5 times the median residual: the
   kinetics residuals are not localized enough at the cliffs for it to concentrate points there.
3. **Collapses are the Phase 3a mechanism again**: s1 and s2 froze with the charge weight at 2-5 and the inventory weight
   at 470-820 (good runs: 0.21 / ~750).  Multi-start handled it (3 of 5 good).
4. **Every run is at ~30-130 mV until the inventory term switches on at 40 000 steps**, then drops to ~5 mV within
   5 000 steps (panel c): a fifth of the budget is spent before the term that makes the solution unique is active.
5. **Parameter recovery** (`scripts/lispan_chem_recovery.py --design configs/lispan_chem_design_v2.json`, chem_v2_norar_s0,
   0.1 + 1 C, 7 parameters, 13 chemistries, 10 noise draws; empirical / linearized sd: median 0.98).  Median
   noise-free error over the chemistries: dU0_2 1.4 mV, dU0_3rel2 2.1 mV, b_12 0.3 %, b_3 0.5 %, k0_1 3.4 %, d 0.4 mV,
   dZ 0.9 Ohm cm2.  Bias / sd at 5 mV noise: median 0.75 (by parameter 0.5-1.3), 90 % of the values below 3; the three
   literature-like chemistries at most 1.5.  The exceptions: p5 (k0_1 x0.115 and dU0_3rel2 -0.255, both at the edge
   of the box; 4-7 sd, k0_1 +83 %, d -31 mV, the k0_1 / d / dU0 valley) k0_1 at p1 / p7 (x45 / x93, near-equilibrium
   step 1, -37 %, 2-2.5 sd) and dZ at p1 (-3.1 Ohm cm2, 4.9 sd).  Revised gate part 3: the median passes, "every chemistry < 3 sd" fails at the box corner.

**Decision.**  chem_v2_norar_s0/final.pt is the network for the two-curve tool (v2).  Inside the realistic region its
error is at or below that of a 5 mV measurement; fits that land near the box edge (k0_1 near x0.1 or x100, dU0_3rel2
near -0.26) must be flagged.  For further training: no residual-based sampling; try the inventory term earlier
(e.g. from 15 000-20 000 steps; in single-rate training it fought the extent ODEs when active from step 1).

## v3: wider rate range (built 2026-10-27, not trained yet)

Purpose: the v2 network covers 0.1-1 C, so a two-curve fit can only use currents within a factor of 10.  The fast
curves of the literature map above 1 C at their w_S (docs/TWO_CURVE_TOOL.md, "The rate range").  v3 keeps everything
of v2 that worked and changes the rate range, plus three recipe items.

| Item | v2 | v3 | Reason |
|---|---|---|---|
| Rate range | 0.1-1 C (x10) | **0.08-3 C** (x37.5) | the pairs in the table below become possible |
| Chemistry inputs and box | dU0_2, dU0_3rel2, b_12, b_3, k0_1 | unchanged | change one thing at a time |
| FV design | 403 chemistries x 0.1/0.2/0.5/1 C = 1612 runs | same 403 chemistries x **0.08/0.1/0.2/0.5/1/2/3 C = 2821 runs** (the 1612 reused), all reach the cut-off | |
| Window | degree 4 (210 coef.), max 3.6 %, held-out 4.7 %, fraction 0.95 | **degree 6** (924 coef.), max 1.6 %, held-out 4.0 % (worst over-prediction 3.1 %), fraction **0.95**; evaluation cases covered 92.4-97.9 % | degree 4 over 0.1-3 C: max 6.3 %, held-out 8.8 %, which would have forced a fraction of ~0.91 |
| Evaluation | 13 chemistries x 0.1/0.3/1 C | same chemistries x **0.1/0.3/1/3 C** (52 references; the 39 of v2 reused, identical) | the high end of the range |
| Residual-based sampling | on (s0-s2, warm) | **off** | no benefit in v2 (same-seed pair) |
| Inventory term start | 40 000 | **20 000** (one variant keeps 40 000) | v2 runs sat at 30-130 mV until it started |
| Steps | 200 000 | **250 000** | wider range |
| Warm start | from the 3a rate network rate_v4_s0 | from the **v2 tool network chem_v2_norar_s0** (same inputs; `warm_start_from_rate` now also copies chem checkpoints) | smoke: 18.3 mV mean after 60 steps vs ~280 mV from scratch |

FV feasibility at high rates (60 chemistries of the box, 20/10 cells):
- 1.5, 2 and 3 C all reach the cut-off.
- The salt stays within 0.64-1.38 x c0 at 3 C, inside the network's +-50 % salt bound.
- Capacity at 3 C is 811-1254 mAh/g (median 1001).
- One case stalled at the default tolerance: the litHao chemistry at 1.5 C.  It solves in 6 s at rtol 1e-5.
  `scripts/lispan_chem_design.py` therefore now retries with a time limit (rtol 1e-6, then 1e-5, then 24/12 cells);
  one of the 1209 new design runs needed the retry (a 3 C case).

Speed: with a degree-6 window, the per-condition window features (924) took 2 x 0.15 s per step on the quadrature-
expanded conditions (18 432 rows).  `_t_end` now evaluates the window once per row when the condition tensor is a
quadrature expansion (detected exactly; test added).  The sandbox now runs 0.39-0.43 s per step (v2: 0.56), so
250 000 steps take ~30-35 h on one Grace core, five runs ~150-175 SU.

Variants (`configs/lispan_chem_variants_v3.txt`):
- chem_v3_s0, chem_v3_s1, chem_v3_s2;
- chem_v3_ct40k_s0: seed 0 with the inventory term from 40 000;
- chem_v3_warm_s0: from `results/lispan_chem_runs/chem_v2_norar_s0/final.pt`, lr 3e-4.

Smoke tests (sandbox, final design): from scratch 100 steps, 0 skipped batches, 315 / 255 mV select / test.  Warm
start 60 steps: 20.9 / 16.7 mV (mean 18.3, worst 55 mV), 0 skipped batches (a 120-step warm smoke on the 0.1-3 C
design had one).  The salt bound is c0 (1 + 0.5 tanh) at the fast end (ce_scale capped at 0.5).

Two-curve pairs that become possible (model rate = 0.767 i / w_S at the Phase 2 w_S):

| Data | Pair | Model rates |
|---|---|---|
| Hao 2025 (w_S ~0.67) | 0.1 + 2 C | 0.081 / 1.62 C (the Phase 2 pair; 0.1 C sits near the lower limit) |
| Chen 2019 (w_S ~1.2) | 0.4 + 2 or 4 A/g | 0.26 / 1.3 or 2.6 C (6 and 10 A/g stay outside) |
| Ma 2026 SPAN / SPAN-A (w_S 0.7-1.1) | 0.2 + 1 C | 0.23-0.36 / 1.2-1.8 C (2 C only for SPAN-A) |
| Simanjuntak 2024 | 0.1 + 1 C | w_S free (v2: fixed at 1 by the range) |

Acceptance as v2 (revised gate, `scripts/lispan_chem_summary.py`), then the known-truth test of the tool
(`scripts/lispan_two_curve_synthetic.py --network <v3 run> --design configs/lispan_chem_design_v3.json`) to set the new
network's error floor (`NETWORK_SD` in two_curve.py) and the literature validation with the wider pairs.

## v3 results (Grace, array 20311464, finished 2026-10-29)

Five runs, 37.8-46.8 h each (s0 ran on a slower node), MaxRSS ~1.0-1.1 GB.  Run folders and logs:
`phase3b_chem_runs_v3.tgz` in the owner's folder; figure `report/phase3/figures/p3b_v3_errors.png`.

| Run | final out-of-cliff select / test [mV] | worst test | cliff shift max | note |
|---|---|---|---|---|
| chem_v3_s1 | 4.76 / 4.69 | 13.6 | 1.23 % | best |
| chem_v3_s0 | 4.93 / 4.77 | 13.3 | 1.25 % | |
| chem_v3_s2 | 5.14 / 7.95 | 26.5 | 2.18 % | |
| chem_v3_warm_s0 | 7.12 / 7.81 | 42.1 | 1.55 % | from v2: fast start (8 mV at 40k), then the same plateau |
| chem_v3_ct40k_s0 | 157 / 148 | 256 | 3.75 % | collapsed at ~115k (inventory residual 81 -> 5e4) |

Revised gate: 0 of 5 pass (v2: 3 of 5 at 1.02 / 1.15 mV).

Where the error is:
1. **By rate** (chem_v3_s1, mean over the 13 chemistries outside cliffs): 0.1 C 5.6, 0.3 C 3.6, 1 C 1.9 and
   3 C 7.9 mV.  The v2 network gives 1.1, 0.9 and 1.3 mV at 0.1, 0.3 and 1 C.  The ends of the range suffer, the
   middle much less.
2. **Along the discharge:** a smooth drift that grows in the last ~30 % (step 3), not at the cliffs.  At 0.1 C it is
   +30 mV for p2, -20 mV for p1 and -33 mV for p8 in the last 20 %.  At 3 C there is an offset growing from ~2 to
   ~11 mV.
3. **Cause:** the charge stored in the extents drifts away from the charge passed.  The inventory error of
   chem_v3_s1 is 1-25 mol/m3 (worst at 0.08-0.1 C for a chemistry with dU0_2 +0.2 V and b_3 x2.7, and at 3 C).  For the v2 network it is
   0.5-2 mol/m3.  The point-wise extent-ODE residuals are of similar size (1.5e-3 vs 0.9e-3 in v2), but now
   systematic, so they integrate into a drift.
4. **The plateau is systematic.**  All seeds and the warm start level off at ~4.7-8 mV after ~150k steps.
   Sandbox continuations of chem_v3_s1 (3000 steps, lr 1e-4, frozen weights) did not change it:
   - with the v3 weights: test 4.46 mV;
   - with the inventory weight raised 18x: test 4.64 mV, and the inventory residual did not move (its gradient is
     present and large).
   So more steps or more inventory weight alone will not fix it.  What remains is network capacity or sampling
   density for the 37.5x range, or a structural fix: a hard charge-inventory constraint, the analogue of the
   current projection.
5. **Inventory switch-on:** the start at 20 000 steps worked in 4 of 4 runs.  The paired 40 000 run collapsed late,
   which is weak evidence (n = 1) for the earlier start.

Usability (known-truth test of the tool with chem_v3_s1, 0.2 + 2 C, w_S free; `results/lispan/two_curve/
synthetic_v3s1_0.2+2.json`).  Noise-free rms errors over the 9 unflagged chemistries:

| network | w_S | dU0_2 | dU0_3rel2 | b_12 | b_3 | k0_1 | d | Z_CC |
|---|---|---|---|---|---|---|---|---|
| chem_v3_s1 | 0.34 % | 5.4 mV | 5.1 mV | 2.4 % | 2.9 % | 24 % | 4.5 mV | 2.9 Ohm cm2 |
| v2 | 0.14 % | 0.6 mV | 2.1 mV | 0.2 % | 0.7 % | 12 % | 0.6 mV | 1.3 Ohm cm2 |

The v3 errors are 2-8x the v2 floor, but comparable to or below the statistical sd of real two-curve fits (Chen,
Hao: w_S 0.5-1 %, dU0_2 5-9 mV, d 4-8 mV, dU0_3rel2 23-35 mV, k0_1 16-27 %, Z_CC 5-9 Ohm cm2).  Added as the v3 floor
(`NETWORK_SD["chem_v3_s1"]`): with it 88-100 % of the 5 mV-noise errors lie within 2 sd (without it 58-92 %).

Decision:
- chem_v3_s1/final.pt is a usable, lower-accuracy network for the fast literature pairs that v2 cannot take, with
  its floor in `sd_total`.
- v2 stays the tool network whenever the currents fit in 0.1-1 C.
- A better wide-range network needs a change of method (see the options in the status line), not more of the same.

New training options (rate_pinn.RateTrainConfig): `initial_weights` (group weights of a new run) and
`freeze_weight_min` (lower bounds at the freeze), used by the continuation probes; tested.

## Status

2026-10-22: code, tests (9 chem + 12 rate + 14 other Li-SPAN tests pass), design (644 FV runs) and the 24 references.
Smoke runs of `scripts/lispan_chem_train.py` (60 steps, sandbox): from scratch the loss goes 25.5 -> 9.0 and the
evaluation is 259 / 300 mV (select / test); the warm start begins at loss 1.7 and 51 / 44 mV (mean 47, worst 92 mV;
the wave-1 network alone at step 0: 50 mV mean, 8.6 / 7.9 / 49.9 mV at p0 0.1 / 0.3 / 1 C - the offset comes from the
different window and Z_CC); from the wave-4 network the same (51 / 44 mV).
2026-10-23: Phase 3a wave 4 analysed; recipe set (rollbacks from step 0 with the budget window, freeze at 60 000) and a
second warm start added.  Ready to submit (`sbatch slurm/lispan_chem_array.slrm`, array 0-4).
2026-10-25: v1 trained (5 runs; results above).  Gate as proposed: 0 of 5; outside the cliffs chem_v1_s2 reaches
0.88 / 0.76 mV (select / test).  Parameter recovery and the literature coverage check done; v1 box too narrow for
most literature cells; v2 proposed, waiting for the owner's decision.
2026-10-25 (later): v2 built (code, 1612-run FV design, 39 references, configs, slurm, tests 13 chem + 12 rate + 14 other
Li-SPAN pass); ready to submit (`sbatch slurm/lispan_chem_array.slrm`, now the v2 default).
2026-10-27: v2 trained and analysed (results above); chem_v2_norar_s0/final.pt chosen for the tool.  Next: the two-curve
tool (fit with w_S on real data, uncertainty, validity flags) validated against FV fits of the literature datasets in
the v2 parameterization.
2026-10-27 (later): two-curve tool built and validated (docs/TWO_CURVE_TOOL.md).  On six published datasets the network
fit matches an FV fit of the same points within 1 sd in 45 of 47 parameters (the two exceptions: b_3 at the box edge,
flagged) and within 0.6 mV in misfit, at 12-60x lower cost.  Usable rate pairs are limited by the network's 0.1-1 C
range (current ratio <= 10); the literature fast rates (2 C and above) need a wider-range network.
2026-10-27 (night): v3 built (0.08-3 C; section above): FV design (2821 runs), degree-6 window, 52 references,
configs, slurm default, warm start from chem checkpoints, window speed-up; 46 Li-SPAN tests pass.  Ready to submit
(`sbatch slurm/lispan_chem_array.slrm`, array 0-4).
2026-10-29: v3 trained (results above): 0 of 5 pass; best chem_v3_s1 4.8 / 4.7 mV outside cliffs, a late-discharge drift
from the charge inventory; usable for wide pairs with its floor (2-8x v2's).  Options for a better wide-range network:
(a) hard charge-inventory constraint in the extents (structural, analogous to the current projection), (b) width 128
and/or 2x collocation points (capacity / density), (c) two overlapping networks (e.g. 0.08-1 C = v2 extended, and
0.3-3 C).  Waiting for the owner's choice.
