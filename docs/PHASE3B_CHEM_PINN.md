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
