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
- `configs/lispan_chem_design_v1.json` (+ `.samples.jsonl`), `configs/lispan_chem_v1.json`,
  `configs/lispan_chem_variants_v1.txt`; references in `data/lispan_chem_refs/lispan_chem_design_v1/` (24 x ~12 KB).
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

## Status

2026-10-22: code, tests (9 chem + 12 rate + 14 other Li-SPAN tests pass), design (644 FV runs) and the 24 references.
Smoke runs of `scripts/lispan_chem_train.py` (60 steps, sandbox): from scratch the loss goes 25.5 -> 9.0 and the
evaluation is 259 / 300 mV (select / test); the warm start begins at loss 1.7 and 51 / 44 mV (mean 47, worst 92 mV;
the wave-1 network alone at step 0: 50 mV mean, 8.6 / 7.9 / 49.9 mV at p0 0.1 / 0.3 / 1 C - the offset comes from the
different window and Z_CC); from the wave-4 network the same (51 / 44 mV).
2026-10-23: Phase 3a wave 4 analysed; recipe set (rollbacks from step 0 with the budget window, freeze at 60 000) and a
second warm start added.  Ready to submit (`sbatch slurm/lispan_chem_array.slrm`, array 0-4).
