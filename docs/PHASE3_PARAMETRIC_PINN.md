# Phase 3a: rate-parametric Li-SPAN PINN

Goal: one trained network that returns the discharge curve (and the internal fields) of the nominal Simanjuntak cell at
**any C-rate in a range** in one forward pass, instead of one training per rate.  This is the forward half of the
two-curve calibration tool; Phase 3b adds 3-5 chemistry parameters as further inputs.

## What changed with respect to the single-rate PINN (`src/dfn_pinn/lispan/pinn.py`)

| Item | Single-rate PINN | Rate-parametric PINN (`src/dfn_pinn/lispan/rate_pinn.py`) |
|---|---|---|
| Inputs | position, time | position, time, **C-rate r** (feature 2 (ln r - ln r_min)/(ln r_max - ln r_min) - 1) |
| Time coordinate | T = t / t_end | T = t / t_end(r), with t_end(r) = 0.98 exp(c0 + c1 ln r + c2 ln^2 r) / r |
| Window t_end | 0.98 x FV cut-off time | quadratic fit of ln(r t_cut) to FV cut-off times at `window_rates` (0.1, 0.2, 0.5, 1 C: max error 0.49 %) |
| Scales (residuals, c_e and phi_e outputs, ramp, short-time features, IC factor) | from the protocol current | per point, from I0(r) = 10 r A/m2 |
| Collocation points | 768 / 512 / 96 | 1536 / 1024 / 192 (one more input dimension) |
| Adam steps | 30 000 | 60 000 |
| Charge-inventory residual from step | 10 000 | 20 000 |

Everything else (hard extents, hard current projection, implicit Butler-Volmer root, residuals, adaptive weights, Huber
on the potential residuals) is the single-rate code: the parametric model subclasses `LiSPANPINN` and reuses its
kinetics.  The cut-off times are the only FV information the network uses; the evaluation compares with FV
references at seven rates (`eval_rates`), three of which (0.15, 0.3, 0.7 C) are not window anchors.

The trained model answers `model.discharge_curve(rate)` (time, voltage, capacity in mAh/g_S) and refuses rates
outside the trained range (no extrapolation).  `load_rate_model(final.pt)` rebuilds it from a checkpoint.

## Files

- `src/dfn_pinn/lispan/rate_pinn.py`: model, residuals, sampler, evaluation, training (`train_rate`), I/O.
- `scripts/lispan_rate_train.py`: CLI; `--run-dir` folders resume automatically from `latest.pt`.
- `scripts/lispan_rate_summary.py`: error table + gate checks for finished or running waves (standard library only).
- `configs/lispan_rate_0.1-1C.json`: Phase 3a recipe (0.1-1 C).
- `configs/lispan_rate_variants.txt`: one line per SLURM array task (`<name> key=value ...`).
- `slurm/lispan_rate_array.slrm`: one single-core job per variant; resubmitting resumes unfinished runs.
- `tests/test_lispan_rate_pinn.py`: window fit, hard structure (extents, current projection, charge passed),
  residual shapes, save/load/resume.
- `src/dfn_pinn/lispan/model.py`: `export` now writes the FV reference atomically (parallel jobs create the same
  cached file).

## Cost (measured)

Grace, existing single-rate PINN, 3000 steps (2026-10-17): 279 / 212 / 153 ms per step on 1 / 4 / 8 cores, so
single-core jobs give the most training per SU.  On the cloud sandbox the parametric model takes 0.52 s per step vs
0.30 s for the single-rate model (1.7x for twice the points), which projects to ~0.48 s per step on one Grace core:
~8 h for 60 000 steps (~8-10 SU per variant with evaluations; the 2x-points and the 120 000-step variants ~15-17 h).

Why many steps: the electrolyte transient through the 618 um separator (diffusion time ~ L_sep^2/D ~ 1500 s) is
comparable to the 1 C window, so the salt field is the slow part of training.  The single-rate 1 C run needed ~12 500
steps to bring the salt residuals from ~0.8 to 0.04 (V 2.3 mV) and 30 000 for 0.23 mV; the parametric network learns
that transient for every rate at once.  Local check (sandbox, 3000-step schedule): salt residuals ~0.5-0.8 for the
first thousand steps, exactly like the 1 C single-rate run, while the kinetics residuals are already at 1e-3.

## How to run (Grace)

```bash
cd /scratch/user/$USER/dfn-pinn-project
git pull
mkdir -p slurm_logs
sbatch slurm/lispan_rate_array.slrm            # wave 2 (default since 2026-10-20): 4 variants, 1 core each, ~80-90 SU
squeue -u $USER
tail -n 3 results/lispan_rate_runs/*/train.log # progress; evaluations every 5000 steps
```

After the wave: `python scripts/lispan_rate_summary.py` prints the error table (all evaluation rates, skipped batches,
rollbacks with their steps) and the three gate checks for `results/lispan_rate_runs/rate_v2_*`; pass other run folders
as arguments (e.g. `results/lispan_rate_runs/rate_w96_*`) and `--json <file>` to keep it.

GPU (if the GPU check shows a large speed-up): add a variants line with `device=cuda` and submit that index with
`sbatch --partition=gpu --gres=gpu:a100:1 --cpus-per-task=1 --array=<i> slurm/lispan_rate_array.slrm`.

## Acceptance (gate before Phase 3b)

1. Voltage vs FV within 2 mV rms at all seven evaluation rates (single-rate runs reach 0.14-0.44 mV).
2. No systematic error growth at the non-anchor rates (0.15, 0.3, 0.7 C) compared with the anchors.
3. Seed spread (`rate_w96_s0` vs `rate_w96_s1`) below the error itself.

Wave 1 variants (`configs/lispan_rate_variants.txt`): `rate_w96_s0`, `rate_w96_s1` (seed spread), `rate_w128_s0`
(network capacity), `rate_w96_2xpts_s0` (collocation points), `rate_w96_120k_s0` (training length).  If criterion 1
fails, they show which of these is the limit.

Correctness check (`tests/test_lispan_rate_pinn.py::test_matches_single_rate_model_at_fixed_rate`): with the
rate-feature weights zeroed and the other weights copied from a single-rate model built for the same current and
window, every residual and the voltage agree to round-off at 0.1, 0.37 and 1 C.

## Wave 1 results (Grace, array 20184218, finished 2026-10-19/20)

Voltage rms error vs the FV solver [mV] at the last evaluation (window up to ~98 % of the cut-off; t > 100 s):

| Run | Steps | Wall h | 0.1 C | 0.15 | 0.2 | 0.3 | 0.5 | 0.7 | 1 C | mean | worst | V max worst | skipped batches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rate_w96_120k_s0 | 120 000 | 18.4 | 0.33 | 0.11 | 0.08 | 0.08 | 0.07 | 0.09 | 0.33 | **0.15** | **0.33** | 1.7 | 219 |
| rate_w96_s1 | 60 000 | 10.7 | 0.72 | 0.29 | 0.20 | 0.19 | 0.14 | 0.19 | 0.38 | 0.30 | 0.72 | 4.1 | 57 |
| rate_w96_2xpts_s0 | 60 000 | 17.5 | 0.51 | 0.33 | 0.30 | 0.25 | 0.27 | 0.33 | 0.65 | 0.38 | 0.65 | 2.8 | 97 |
| rate_w96_s0 | 60 000 | 11.6 | 0.96 | 0.70 | 0.46 | 0.41 | 0.45 | 0.45 | 0.66 | 0.58 | 0.96 | 3.4 | 51 |
| rate_w128_s0 | 60 000 | 14.2 | 114.6 | 109.7 | 102.1 | 93.1 | 77.0 | 62.5 | 43.1 | 86.0 | 114.6 | 720.7 | 31 |

c_e max error at 1 C: 2.9 (120k), 4.0-4.5 (60k runs), 6.9 mol/m3 (w128).  MaxRSS 0.94-1.65 GB.  Grace time per step
on one core: 0.55-0.70 s (w96), 0.85 s (w128), 1.05 s (2x points).  Cost of the wave: ~72 SU.

Reading:

1. **Gate criterion 1 met.**  One network covers 0.1-1 C with 0.07-0.33 mV rms (120k steps), the accuracy of the
   single-rate networks (0.14-0.44 mV, one 30k-step training per rate).  Four of five runs stay below 1 mV at every rate.
2. **Criterion 2 met.**  The non-anchor rates (0.15, 0.3, 0.7 C) are as good as the anchors; the largest errors sit at
   the ends of the range (0.1 and 1 C), an edge effect of the sampling (fewer neighbouring rates).  Remedy: train on a
   slightly wider range than the one used (e.g. 0.08-1.25 C) or oversample the ends.
3. **Training is not yet robust (criterion 3 open).**  Three of five runs had an instability: rate_w96_s0 jumped to
   ~300 mV at 15 000 steps and recovered only at 35 000; rate_w96_2xpts_s0 jumped to 167 mV at 15 000 and recovered by
   25 000; rate_w128_s0 jumped after 20 000 (charge inventory switched on at 20 000) and froze from 30 000 on (identical
   evaluations to the last digit: tanh units saturated, zero gradients).  Every run also skipped 31-219 batches with a
   non-finite loss (huge Butler-Volmer rates where the Delta-phi root sits at the bracket edge; t_end(r) a f/(2 c) can
   reach 1e29 before squaring in float32).  The seed spread (s0 0.58 vs s1 0.30 mV) is dominated by s0's lost steps.
4. **Training length matters more than width or points**: the 120k run (lr decay twice as slow, inventory from 40k)
   had no excursion and the lowest error.

Diagnosis of the excursions (history.json of the five runs, group weights reconstructed from the weighted loss by
least squares over each 50-step adaptive-weight window; figure report/phase3/figures/p3a_training.png):

- They are not single bad batches but a **runaway of the adaptive loss weights** (gradient-norm balancing,
  `AdaptiveWeights`): before each excursion the ODE and boundary groups sit at weights ~100-1000 (the cap is 1e3);
  a small perturbation (salt inventory bump at ~9 900-10 000 steps in rate_w96_s0 and rate_w96_2xpts_s0) raises the
  kinetics residuals, their gradients grow, the rule LOWERS the weight of exactly the group that is getting worse
  (charge: 54 -> 10, charge_s 1.5e-3 -> 66), so nothing pulls it back: positive feedback over ~500 steps.
- In rate_w128_s0 the trigger is the charge-inventory term switched on at 20 000: its weight fell to ~0.2 while its
  residual grew from ~1 to ~1e4, after which the kinetics followed; the network then froze (constant losses from
  ~29 000 on; 19 % of the reaction-2 rate outputs beyond their clamp).
- After the jump, Adam's second moments are inflated for ~1/(1-beta2) ~ 1000s of steps, so the network sits on a
  plateau (flat losses in the figure) until they decay - hence the 5 000-20 000-step losses.

Fix for wave 2 (to be validated by replaying the excursions from the Grace checkpoints step_10000.pt / step_20000.pt):
a group whose loss is rising above its running best may not lose weight (and is pushed up instead), and a rollback
guard restores the last healthy in-memory snapshot if the kinetics residuals exceed ~20x their running median.

### Stabilization and its test (2026-10-20)

Implemented in `rate_pinn.py` (both off by default so that wave-1 runs resume unchanged; on in
`configs/lispan_rate_wave2.json`):

- `GuardedWeights` (`weight_guard`): per-step EMA of every group's unweighted loss; a group whose EMA exceeds 3x its
  best EMA may not lose weight at the adaptive update and is multiplied by 1.25 instead (cap 1e3).
- `rollback`: EMA of the kinetics losses (ode1-3, charge_s, charge_e) against its running median (last 2000 steps,
  seeded from history.json on resume); above 100x the median (and 1e-2) the last healthy in-memory snapshot (every 500
  steps) is restored and the learning rate halved for 2000 steps.  Replaying the wave-1 histories offline, a factor of
  20 would have fired 28-40 times in the two runs that recovered by themselves, 100 only 5 times; every lasting
  excursion fires at either factor.
- `stop_step` (stop and save latest.pt at a given step) and `scripts/lispan_rate_replay.py` (resume a checkpoint into a
  new folder with changed settings).

Test: replays from the Grace checkpoints (same network, Adam state, adaptive weights and random generator; same
batches until the settings differ) on the sandbox, with and without the stabilization:

| Replay | ode1 | charge_s | charge_total | mean V rmse over 7 rates | worst |
|---|---|---|---|---|---|
| rate_w96_2xpts_s0, start (step 10 000) | 5e-3 | 4e-3 | - | 16.4 mV | 26.0 mV |
| ... step 11 000, wave-1 settings | 8.0 | 36 | - | 138.6 mV | 202.8 mV |
| ... step 11 000, weight guard + rollback | 4.6e-4 | 9.4e-4 | - | 4.2 mV | 8.9 mV |
| rate_w128_s0, start (step 20 000) | 8e-4 | 2e-4 | 10 | 3.3 mV | 7.5 mV |
| ... step 23 000, wave-1 settings | 0.23 (0.59 at 22 800) | 0.08 (0.87) | 2.5e3 (4.3e3) | 47.8 mV | 55.5 mV |
| ... step 23 000, weight guard + rollback | 6.3e-4 | 7.4e-5 | 0.76 | 1.28 mV | 2.35 mV |

Figure: report/phase3/figures/p3a_replay.png (report/phase3/make_replay_figure.py).  Both wave-1 excursions are
reproduced without the fix (2xpts within 300 steps; w128 at ~22 750, the same place as on Grace, with the charge
inventory running away first) and absent with it.  In the w128 replay the guard kept the inventory weight at 0.5-1.1
(0.22 with the plain rule) and one rollback fired at 22 311.  These two replays used rollback factor 20 and a permanent
halving of the learning rate.  Final settings (factor 100; learning rate halved for 2000 steps, halved again for every
further rollback inside that window), same replays:

| Replay (final settings) | rollbacks | mean V rmse | worst | vs wave-1 settings |
|---|---|---|---|---|
| rate_w128_s0, 20 000 -> 23 000 | 1 (22 321) | 1.03 mV | 1.70 mV | 47.8 / 55.5 mV |
| rate_w96_2xpts_s0, 10 000 -> 11 500 | 2 (10 217, 10 482) | 6.25 mV | 10.1 mV | 138.6 / 202.8 mV (at 11 000) |

The 2xpts state at step 10 000 is itself already at the start of the excursion, so it keeps producing bumps (another at
~11 300, which recovers to ~7e-3 without a rollback); with the stabilization they stay at 1e-2 instead of 10-40 and the
run keeps improving (16.4 -> 6.3 mV) where the wave-1 settings lose it (-> 139 mV).  Without the compounding a second
rollback inside the window (2xpts_final2) left the run at 22 mV at step 11 000.  The weight guard alone (rollback
blind, no baseline: a replay without history.json) did NOT stop the 2xpts runaway; the rollback is the essential part,
the guard keeps groups such as the charge inventory from being switched off (w128: inventory weight 0.5-1.1 vs 0.22).

In the 2xpts replay one rollback fires at step 10 190 (kinetics EMA 20x its median) and a second, smaller bump at
~10 400 recovers on its own.  The
rate_w96_s0 replay did not reproduce its excursion on the sandbox within 1600 steps (chaotic divergence from the Grace
trajectory), and is not counted either way.

Wave 2 (`configs/lispan_rate_wave2.json`, `configs/lispan_rate_variants_wave2.txt`, array 0-3, 36 h limit): the
120k-step recipe with the stabilization, seeds 0/1/2 (criterion 3), and one run on the wider training range 0.08-1.25 C
(window anchors 0.08/0.2/0.5/1.25 C: windows 96.8-98.9 % of the cut-off) evaluated also at 0.08 and 1.25 C.
Then Phase 3b (chemistry parameters as inputs).
