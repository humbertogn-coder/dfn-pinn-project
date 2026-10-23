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
- `configs/lispan_rate_wave2.json` / `_variants_wave2.txt` (weight guard + rollback), `configs/lispan_rate_wave3.json`
  / `_variants_wave3.txt` (rollback with the best-recent-level baseline, no guard, best.pt) and
  `configs/lispan_rate_wave4.json` / `_variants_wave4.txt` (+ weights frozen at 50 000, rollbacks from 30 000 with a
  budget per 20 000 steps).
- `scripts/lispan_rate_replay.py`: resume a checkpoint into a new folder with changed settings, up to a given step.
- `report/phase3/make_figures.py`, `make_replay_figure.py`, `make_wave2_figures.py`, `make_wave3_figure.py`: figures in
  report/phase3/figures.
- `slurm/lispan_rate_array.slrm`: one single-core job per variant; resubmitting resumes unfinished runs.
- `tests/test_lispan_rate_pinn.py`: window fit, hard structure (extents, current projection, charge passed),
  residual shapes, save/load/resume, equality with the single-rate model, weight guard, rollback, rollback monitor on
  slow ramps, best.pt selection.
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
sbatch slurm/lispan_rate_array.slrm            # wave 4 (default since 2026-10-22): 4 variants, 1 core each, ~72 SU
squeue -u $USER
tail -n 3 results/lispan_rate_runs/*/train.log # progress; evaluations every 5000 steps
```

After the wave: `python scripts/lispan_rate_summary.py` prints the error table (all evaluation rates, skipped batches,
rollbacks with their steps) and the three gate checks for the latest wave (`results/lispan_rate_runs/rate_vN_*`); pass other run folders
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

(Wave 2 showed that the weight guard does harm; it is off from wave 3 on, see "Wave 2 results".)

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
Then Phase 3b (chemistry parameters as inputs).  Results and the change for wave 3: next section.

## Wave 2 results (Grace, array 20213153, finished 2026-10-21)

`python scripts/lispan_rate_summary.py` (voltage rms error vs FV [mV], final model, window up to ~98 % of the cut-off,
t > 100 s):

| Run | 0.1 C | 0.15 | 0.2 | 0.3 | 0.5 | 0.7 | 1 C | mean | worst | V max worst | skipped | rollbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rate_v2_s1 | 0.29 | 0.24 | 0.21 | 0.20 | 0.16 | 0.21 | 0.36 | **0.24** | **0.36** | 2.0 | 82 | 10 |
| rate_v2_s2 | 0.59 | 0.40 | 0.38 | 0.26 | 0.35 | 0.49 | 0.62 | **0.44** | **0.62** | 2.3 | 1580 | 10 |
| rate_v2_s0 | 306 | 302 | 294 | 277 | 245 | 220 | 194 | 262 | 306 | 815 | 96 | 3 |
| rate_v2_wide_s0 (0.08-1.25 C; also 1065 at 0.08 C, 937 at 1.25 C) | 1066 | 1072 | 1076 | 1075 | 1068 | 1034 | 967 | 1040 | 1076 | 1546 | 497 | 6 |

18.7-18.8 h per run, MaxRSS 0.99-1.13 GB, ~75 SU (Grace balance after the wave: 1 621.66 SU on the group account).
Gate: s1 and s2 pass criteria 1 and 2 (s1 is the second-best network so far); s0 and the wide run failed, so criterion 3
fails (one of three seeds collapsed).  Figure: report/phase3/figures/p3a_wave2.png (report/phase3/make_wave2_figures.py).

Diagnosis (history.json of the four runs and the checkpoints every 20 000 steps):

1. **Paired comparison.**  rate_v2_s0 is rate_w96_120k_s0 of wave 1 plus the weight guard and the rollback, nothing else
   (config.json diff).  Wave 1: 0.15 mV.  Wave 2: collapse between 5 000 and 10 000 steps (176 mV), frozen afterwards
   (identical evaluations from 15 000 to 45 000 steps), 262 mV at the end.
2. **The weight guard inflates the weights.**  "Rising" means the loss EMA is above 3x the best EMA ever reached; the
   ordinary fluctuations of the residuals exceed that, so a group gets x1.25 at every 50-step update and reaches the cap
   (1000) within ~1 500 steps.  By 3 000-5 000 steps the ode, charge, boundary and salt groups were all at 1000 in s0,
   s2 and the wide run (wave-1 checkpoints at 10 000 steps: ode 250-300, charge 28-112, salt 6-15, boundary 320-370).
   In both failed runs the salt weight stayed at the cap to the end (relative to the ode group 20-50x heavier than in
   wave 1), and from 40 000-60 000 steps the charge-inventory weight too (wave 1: 0.21).  In the two runs that ended
   well the salt weight came back to 6-30, the wave-1 range (panel c).
3. **The rollback monitor follows slow ramps.**  In s0 the kinetics losses rose x5-10 per 1000 steps from 5 000 steps
   (1.9e-2) to 10 000 (130).  The baseline (running median of the last 2000 steps) rose with them, so the EMA stayed
   ~10x the baseline and never reached the 100x trigger; and "healthy" snapshots, judged against the same baseline,
   were already degraded (the rollback at 6 819 restored step 6 000, whose kinetics loss was 6x the level at 5 000).
   s2: the same ramp from 1 000 to 4 000 steps, collapse at 6 000-8 000 (~1 100 mV), rescued only when the charge
   inventory came in at 40 000 (1.15 mV at 70 000, 0.44 mV final).  s1 and s2 used up the 10 allowed rollbacks (s1:
   seven between 29 097 and 32 481, restoring 29 000-31 500 with the learning rate down to 1/64; it recovered).
4. **Wide run:** a kinetics ramp from 2e-2 to 65 between 30 000 and 34 000 steps (rollbacks at 29 126 and 30 818 against
   a baseline already raised), no recovery (inventory weight at the cap after 60 000).  Whether the wider range is
   harder by itself cannot be told from this run.

### Change for wave 3 and its test (2026-10-21)

- `weight_guard` off: back to the plain gradient-norm rule of wave 1 (the guard was meant to stop a group from being
  switched off while it grows; the replays below show the rollback alone contains both wave-1 runaways).
- `rollback_horizon = 5000` (`HealthMonitor` in `rate_pinn.py`): the baseline is the LOWEST 500-step running median of
  the last 5 000 steps, i.e. the best recent level, so a ramp of any speed fires at ~100x that level and only snapshots
  close to it count as healthy.  A lasting level shift below 100x does not fire and is absorbed after 5 000 steps.
  Tests (`tests/test_lispan_rate_pinn.py`): ramps of x10 and x3 per 1000 steps fire at ~100x with the new baseline and
  never with the wave-2 one; a lasting x5 shift does not fire.
- `max_rollbacks = 20`.
- `keep_best`: the evaluation with the lowest mean error over the ANCHOR rates (0.1/0.2/0.5/1 C) is kept as best.pt and
  best_metrics.json; the non-anchor rates stay unseen by that choice.  The gate is still checked on final.pt.

Replays on the sandbox from the same wave-1 checkpoints (identical to the earlier replays step by step until the first
rollback):

| Replay | settings | rollbacks | mean V rmse | worst |
|---|---|---|---|---|
| rate_w96_2xpts_s0, 10 000 -> 11 500 | wave 1 (no stabilization), at 11 000 | - | 138.6 mV | 202.8 mV |
| | wave 2 (guard + rollback) | 2 | 6.25 mV | 10.1 mV |
| | wave 3 (rollback with the horizon baseline, no guard) | 1 (10 207 -> 10 000) | 5.91 mV | 13.7 mV |
| rate_w128_s0, 20 000 -> 23 000 | wave 1, at 23 000 | - | 47.8 mV | 55.5 mV |
| | wave 2 | 1 | 1.03 mV | 1.70 mV |
| | wave 3 | 1 (22 781 -> 22 500) | 1.49 mV | 3.19 mV |

Wave 3 (`configs/lispan_rate_wave3.json`, `configs/lispan_rate_variants_wave3.txt`, default of
`slurm/lispan_rate_array.slrm`): the same four lines as wave 2 (seeds 0/1/2 on 0.1-1 C, seed 0 on 0.08-1.25 C), so each
run is a paired comparison with its wave-2 twin; ~75 SU.  Decision after wave 3: three of three seeds below 1 mV ->
criterion 3 met, Phase 3b.  If a seed still collapses, the documented procedure becomes multi-start (two or three seeds,
keep the best by the anchor-rate evaluation) and the success rate is reported with it.

## Wave 3 results (Grace, array 20231451, finished 2026-10-22)

| Run | 0.08 C | 0.1 C | 0.15 | 0.2 | 0.3 | 0.5 | 0.7 | 1 C | 1.25 C | mean | worst | best.pt (step): mean / worst | rollbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rate_v3_s1 | - | 0.42 | 0.17 | 0.13 | 0.15 | 0.13 | 0.14 | 0.31 | - | **0.21** | **0.42** | 0.21 / 0.39 (115 000) | 18 |
| rate_v3_wide_s0 | 0.96 | 0.59 | 0.41 | 0.44 | 0.52 | 0.60 | 0.67 | 0.64 | 1.17 | **0.67** | **1.17** | 0.49 / 1.12 (65 000) | 7 |
| rate_v3_s2 | - | 4.72 | 4.43 | 4.54 | 5.32 | 7.05 | 8.60 | 10.11 | - | 6.40 | 10.11 | 0.64 / 1.62 (60 000) | 20 |
| rate_v3_s0 | - | 33.88 | 27.44 | 20.52 | 12.65 | 11.08 | 14.11 | 15.88 | - | 19.37 | 33.88 | 4.35 / 5.47 (75 000) | 20 |

17.2-18.2 h per run, MaxRSS 0.96-1.07 GB, ~72 SU (Grace group balance 1 549.46 SU).  Figure:
report/phase3/figures/p3a_wave3.png (report/phase3/make_wave3_figure.py).

Gate: s1 and the wide run pass criteria 1-2 on final.pt (the wide run is the first network valid over 0.08-1.25 C);
s2 passes on best.pt only; s0 fails; criterion 3 fails again.  Against the wave-2 twins: wide 1040 -> 0.67 mV,
s0 262 -> 19 mV, s1 0.24 -> 0.21 mV, s2 0.44 -> 6.4 mV (best.pt 0.64).

Count over the runs without the weight guard (wave 1 w96 + wave 3): 6 of 8 below 2 mV on final.pt, 7 of 8 on best.pt.

Diagnosis (figure panels b-c; history.json):

1. **Every excursion starts in the solid-charge residual (charge_s) and the charge-group weight falls with it**, from
   100-800 to ~0.2 in a few thousand steps, while the ode / salt / boundary / inventory weights go to the cap: the
   positive feedback of the gradient-norm rule found in wave 1, now without the guard that was meant to stop it.
   The late, fatal ones are set off by the charge inventory (charge_total, on from 40 000): s2 at ~64 500
   (charge_total 0.9 -> 1 800 within 4 000 steps, then the kinetics), s0 at ~91 500, the wide run at 84 000-110 000
   (recovered).  Before them the inventory residual was already poor in the runs that failed (median over
   60 000-64 000 steps: s1 0.07, s2 0.28, wide 0.51, s0 3.4).
2. **The rollback budget was spent early.**  s0, s1 and s2 used 18-20 of the 20 rollbacks before ~26 000 steps,
   most of them in loops back to the same snapshot (s1: eleven restorations of step 20 000; learning rate down to
   1/128), so nothing protected the late phase.  The early kinetics bursts (x10-100, ~every 1 000-3 000 steps until
   ~30 000) recover by themselves in every run with or without rollbacks (wave 1 had them too).
3. **s0 froze** after ~97 000 steps (identical evaluations from 105 000: the saturated state seen in wave 1 w128 and
   wave 2 s0).

Replays on the sandbox (2026-10-22):

| Replay | settings | outcome |
|---|---|---|
| rate_v3_s1 20 000 -> 23 000 | wave 3 as on Grace (control) | the same loop: 6 rollbacks to 20 000 (Grace: 7), 4.8 -> 9.2 mV |
| | + hold guard (weight_guard_ref "recent", boost 1.0) | same loop (stopped at 21 700) |
| | + charge-weight floor 0.3 x or 1.0 x the ode weight | same loop (stopped at 21 200) |
| rate_v3_s2 60 000 -> 66 000 (Grace: inventory runaway at ~64 500) | wave 3 as on Grace (no rollbacks left) | no runaway this time (chaotic: not reproduced), 0.64 -> 0.53 mV |
| | adaptive weights frozen at their 60 000 values | no runaway, 0.64 -> 0.59 mV |

Reading: the early loops are not caused by the charge weight (holding or flooring it does not change them); they come
from rolling back bursts that the run would survive anyway.  The late failures are random events (the s2 runaway does
not recur from the same state on another machine), each one a weight runaway set off by the charge inventory after the
rollbacks were used up.  Freezing the weights removes that mechanism by construction and cost nothing in the replay.

### Change for wave 4 (2026-10-22)

`configs/lispan_rate_wave4.json` = wave 3 plus:

- `adaptive_stop = 50000`: the adaptive weights are frozen at step 50 000 at the median of their values over the
  previous 5 000 steps (inventory on since 40 000, so its weight has settled); no weight runaway is possible after it.
  A rollback after the freeze keeps the frozen weights.
- `rollback_start = 30000` and `rollback_budget_window = 20000` (10 rollbacks per 20 000 steps): no rollbacks of the
  early bursts, protection for the whole late phase.
- Unchanged: rollback with the best-recent-level baseline, best.pt, no weight guard.  The hold guard
  (`weight_guard_ref = "recent"`, `weight_guard_boost = 1.0`) and `charge_weight_floor` stay available but off (no
  effect in the replays).

`configs/lispan_rate_variants_wave4.txt`: the same four lines again (rate_v4_s0/s1/s2/wide_s0), paired with waves 2-3;
~72 SU.  Decision: 3 of 3 seeds below 1 mV on final.pt -> criterion 3 met.  Either way the training procedure for
Phase 3b is multi-start (2-3 seeds, keep the best by the anchor-rate evaluation): with ~75 % of runs succeeding,
three seeds give a usable network with ~98 % probability.

### Phase 3a networks to use

- 0.1-1 C: `results/lispan_rate_runs/rate_w96_120k_s0/final.pt` (wave 1; 0.07-0.33 mV, mean 0.15 mV), runner-up
  `rate_v3_s1/final.pt` (0.13-0.42 mV, mean 0.21 mV).
- 0.08-1.25 C: `rate_v3_wide_s0/best.pt` (step 65 000; mean 0.49 mV, worst 1.12 mV at 1.25 C; chosen on the anchor rates
  only) or its final.pt (0.67 / 1.17 mV).
