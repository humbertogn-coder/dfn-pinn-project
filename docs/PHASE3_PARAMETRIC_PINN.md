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
sbatch slurm/lispan_rate_array.slrm            # 5 variants, 1 core each (~60-70 SU in total)
squeue -u $USER
tail -n 3 results/lispan_rate_runs/*/train.log # progress; evaluations every 5000 steps
```

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
