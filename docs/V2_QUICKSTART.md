# DFN PINN v2: quick start (Windows, Anaconda Prompt)

All commands from the repository root (`DFN_PINN_Project`), environment
`dfn-pinn` (Python 3.11, torch, PyBaMM). v2 only adds files; v1 is unchanged.

```bat
conda activate dfn-pinn
cd /d "%USERPROFILE%\OneDrive\Documents\Claude\PINN-DFN-Project\DFN_PINN_Project"
python -m pip install -e .
```

## 1. Unit tests (seconds)

```bat
python -m pytest -q tests/test_v2_model.py
```

## 2. PyBaMM reference (about 1 min)

```bat
python scripts/v2_make_reference.py --nx 80 --nr 120
python scripts/v2_make_reference.py --nx 40 --nr 60
python scripts/v2_check_equations_fd.py results/v2_reference/ref_I5A_t3000s_ramp30s_x80_r120.npz
```

The first command writes the evaluation reference used by the configs. The
second is a coarser mesh for the mesh-uncertainty check. The third verifies
the v2 sign/unit conventions against PyBaMM (finite differences, no network).

## 3. Forward training

```bat
python scripts/v2_train.py --config configs/v2_forward_1C.json --name forward
```

The config already includes `fourier_t = 4` (Fourier time features), which
is what brings the voltage error below 1 mV rmse in 20 000 steps; without it
(`--set fourier_t=0`) the same training stalls at ~2.6 mV with a systematic
bias (docs/V2_RESULTS.md, 2d-2e).

The config also sets `collector_bc = "hard"` (electrolyte x-features with
zero slope at both collectors, so zero salt flux and zero electrolyte
current there are exact; V2_RESULTS.md section 7: c_e error halved with
respect to the soft boundary terms). `--set collector_bc=soft` restores
the 2f behaviour. `fourier_period=2` (half-period time features) is
implemented but gave no gain.

Overrides: `--set adam_steps=40000 dtype=float64 width=96`. Output:
`results/v2_runs/forward_<UTC time>/` with `train.log`, `history.json`,
checkpoints every 2500 steps, `final.pt`, `final_metrics.json`. Diagnostic
figure:

```bat
python scripts/v2_plot_run.py results/v2_runs/<run folder> results/v2_reference/ref_I5A_t3000s_ramp30s_x80_r120.npz
```

Measured cost in the Claude cloud sandbox (2 CPU cores, float32): about
0.27 s per Adam step, i.e. 20 000 steps in about 90 min. Run several
experiments in parallel only if the machine has spare cores
(`--set threads=2`).

## 3b. Optional second-order refinement (mini-batch L-BFGS, float64)

```bat
python scripts/v2_train.py --init results/v2_runs/<forward run>/final.pt --name forward_lbfgs --set adam_steps=0 lbfgs_iters=800 lbfgs_resample_every=25 lbfgs_batch_mult=4 lbfgs_eval_every=200 dtype=float64
```

The collocation batch is resampled every 25 iterations; a fixed batch
overfits (V2_RESULTS.md 2b). About 1 h per 800 iterations on one core and
1-2 GB of memory with `lbfgs_batch_mult=4`. Gains are modest (10-20 %).

## 3c. Warm starts

`--init <checkpoint>` restores the networks, the adaptive group weights and
(unless `--fresh-optimizer`) the Adam moments, with the learning-rate
schedule restarted at `lr`. For a continuation use a small `lr` (1e-4 or
less): a fresh Adam state at 2e-3 destroys a converged model in two steps.

Interrupted runs: every run writes a rolling `latest.pt` every
`latest_every` steps (default 250, model + Adam state + step + group
weights). `python scripts/v2_train.py --resume results/v2_runs/<run folder>`
continues it in place with the stored config (step counter, learning-rate
schedule and history continue; pass the matching `--config` only so the
reference path is known). The same `--resume` exists in
`scripts/v2_inverse_multirate.py`.

## 3d. Seed study

```bat
python scripts/v2_train.py --name seed1 --set seed=1
python scripts/v2_train.py --name seed2 --set seed=2
python scripts/v2_summarize_runs.py results/v2_runs/seed1_* results/v2_runs/seed2_* results/v2_runs/<seed 0 run>
```

## 3e. Regression test of an architecture (no physics)

```bat
python scripts/v2_check_residuals.py --ref results/v2_reference/ref_I5A_t3000s_ramp30s_x80_r120.npz --steps 3000 --skip-residuals --no-theta --points 1024 --fourier 4
```

Fits the networks directly to the PyBaMM fields; tells whether an
architecture can represent the solution at all before spending PINN time
on it (this is how `fourier_t` was selected).

## 4. Synthetic inverse problem

```bat
python -c "import numpy as np,sys; sys.path.insert(0,'src'); from dfn_pinn.v2.reference import load; r=load('results/v2_reference/ref_I5A_t3000s_ramp30s_x80_r120.npz'); np.savez('results/v2_reference/data_V_I5A_t3000s.npz', t=r['t'], V=r['V'], I=r['I'])"
python scripts/v2_train.py --config configs/v2_inverse_synthetic.json --init results/v2_runs/<forward run>/final.pt --name inverse
```

The parameters to identify (`inverse_params`) are learned as log-multipliers
of the Chen2020 values (true value = 1). Noise in mV is set by
`data_noise_mV`.

Before (or instead of) an inverse run, predict its bias from the forward
model and the PyBaMM sensitivities (V2_RESULTS.md 3.3):

```bat
python scripts/v2_identifiability.py
python scripts/v2_bias_prediction.py results/v2_runs/<forward run> --params D_p k_n D_n
```

The first command writes `results/v2_identifiability/sensitivities_1C.npz`
(and 0.5C, 2C); the second projects the forward voltage error on the
sensitivities and prints the parameter bias the inverse problem will
inherit (e.g. D_n -3.7 % for the 2f model, observed -4.1 %). `--t-min` /
`--t-max` restrict the data window, `--checkpoint` picks another
checkpoint of the run.

## 4b. Multi-C-rate inverse problem (shared parameters)

```bat
python scripts/v2_make_reference.py --current 2.5 --t-end 6500 --nx 80 --nr 120
python scripts/v2_make_reference.py --current 10 --t-end 1400 --nx 80 --nr 120
python scripts/v2_inverse_multirate.py --config configs/v2_inverse_multirate.json --name multirate
```

One DFNPINN per protocol (own networks, scales and adaptive weights), the
physical log-multipliers shared. The data files (t, V) are created from the
references if missing. `param_warmup_steps` keeps the parameters frozen
while the fields form (without it the electrolyte parameters ran to 20x in
the first 500 steps), `param_log_bound` bounds the multipliers (default
factor 0.1-10). Cost about 1 s per step for three protocols on two cores.
`scripts/v2_identifiability.py` (PyBaMM, no PINN) tells which parameters
are worth including for a given set of C-rates and noise level.

Hierarchical estimation (release parameters in stages):

```bat
:: stage 1: solid-phase parameters, electrolyte fixed (configs/v2_inverse_multirate.json)
python scripts/v2_inverse_multirate.py --name stage1
:: stage 2: warm start from stage 1 (networks + estimates kept), release D_e and kappa_e
python scripts/v2_inverse_multirate.py --name stage2 --init results/v2_runs/stage1_<ts>/final.pt --set "inverse_params=[\"D_p\",\"k_n\",\"D_n\",\"D_e\",\"kappa_e\"]" "inverse_init={}" lr=2e-4 adam_steps=5000
```

Within one run, `param_release_steps={"D_e": 5000}` keeps a parameter frozen
until the given step (per parameter; `param_warmup_steps` is the common
minimum). `--resume <run folder>` continues an interrupted run from its
rolling `latest.pt` (written every `checkpoint_every` steps, optimizer state
included); `--init` is a warm start with a fresh optimizer, so use a small
`lr` as in 3c.

## 5. Variants for the C-F factorial and other options

```bat
python scripts/v2_train.py --name C_direct_soft --set kinetics=direct projection=false
python scripts/v2_train.py --name D_inverse_soft --set kinetics=inverse projection=false
python scripts/v2_train.py --name E_inverse_hard --set kinetics=inverse projection=true
python scripts/v2_train.py --name F_direct_hard --set kinetics=direct projection=true
```

With `projection=false` a soft electrode-integrated current term
(`soft_current`, group "current") replaces the hard projection. Outcome
(docs/V2_RESULTS.md section 5): the two direct-kinetics variants C and F
do not converge (voltage error 460 mV after 10 000 steps, the v1 failure
mode); D converges to 0.70 mV rmse and E (the default) to 0.57 mV, E being
about 2x ahead of D at every step. Other implemented options (tested, not
default): `kinetics=hard` (phi_e defined by inverse BV in the electrodes),
`inventory=derived` (j derived from the learned particle mean),
`causal=true` (causal time weights), `x_edge_fraction=0.3` (denser sampling
at the separator side), `adaptive=false fixed_group_weights={...}`.
