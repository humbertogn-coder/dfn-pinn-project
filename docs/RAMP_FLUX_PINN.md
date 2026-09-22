# Linear-Ramp PINN

Run `python scripts/train_ramp_flux_particle.py` for the full seed-42 experiment.
Defaults: 2000 Adam steps, 300 L-BFGS maximum iterations, full_radius sampling,
512 interior/101 initial/128 surface points, float64 CPU, unchanged startup
architecture and initialization stream. No soft mass penalty is added.

The new script does not change the constant-flux trainer or projection source.
A wrapper replaces the constant mean target with 1-75*t^2 while preserving its
quadrature and autograd path. Surface loss uses c_r+50*t, not c_r+1. The hard
initial value remains one. Loss weights stay PDE:initial:surface = 1:100:1.
The existing square-root-time features are intentionally retained as a control,
not claimed optimal for the smooth ramp.

Validation uses the ramp series with 4096 modes and a 2048-mode sensitivity
check including center and surface. Mass is checked using independent 128/256
radial nodes. Both original log-time-grid and physical-volume/time PDE RMS are
reported. Concentrations are also checked for sampled extrema. Flux error is
absolute, equivalently normalized by peak imposed flux one, not by q(t) near
zero. Final extracted inventory is 0.03; the previous step's 0.06 normalization
must not be silently reused for acceptance. No automated ramp acceptance is
declared by this trainer.

Each run saves model, initial-weight hash, original training points, history,
evaluation arrays, report and plot. Checkpoint predictions are reproduced on
the full evaluation grid before completion. Paths use a separate ramp_flux_pinn
prefix so constant-flux selectors cannot accidentally load these runs.

For an integration-only smoke test use `--adam-steps 2 --lbfgs-steps 1`.
Such a run exercises training/evaluation/artifacts but provides no evidence of
converged accuracy. A full run may take roughly 20 minutes plus evaluation on
the previously measured CPU. This trainer does not resume optimizer state.

## Integration Verification

All 167 repository tests passed. Smoke run:
`results/ramp_flux_pinn_20260922T045709518537Z`, seed 42, two Adam steps and
one L-BFGS maximum iteration. Checkpoint reload succeeded; all metrics and
artifacts were produced. Maximum concentration error was 0.08021114 and flux
residual 0.9699267: these are NOT accuracy results of the full training budget.
Independent mass error was about 1.12e-13, illustrating that hard inventory
alone does not make the concentration field accurate. The full 2000/300
experiment has not been run as part of this implementation check.
