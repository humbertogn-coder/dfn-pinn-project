# Constant-Flux Particle PINN Pilot

Run `python scripts/train_constant_flux_particle.py`. Default seed 42, CPU,
float64, one thread, 2000 Adam steps and 300 maximum L-BFGS iterations.
Optional flags: --seed, --adam-steps, --lbfgs-steps. This is a new pilot, not
an equal-budget comparison with the previous no-flux model.

Synthetic sphere: c_t=laplacian(c), initial c=1, center gradient=0,
surface gradient=-1 for t>0, and t in [0,0.02]. Exact mean is 1-3t.
Uniform initial data and switched flux conflict at the exact surface/time-zero
corner. Flux and interior training start at 1e-5; initial data remain at zero.
No accuracy claim covers times between zero and 1e-5.

The 2x32 tanh MLP uses rho squared for center symmetry. Other conditions are
soft losses (PDE:initial:flux weights 1:100:1). There are 256 uniform interior
points plus 256 outer-region/log-time points, 101 initial samples, and 128
surface samples combining linear and log time grids. No analytic interior
labels, exact transient features or mean-conservation penalty are used.

The independent analytic reference reuses the already-tested constant-flux
spherical series and verifies 512 versus 1024 terms on the evaluation grid.
Reported initial, early (t<=1e-3), later and surface errors are separate.
Volume means use 128-point spherical quadrature. Mean error and change error
are both reported so initial bias cannot conceal drift. Evaluation RMS is
unweighted on the chosen log-time grid, not a continuous physical-time norm.

Each new results directory contains model.pt, report.json, evaluation.npz,
loss_history.csv and benchmark.png. Loss history records closure evaluations,
not just accepted optimizer iterates. This difficult startup case can have
larger errors than the closed sphere. Completion is not an accuracy pass;
review metrics before increasing model complexity or running a seed study.

## Recorded Baseline

Run `constant_flux_pinn_20260916T192311166926Z`, seed 42, default settings:

| Diagnostic | Value |
|---|---:|
| Maximum initial concentration error | 3.029476e-2 |
| Maximum positive-time concentration error | 2.687363e-2 |
| Positive-time RMS concentration error | 3.595663e-3 |
| Maximum error for t > 1e-3 | 7.986483e-3 |
| Maximum surface flux residual | 3.170935e-2 |
| Maximum mean balance error | 1.598407e-4 |
| Maximum mean change error | 1.693549e-4 |
| Positive-time PDE residual RMS | 2.516330e-2 |
| Series truncation discrepancy | 3.330669e-15 |

The largest sampled positive-time error occurs at rho=1, t=1e-5. The figure
was visually inspected: mean balance tracks the expected evolution while the
surface response is biased at startup. All 102 software tests passed; that
does not establish adequate learned-solution accuracy. This baseline exposes
a startup limitation and should not be promoted to full-DFN training without
further investigation. Preserve it when comparing changes to startup treatment.
