# First Single-Particle PINN

Run `python scripts/train_single_particle.py` in the project environment.
The default CPU float64 run uses seed 42, one thread, 1000 Adam steps followed
by at most 150 L-BFGS iterations. Output is a new timestamped results directory.
`--seed`, `--adam-steps`, `--lbfgs-steps`, and `--output` are explicit overrides.
The destination must not already exist. No reference HDF5 files are modified.

This is a synthetic dimensionless closed sphere with D*t_ref/R^2=1, tau in
[0,0.1], and exact concentration
`0.5 + 0.1*sinc(alpha*rho/pi)*exp(-alpha^2*tau)`, where tan(alpha)=alpha
and alpha is the first positive nonzero root. It is not a Chen2020 discharge.
The mean concentration is exactly 0.5 and both radial boundaries have zero flux.

A 2-input, two-hidden-layer (24 units each) tanh MLP predicts the concentration
perturbation. Input rho^2 imposes center symmetry; the surface condition and
initial profile are soft losses. There is no hard-coded exact time evolution.
Training uses fixed random interior collocation points, initial data and zero
surface flux. No analytic interior labels or conservation penalty are used.
Loss terms are normalized by the 0.1 initial amplitude; weights are 1:10:1 for
PDE:initial:surface. Architecture and hyperparameters are pilot choices.

Evaluation uses a separate regular grid and 64-point radial volume quadrature.
Initial and boundary checks necessarily revisit the same physical conditions;
this is not an independent-protocol test. Reports include concentration error,
initial error, center/surface derivatives, PDE RMS, volume mean error and drift.
Concentration RMS is an unweighted grid statistic, not a volume-weighted norm.
Conservation uses the correct 3*rho^2 radial weight.

Artifacts: model.pt (state dict), report.json, loss_history.csv (closure
evaluations, not optimizer iterations), evaluation.npz and benchmark.png.
L-BFGS line search evaluations may be rejected; history is not a monotone series
of accepted iterates. The saved model is the final optimizer state, not selected
using analytic validation. Multi-seed robustness and full DFN training remain
pending. No accuracy threshold or publication claim is implied by completion.

## Recorded Pilot Run

Run `single_particle_20260916T183318096299Z`, seed 42, default settings,
PyTorch 2.14.0+cpu and Python 3.11.16:

| Diagnostic | Value |
|---|---:|
| Maximum absolute normalized concentration error | 8.728282e-5 |
| RMS normalized concentration error | 2.006243e-5 |
| Maximum initial concentration error | 8.728282e-5 |
| Maximum surface radial derivative magnitude | 6.106304e-4 |
| Maximum center radial derivative magnitude | 0 (by construction) |
| Maximum volume mean error relative to 0.5 | 1.983590e-5 |
| Maximum volume mean drift from predicted initial | 2.001338e-5 |
| Validation PDE residual RMS | 1.042824e-3 |

The timer reported 7.85 s for optimization, evaluation and initial artifact
writing; imports and figure generation are excluded. There were 157 L-BFGS
closure evaluations after 1000 Adam evaluations. The figure was visually
inspected, and all 98 project tests passed. These are sampled diagnostics for
one pilot, not continuous-domain bounds or multi-seed robustness evidence.
