# Synthetic Particle Inventory Projection

Implemented as `UnitFluxInventoryProjection` in `dfn_pinn.particle_inventory`.
Run `python scripts/check_particle_inventory.py` before training integration.

For raw concentration u(r,x,t), define c = u - mean_Q(u) + 1 - 3t.
The radial Gauss-Legendre measure is proportional to 3*r^2*dr and normalized
to sum to one, eliminating roundoff in its constant-volume normalization.
The target applies only to this unit-radius, unit-diffusivity, constant unit
outward-flux synthetic benchmark with initial concentration one. It is not a
DFN prescription for particles with unknown, spatially varying reaction flux.

The correction is uniform in r at fixed x,t. First and second radial derivatives
are unchanged, including the surface-flux derivative and center symmetry.
The time derivative changes by -d(mean_Q(u))/dt - 3; this term MUST remain in
the autograd graph for the diffusion residual. Each query uses separate
quadrature evaluations at its own x,t. No detach, caching or unique-time
operation is used, and the underlying network must be pointwise.

This enforces the discrete mean, not the PDE, flux, positivity or a continuous
integral. A uniform initial state is preserved when the raw model already
enforces it. Other initial radial errors are not corrected by fixing the mean.
A deliberately underresolved polynomial test demonstrates that an independent
quadrature can disagree despite exact discrete conservation. Constant radial
offset modes of the raw network are removed and may have zero gradients.

Tests cover analytic means, first/second time and radial derivatives, parameter
gradients through the PDE, batch independence, gradcheck/gradgradcheck and
invalid inputs. The wrapper adds one radial quadrature per query, increasing
memory and runtime. Chunked evaluation is needed before integration into the
large validation grids.

## Training Integration

Run `python scripts/train_constant_flux_particle.py --variant inventory --sampling full_radius --seed 42`.
The startup network is wrapped in a 64-node projection without additional
trainable parameters. PDE/initial/flux losses and their weights are unchanged.
Mass penalties are explicitly disallowed in this variant. Initial weights and
the random collocation stream match startup controls. Optimization accumulates
full-objective gradients over 64-query chunks before each optimizer step;
these are not stochastic minibatch updates. Inference uses 128-query chunks.
Tests compare full and chunked losses/parameter gradients and checkpoint reloads.

Training retains 2000 Adam steps and 300 L-BFGS maximum iterations. Equal step
budgets are not equal runtime budgets: each query requires radial quadrature.
Final inventory errors use independent 128/256-node volume integration on the
usual validation times, without changing the internal 64-node projection.
`python scripts/compare_inventory_training.py` compares the latest projected
and unpenalized full-radius runs; explicit --projected/--control directories
pin the comparison. It checks matching settings, weights and original points,
reproduces saved metrics and computes physical-volume/time PDE diagnostics.

## Verification

On 2026-09-20, all 11 focused tests and all 145 repository tests passed.
Finite-difference derivative checks use interior inputs so their perturbations
remain in the valid domain; separate analytic tests cover radial boundaries.

## Seed-42 Training Evidence

Projected run: `results/constant_flux_pinn_20260920T204022465636Z`.
Matched control: `results/constant_flux_pinn_20260920T054646100272Z`.
Verified comparison: `results/inventory_comparison_20260920T210129246195Z/report.json`.
All saved metrics reproduced after checkpoint reload. Initial parameter hashes
and all original interior, initial and surface point arrays match. Integration
tests and the full suite passed (148 tests). Training took 1197.81 seconds on
the local CPU, excluding final evaluation and plotting.

| Metric | Control | Inventory projection |
| --- | ---: | ---: |
| Maximum concentration error | 2.683769e-3 | 6.391678e-5 |
| RMS concentration error | 1.656522e-4 | 8.558141e-6 |
| Maximum flux residual | 4.558831e-2 | 2.727221e-3 |
| Maximum mean balance error | 3.912067e-4 | 1.277610e-9 |
| Original log-time grid PDE RMS | 2.557995e-1 | 2.939058e-1 |
| Physical-volume/time PDE RMS | 3.064931e-2 | 1.659028e-2 |

Independent 256-node integration gives maximum inventory error 1.277594e-9;
128/256-node means differ by at most 1.654232e-14. These checks concern sampled
times and do not bound earlier times or continuous-domain extrema.

The initial, concentration, flux and mass targets in FLUX_PILOT_ACCEPTANCE.md
are met for this ONE seed only. The required all-seed acceptance is not met
by this experiment alone. The original log-time grid PDE RMS worsens, despite
better physical-volume/time RMS and concentration accuracy. Preserve this
regression rather than claiming uniform residual improvement. Next: assess
whether the computational cost warrants a matched multi-seed projected study;
do not claim DFN readiness or tune to seed 42 alone.
