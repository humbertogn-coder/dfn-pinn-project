# First Assembled DFN Smoke

Status: WIRING_SMOKE_COMPLETE_NOT_PHYSICAL_ACCEPTANCE.
Run: results/dfn_smoke_20260924T054745712381Z.
This is the first trainable whole-cell residual assembly, not a validated
DFN solution or reproduction of the earlier PyBaMM discharge reference.

## Connected Components

src/dfn_pinn/dfn_smoke.py contains 12 learned pointwise branches: three
electrolyte concentrations, three electrolyte potentials, two solid potentials,
two particle concentrations and two reaction currents. Mixed direct-BV uses
the SAME learned electrode current in particle flux, particle inventory,
electrolyte salt source and both charge equations. No current projection or
reference-data supervision is used. The 34 residual terms are:

- Three salt balances and three electrolyte charge balances.
- Two solid charge balances.
- Two particle diffusion, two surface-flux, two kinetic, two center and two
  particle-inventory residuals.
- Eight electrolyte interface continuity residuals across two interfaces.
- Two insulating solid boundaries and six collector conditions.

Four total-current diagnostics are reported separately, not added as redundant
boundary losses. The negative solid collector fixes the only potential gauge.
Potentials carry fixed initialization offsets in the same common gauge; those
offsets are not data-fitted or additional constraints.

## Explicit Smoke Configuration

SETTINGS in the module is saved with the checkpoint/report. It includes geometry,
porosity, solid fractions, particle radii/diffusivities, concentration scales,
conductivities, temperature, 5 A applied cell current, area 0.1027 m2 and explicit
pybamm_26_8 kinetics. Time window: 1 s; t_ref=3600 s. Coordinate chain rules
retain that distinction. Isothermal, no convection/side reactions.

Initial solid stoichiometries are deliberately synthetic: negative 0.8 and
positive 0.4, NOT the prior working reference's initial-SOC=1 values. Electrolyte
initial concentration is 1000 mol/m3. Concentration ICs and particle-center
symmetry are hard constraints; particle flux is evaluated only at positive
times because uniform initial concentration and switched current create an
incompatible exact surface/time corner. Potential algebraic consistency at
startup is NOT presumed from the initialization offsets.

All networks use two 12-wide tanh layers and float64 CPU. One seed (42), eight
interior points per region, three boundary times and 8-node inventory quadrature.
All residual MSE weights are one for this wiring check, not an endorsed
scientific loss normalization. Three Adam steps at 1e-4; one CPU thread.
No large training, seed study, reference simulation or full suite was run.

## Verification and Findings

One focused structural test passed before execution. The runtime checked all
34 residual shapes/values, finite gradients in all 12 branches on each step,
hard initial concentration errors <=1e-12, and exact saved-model replay of
all 12 output fields on frozen points. Source hashes, settings, optimizer state,
frozen samples, predictions and per-term losses/gradient norms are saved.

Total loss went from 510.1923 to 507.6451 over the three pre-update evaluations.
This small decrease is NOT an accuracy result. Final normalized total-current
diagnostic maxima are about 1.119 (negative interior), 1.002 (positive interior),
0.990 (negative collector) and 1.002 (positive collector). These order-one
errors demonstrate that the smoke model does not satisfy cell-current balance.
All physical accuracy and convergence claims remain open. The previous
variable-current particle trials still fail their own acceptance gates.

Reproduce only when another smoke execution is needed:

```bat
python scripts/run_dfn_smoke.py
```

## Matched Working Reference

Run: results/dfn_smoke_reference_20260924T195608554436Z.
Status: MATCHED_WORKING_REFERENCE_PROVISIONAL.
Script: scripts/export_dfn_smoke_reference.py.

The isothermal Chen2020 DFN was solved over 1 s at 5 A with explicit initial
stoichiometries 0.8/0.4 and electrolyte concentration 1000 mol/m3. No initial_soc
override was supplied. Geometry, transport scalar parameters, kinetics mode,
PyBaMM version and actual initial concentration arrays were checked against
the assembly settings. Solver tolerances were rtol=1e-9 and atol=1e-11.

Two joint mesh levels were run: x40_r160 and x80_r320. Their sampled maximum
voltage difference was 6.911225e-4 V (0.691123 mV). Fine-mesh voltage changed
from 3.884860 to 3.876321 V. Maximum total lithium drift was 2.831e-15 mol on
that mesh. Global conservation is not an internal-field accuracy certificate.

reference.h5 stores both meshes, coordinates, units, time samples, voltage,
current and 13 internal fields. report.json records settings, parameter
provenance, source/data hashes and conservation diagnostics. The finer mesh
is a provisional comparison reference, not an exact solution. This short
two-level study does not certify internal-field or continuous-time convergence.
No PINN training or full test suite was run in this reference step.

## Next Scientific Step

The isolated-operator phase is over for this assembly. Use the matched working
reference to assess the frozen assembly and set acceptance criteria. Do not
compare against the old initial-SOC=1 reference as if the problems matched.
Pin this setup and specify voltage/internal-field/conservation criteria before
any meaningful whole-cell training budget. A reference-matching task is not
another operator test and does not remove the particle-model limitations.
