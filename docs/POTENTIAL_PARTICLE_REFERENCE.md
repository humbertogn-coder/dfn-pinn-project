# Potential-Driven Particle Reference

This is the first variable-current physical case after the manufactured
particle coupling test, not a full DFN. Only a numerical reference has been
computed; no PINN has been trained on this case.

## Problem

Negative Chen2020 particle: R=5.86e-6 m, D=3.3e-14 m2/s, c_max=33133 mol/m3,
uniform initial stoichiometry 0.5, electrolyte concentration 1000 mol/m3,
T=298.15 K, phi_e=0. Physical duration is 10 seconds.

    phi_s(t) = U_n(0.5) + 0.005*(1-cos(pi*t/10))/2  [V]

This starts at equilibrium with zero initial current and zero slope of the
potential protocol. Current is an algebraic unknown constrained by symmetric
one-electron Butler-Volmer using the instantaneous surface concentration.
Surface diffusion flux uses the SAME current; it is not prescribed.

The reference uses PyBaMM finite volumes and IDAKLUSolver, independently of
the Torch PINN implementation. Raw Chen2020 OCP/exchange-current functions are
used explicitly. Future Torch training MUST use mode='raw' for this case;
the previous manufactured case used mode='pybamm_26_8'. Do not silently mix
those conventions. This raw-fit benchmark does not change the full-DFN
constitutive contract.

## Verification

Script: scripts/check_potential_particle_reference.py.
Output: results/potential_particle_reference_20260923T004135144229Z.
Meshes: 80, 160, 320 radial cells (uniform). Standard rtol/atol=1e-9/1e-11;
one 320-cell repeat uses 1e-10/1e-12. Output times combine a 1001-point uniform
grid with logarithmic startup samples from 1e-6 to 0.1 seconds.

| Comparison | Maximum surface stoichiometry difference | Maximum current difference [A/m2] |
| --- | ---: | ---: |
| 80 vs 160 | 1.627962e-6 | 1.388877e-7 |
| 160 vs 320 | 4.266984e-7 | 3.639917e-8 |
| 320 standard vs tight | 4.354918e-10 | 3.755431e-11 |

Sampled checks passed: finest-pair and tolerance differences <=1e-5 in
surface/common-grid concentration and <=1e-4 A/m2 in current; inventory
residual <=1e-9 and BV residual <=1e-7 A/m2 for every run. These thresholds
were set in the script before execution. Internal comparisons interpolate to
80-cell centers, so they include representation effects. The finest mesh is
not exact, and continuous-time extrema have not been bounded.

Finest current ranges from numerical zero (-1.58e-30) to 0.06614334617 A/m2.
Surface stoichiometry ranges from 0.4997022973 to 0.5. Maximum inventory
residual is 3.184830e-15 and BV residual 2.210933e-12 A/m2 for the tight run.
Charge is integrated as a solver state. Its consistency with discrete particle
mass is a shared semidiscrete identity, not independent certification of
accuracy. Four short solves were run, not a new training or seed study.

## Next Bounded Step

The PINN setup and prespecified acceptance criteria are now implemented and
documented in POTENTIAL_PARTICLE_EXPERIMENT_V1.md. No new-case training has
been run. The following reference-use requirements remain in force.

Use the tight 320-cell data as a working numerical reference. Before training,
define explicit validation thresholds and a PINN with uniform initial state,
the prescribed potential above, and learned time-varying current. Do not use
reference current or concentrations as training labels without explicitly
changing the experiment. Relative current errors near equilibrium require
absolute tolerances because the reference current approaches zero.

After this variable-current check, proceed to electrolyte transport and charge
equations rather than adding open-ended particle-only tuning. Saved reference
arrays are in reference.npz with settings/provenance in report.json; they are
ignored by Git and need separate backup.
