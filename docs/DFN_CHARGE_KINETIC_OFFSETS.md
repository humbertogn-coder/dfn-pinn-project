# Frozen charge and kinetic potential diagnosis

## Scope and verification

Command: `python scripts/diagnose_charge_kinetic_offsets.py`.
Output: `results/charge_kinetic_offsets_20260930T195936281384Z`.
Input: control step 200 from the completed inventory-weight pair. No optimizer,
training, accepted parameter change, or new checkpoint was produced.

The script verifies source/checkpoint/sample hashes and exactly replays all
five saved common-metric sets. It evaluates 81 points per region at seven
positive times (1e-6, 1e-4, 0.001, 0.01, 0.1, 0.5, 1 s), retains one-sided
interface traces, and writes SI profiles to CSV. Inverse/direct BV round trips
are checked against the frozen learned current. Original tensors and file
hashes remain unchanged. Necessary numerical checks are integrated; no broad
test suite or full independent acceptance audit was run.

## Two distinct potential errors

Kinetics depends on the potential DIFFERENCE:
`eta = phi_s - phi_e - U(c_s_surface)`.
Electrolyte current depends on spatial DERIVATIVES:
`i_e = kappa_eff * (B * d_x(log(c_e)) - d_x(phi_e))`.
Correcting a constant level cannot correct a spatial slope.

At 1 s, the negative electrode has:

- Learned j: 1.3930 to 1.3944 A/m2 active area.
- BV current from existing potentials: -0.01822 to -0.00447 A/m2 active area.
- Actual eta: -1.70 to -0.42 mV.
- Eta required by inverse BV for that learned j: +85.26 to +85.30 mV.

Thus the frozen negative potentials give the wrong sign of kinetic current
and an approximately 86.6 mV mean overpotential deficit at 1 s. This is an
algebraic consistency finding, not proof of why optimization stalled.

In the separator at 1 s, actual electrolyte current is 4.825 to 4.864 A/m2
geometric area versus the applied 48.6855 A/m2. The actual electrolyte
potential slope is about -15.9 V/m; with the frozen concentration, the slope
needed to carry the applied current is about -159.3 V/m.

Required slopes elsewhere are computed from `i_e_required = I_app - i_s`.
They enforce total-current closure with frozen solid current and concentration
only. They are NOT a solution to local electrolyte charge with frozen j when
solid charge itself fails. No slope correction was applied.

## Disposable constant-offset check

Using mean eta gaps over the 81 uniform electrode points at 1 s, shift all
three electrolyte potentials by -86.6401 mV, and the positive solid potential
by -85.6719 mV. Keep the negative solid potential unchanged. This removes
the two mean eta gaps at that time; it does not minimize global kinetic MSE.

On the existing weight-check sample set:

| Common kinetic metric | Original | Disposable shift |
| --- | ---: | ---: |
| Negative RMS | 0.946642 | 0.0122065 |
| Negative maximum | 0.951052 | 0.0207543 |
| Positive RMS | 0.230451 | 0.253917 |
| Positive maximum | 0.505650 | 0.426270 |

Negative kinetic RMS falls about 78-fold, but its maximum still exceeds the
0.01 gate. Positive RMS worsens despite a lower maximum. No global kinetic
success is claimed. The shifted fields are discarded, not selected as a model.

All nonkinetic residual arrays and current diagnostics are unchanged within
1e-10 on all saved sets (largest nonkinetic residual difference 8.88e-16).
Common electrolyte shifts preserve interface potential jumps; spatially
constant shifts preserve currents, local charge, and solid current boundaries.
The negative solid gauge stays fixed. This is NOT a pure gauge transform:
positive terminal voltage changes by -85.6719 mV, and kinetics changes.
It cannot be adopted without the full physical/reference checks.

## Decision

This separates two unresolved requirements: adequate potential levels for
kinetics and adequate electrolyte slopes for transport. More inventory weight
cannot directly supply either. However, this diagnosis does not prove that
the architecture cannot represent the solution, or that a new scaling will
train successfully. Loss magnitude alone is not gradient conditioning.

The next bounded design should separate potential level and spatial shape
explicitly, with thermal-voltage/ohmic scales, one unchanged solid gauge, and
preserved interface/current boundary conditions. Specify one controlled
potential-representation intervention before another paired training run;
do not adopt the measured offsets as a validated initialization or silently
combine offset calibration, new weights, projection and budget extension.

Status: FROZEN_DIAGNOSTIC_ONLY_NOT_VALIDATED. No new potential representation
has been implemented. Full DFN acceptance remains incomplete with demonstrated
physical failures. No Grace run, seed campaign or further training was launched.
