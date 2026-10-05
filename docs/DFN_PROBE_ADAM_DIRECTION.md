# Actual Adam direction after the short probe

Input: `results/solid_potential_probe_20260928T190218279084Z`.
Report: `results/probe_adam_direction_20260928T191205328323Z/report.json`.
Command: `python scripts/diagnose_probe_adam_direction.py`.

## Procedure and verification

The 20 original Adam steps were reconstructed from the saved initial weights
and samples for each amplitude. Every recorded loss and branch gradient norm,
the final parameter tensors and final sample metrics matched exactly. This
recovered the missing optimizer moments without assuming a fresh optimizer.

One next Adam step was evaluated only on a disposable model/optimizer copy.
Its parameter displacement was dotted with each of the 34 frozen term gradients
on training and held-out samples. Additional disposable evaluations at 0.001,
0.1 and 1 times this displacement check the local prediction against actual
loss changes. No candidate checkpoint was saved or accepted, no reference
simulation ran, and original input/source hashes remained unchanged.

The per-term directional derivatives sum to the total derivative within the
asserted tolerance. On held-out points, the maximum absolute difference between
the 0.001-step finite-difference slope and its autograd prediction is about
4.0645e-5 for either arm. This is a recorded consistency check, not a bound on
finite-difference error. The bounded diagnostic took 11.98 s after preparation.

## Results

Actual changes in normalized term MSE on held-out samples are below.
Negative means improvement; these are loss changes, not RMS residual values.

| Term | Original | Ohmic |
| --- | ---: | ---: |
| Total loss | -7.81655 | -7.79201 |
| Negative electrolyte mass | -2.83243 | -2.83246 |
| Positive electrolyte mass | -4.89707 | -4.89703 |
| Negative solid charge | -0.0548116 | -0.0135686 |
| Positive solid charge | -0.00152118 | +0.00007815 |
| Positive kinetics | -0.00245021 | -0.0122636 |
| Positive collector current | -0.00001219 | -0.00762096 |
| Positive separator solid current | -4.5154e-9 | -0.00011085 |
| Positive particle flux | -0.00030813 | -0.00030808 |

First-order total predictions are -7.86307 and -7.83916. Their difference from
the full candidate changes reflects finite-step nonlinearity. The candidate's
positive kinetics change is predicted as -0.0130954 versus actual -0.0122636.

## Interpretation and bounded next step

Actual Adam does not force a choice between collector current and kinetics
at this snapshot: both improve. The scaled candidate slightly worsens positive
solid charge while improving both of those terms. This supports the local
tradeoff identified by the gradient diagnosis, but is not a proof of a lasting
optimization conflict. Most total improvement comes from electrolyte mass
terms, so total loss alone obscures the current/charge behavior.

The amplitude candidate should not be declared ineffective from just 20 steps,
nor accepted from this one candidate step. Stop adding snapshot diagnostics.
A reasonable next authorized experiment is one matched, fixed 200-step Adam
probe, preserving the original and scaled arms, samples, loss weights, seed
and physics. Save optimizer state and all term metrics at fixed milestones,
and check held-out collector current, charge, kinetics and particle flux before
deciding whether a full-budget comparison is justified. No L-BFGS, seed sweep,
new architecture, weight tuning or automatic extension is included.
