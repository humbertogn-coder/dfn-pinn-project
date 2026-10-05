# Frozen SI coupling profiles

Command: `python scripts/inspect_boundary_coupling.py`.
Input: `results/boundary_200_20260928T201841393914Z`.
Output: `results/boundary_coupling_profiles_20260928T202455325215Z`.

The output contains `profiles.csv`, `report.json` and `profiles_at_1s.png`.
The figure was visually inspected. No training or reference simulation ran.

## Verification and scope

Input/source hashes and final checkpoint metrics on both saved sample sets
were verified before profiling. Model tensors and input hashes remained
unchanged. Profiles use 81 points per region, including separate one-sided
interface traces, at 1e-6, 1e-4, 0.001, 0.01, 0.1, 0.5 and 1 second.
There is no interpolation across interfaces. Extrema are sampled, not
continuous-space/time bounds or a full acceptance audit.

The surface diffusion flux is independently converted from normalized radial
concentration derivatives using `N_out=-D*c_ref/R*d(c_hat)/d(rho)`. Its current
equivalent is `F*N_out`, which must equal the shared learned interfacial j.
The normalized difference was checked against the existing surface residual.
This confirms the diagnostic's conversion/sign convention, not the model's
physical correctness. Butler-Volmer current is evaluated from the saved
surface concentration, electrolyte concentration and potentials.

Through-cell i_s and i_e are A/m2 of geometric electrode area. Interfacial j
and F*N_out are A/m2 of active surface area. Do not compare them directly:
the source coupling is a_s*j in A/m3. Positive interfacial j means outward
particle flux, so negative j in the positive electrode requires inward flux.

## Findings at 1 second

Applied current density is **48.68549 A/m2 geometric area**.

| Region | Electrolyte current range [A/m2 geometric] |
| --- | ---: |
| Negative electrode | 0.15818 to 0.95002 |
| Separator | 0.44521 to 0.45278 |
| Positive electrode | -0.01548 to 0.06126 |

The separator should carry the entire applied current through the electrolyte,
but carries less than 1% at these samples. The solid current obeys its imposed
endpoint values while total interior current drops far below the applied value.
Near-correct collector values do not imply through-cell conservation.

| Active-surface quantity [A/m2 active] | Negative | Positive |
| --- | ---: | ---: |
| Learned reaction j | 1.45376 to 1.45802 | -1.66859 to -1.66640 |
| Particle diffusion equivalent F*N_out | -1.4270e-5 to 4.9366e-5 | 2.2956e-4 to 3.0047e-4 |
| Butler-Volmer current | -0.002941 to 0.000519 | -2.59696 to -1.19831 |

Particle diffusion carries almost none of the reaction-implied current. The
positive particle diffusion flux is outward, opposite to its negative learned
reaction current. Negative kinetics is also nearly inactive relative to its
learned reaction current; positive kinetics has the correct sign but varies
substantially along the electrode.

Learned reaction sources are approximately +5.58e5 A/m3 in the negative
electrode and -6.37e5 A/m3 in the positive electrode. Electrolyte charge
residuals are approximately -5.5e5 and +6.36e5 A/m3, respectively: its current
divergence is not matching those sources. These are predictions of an
unvalidated model, not experimental observations or reference data.

## Decision

The hard solid boundaries resolve only one component. Remaining failures
include electrolyte current transport, particle surface diffusion and negative
kinetics. This diagnostic locates mismatches; it does not establish whether
their cause is scaling, initialization, weighting, representation or budget.

Do not add multiple hard constraints or extend full training automatically.
Before another training intervention, compare the physical-to-normalized
residual coefficients and branch gradient magnitudes for electrolyte charge,
particle diffusion/flux and negative kinetics. That bounded check should
inform one isolated correction or subproblem, not another simultaneous change
to architecture, weights and optimizer settings. Preserve this failed candidate.
