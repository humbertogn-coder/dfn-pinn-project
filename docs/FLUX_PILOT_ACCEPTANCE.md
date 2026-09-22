# Exploratory Particle Targets, Version 1

These proposed engineering targets were set AFTER inspecting five-seed results.
They are not standards, preregistered criteria, or DFN application tolerances.
They express a desired pilot accuracy budget without adjusting limits to pass
the observed runs. Owner review and application-specific justification remain
necessary before treating them as final acceptance requirements.

| Quantity | Target | Interpretation |
| --- | ---: | --- |
| Initial maximum error | 1e-12 | Numerical check of hard initial enforcement |
| Positive-time maximum concentration error | 1e-3 | 0.1% of unit initial concentration |
| Maximum surface flux residual | 1e-2 | 1% of imposed unit outward flux |
| Maximum mean balance error | 6e-5 | 0.1% of total extracted mean concentration, 3 times 0.02 = 0.06 |

The concentration target is a round resolution goal for this synthetic problem,
not a voltage-error guarantee. The flux target controls boundary consistency.
The balance denominator uses extracted material rather than the much larger
initial inventory, so a small absolute drift cannot hide a large relative error
in the transported amount. It uses final extraction for every time, not a
time-dependent denominator singular at startup. These quantities use synthetic
scales; do not transfer the numeric tolerances directly to Chen2020 DFN fields.

All five seeds must satisfy each criterion individually. Averages cannot hide
failed seeds. Missing, failed or invalid runs prevent a complete assessment.
PDE RMS remains diagnostic: its absolute magnitude has no justified mapping to
the required field/voltage errors here. No arbitrary PDE threshold is added.

Run `python scripts/assess_flux_pilot.py`. The latest study is selected even if
incomplete; use `--study PATH` to pin a study directory. The saved summary is
hashed, not revalidated against checkpoints. No simulation or training is run.
Reports record all individual values, limits, ratios and worst cases. A FAIL
is a finding, not a script execution failure.

The analytic comparison samples positive times from 1e-5 to 0.02; neither the
unresolved earlier interval nor intersample extrema are certified. Any future
acceptance claim needs a frozen method, independent seeds, sampling audits and
an application-specific error budget. Retain current failures as controls.

## Observed Assessment

Source: `results/flux_particle_seeds_20260920T071332457325Z/summary.json`.
Report: `results/flux_pilot_assessment_20260920T073056720093Z/report.json`.
Nine focused evaluator tests passed. No retraining was performed.

Both samplers fail these exploratory targets. Each has 5/5 passing initial
conditions, 3/5 passing concentration, 1/5 passing flux and 0/5 passing mass
balance. Worst mass errors are 4.901422e-4 (legacy) and 3.912067e-4
(full_radius), versus the proposed 6e-5 limit.

Next experiment proposal: isolate a mass-balance enforcement change, retaining
the current architectures/sampling as controls and measuring concentration and
flux regressions. Do not relax the limits solely to make existing runs pass.
