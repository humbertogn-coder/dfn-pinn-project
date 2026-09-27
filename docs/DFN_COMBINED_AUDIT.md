# Combined DFN Audit Evidence

Implemented September 25, 2026. This integrates existing historical smoke
reports; it is not a new simulation, training run or physical certification.

## Reproduction

```bat
python scripts/summarize_dfn_audit.py
```

The defaults pin explicit field, balance and PDE report paths within
results/dfn_smoke_20260924T054745712381Z. No latest-directory discovery is used.
The --fields, --balances and --pdes options accept paths relative to --run
or absolute paths. The historical smoke checkpoint/reference schema is
required; future training schemas are intentionally not accepted implicitly.

The integrator verifies current training/reference provenance, the reference
data hash, checkpoint settings and exact prediction replay. All three reports
must name that same checkpoint and reference. Balance/PDE reports must also
match the pinned configuration and record an unchanged checkpoint. Source
reports are hashed and retained, not overwritten. This is integrity tracking
for local scientific artifacts, not cryptographic authentication of authorship.

The required metric inventory is explicit: 11 native-field comparisons,
42 balance/boundary/initial/inventory checks, and 30 local PDE checks. Stored
PASS flags are checked against numerical values and unchanged protocol
thresholds. Concentration and current reference errors are converted using
the physical scales of the matched checkpoint. Missing metrics and unresolved
quadrature imply INCOMPLETE; mismatched identities, nonfinite values, altered
thresholds or inconsistent flags raise an error. Existing admissibility and
peak-location diagnostics remain in the linked component reports.

## Recorded Result

Output within the smoke run:
combined_audit_20260925T201301998426Z/report.json.

| Section | Passing metrics | Total |
| --- | ---: | ---: |
| Native reference fields | 1 | 11 |
| Balances and boundaries | 27 | 42 |
| Local PDEs and quadrature gaps | 12 | 30 |

Sampled metric status: FAIL.
Overall status: INCOMPLETE_NOT_ACCEPTED.

All 83 expected metrics were present. Passing quadrature checks are included
in these counts; counts are not an accuracy score or independent experiments.
The historical checkpoint has three recorded Adam iterations. Its completion
of a wiring smoke cannot demonstrate completion of the planned full budget.
Even a hypothetical all-metric PASS would not promote this historical schema
to full-budget acceptance. The provisional reference and sampled domain
limitations remain unchanged.

Seven focused tests passed in 7.14 s, covering complete-but-smoke evidence,
missing metrics/unresolved quadrature, failed physics, mismatched checkpoint,
changed thresholds, inconsistent flags and nonfinite metrics. No full suite,
training, simulation, commit or push was performed for this step.

## Next Bounded Work

The historical evidence integration is complete. Before a future training run,
define its versioned checkpoint/completion contract and separate physical
reference compatibility from optimizer and sampling settings. Then perform
the bounded cost/gradient/replay dry run. Do not assume the historical report
integrator already supports arbitrary future trainer outputs or clears the
full-budget launch gate.
