# Working Reference v1

## Decision

Use x80/r320 as the development reference for the isothermal Chen2020 1C DFN
case, with initial SOC 1 and IDAKLU tolerances rtol=1e-8, atol=1e-10. This is a
practical working choice, not an exact solution or publication-wide validation.
The earlier x80/r80 HDF5 is retained as a legacy smoke-run artifact.

## Evidence

- Spatial x80/x160 at r320: maximum sampled voltage differences of 0.0594 mV
  at startup and 0.0588 mV later; tested concentrations and global balances pass.
- Radial r160/r320: later sampled voltage difference 0.137 mV; tested
  concentrations and global balances pass. Startup voltage fails at 1.102 mV.
- Radial r320/r640 startup: fine sampling resolves a voltage peak of 0.556969 mV
  at 0.0285 s; surface concentration differences also satisfy provisional limits.
- The sampled voltage maximum changes by 0.00002655 mV between 0.001 s and
  0.0005 s output grids. Analytic sphere benchmarks support the numerical and
  comparison procedures but are not full-cell validation.

See REFERENCE_ACCEPTANCE.md for limits and qualifications. Evidence comes from
the recorded PyBaMM 26.8.0.0 runs. Separate spatial and radial differences must
not be treated as a rigorous bound on their combined error.

## Export

Run `python scripts/export_working_reference.py`.

| File | Interval | Output spacing |
|---|---|---|
| full.h5 | 0 to 2.5 V cutoff | 10 s plus final event |
| startup.h5 | 0-10 s | 0.05 s |
| fine_startup.h5 | 0-0.2 s | 0.0005 s |

All three contain the 23 fields, native coordinates, units and terminal signals
in schema 1.0. Each file is reopened and compared exactly with its simulation.
The completed manifest records sizes and SHA-256 hashes. Generated results stay
out of Git. Each execution creates a new directory and preserves older files.

These are independent simulations, not stitched segments. Do not concatenate
their overlapping samples blindly or split overlaps into training and validation.
A future loader must choose one source per time interval and verify consistency
at overlaps before constructing a combined dataset. Densely sampled startup
must not dominate metrics merely because it contains more samples.

## Remaining Limits

The r320/r640 comparison only covers startup voltage and surfaces, not all fields
over the entire discharge. Fine temporal certification of internal particle
fields is incomplete. Potentials, reaction currents and their derivatives have
not received the same pointwise accuracy study. Ten-second later samples do not
bound intersample extrema. No rigorous continuous-time error bound is claimed.
Use this bundle to develop and debug the PINN; tighten and extend verification
before interpreting small differences as a publishable improvement.
