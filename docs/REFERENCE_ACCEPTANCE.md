# Provisional Reference Acceptance Criteria

These are project working targets, not universal standards. Version 1 applies
to the isothermal Chen2020 1C discharge without side reactions.

| Quantity | Provisional limit |
|---|---|
| Pairwise maximum voltage difference | 1 mV |
| Solid and surface concentration difference / material maximum concentration | 0.001 |
| Electrolyte concentration difference / initial electrolyte concentration | 0.001 |
| Global reaction-current residual / applied current magnitude | 1e-5 |
| Total lithium drift / initial lithium inventory | 1e-8 |

Report 0-5 seconds separately from the rest of discharge. Do not remove startup
samples to obtain a pass. Compare only shared times and coordinates, and report
cutoff times separately. No cutoff-time acceptance threshold is defined yet.
Percentages use fixed physical scales, not local concentrations.

The current radial study uses fixed x80 meshes and radial 160/320. Short runs
sample every 0.05 seconds through 5 seconds; full runs sample every 10 seconds.
The later window therefore starts at the first shared sample after 5 seconds
(normally 10 seconds). Values between samples, including 5-10 seconds, are not
certified. Native particle arrays are interpolated onto r80 centers with identical
x coordinates; surface outputs are separate. Interpolation error is included.

A sampled radial pass is not overall reference acceptance: through-cell
refinement at the selected radial resolution remains pending, as does a denser
sampling audit if required. Reference uncertainty must be smaller than the PINN
effect being measured; tighten these targets if comparisons require it.

Record solver settings, actual wall times, cutoff times, conservation metrics,
and estimated array storage. Storage estimates exclude compression and HDF5
metadata; they are not measured file sizes. Keep the existing HDF5 provisional.

## Through-Cell Study

`python scripts/assess_reference.py --study spatial` compares x80/x160 while
holding r320 fixed. It uses the same provisional limits and sampling windows.
A sampled pass is evidence for this pair, not a certificate for all times,
boundaries, or a combined spatial/radial reference choice.
