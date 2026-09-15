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

## Fine Startup Sampling Audit

`python scripts/check_radial_320_640.py --fine-startup` compares x80/r320 and
x80/r640 over 0-0.2 s at 0.0005 s output spacing. Nested subsets at 0.001 s and
0.05 s show how the reported maximum depends on output sampling. The existing
1 mV and 0.1% concentration-scale limits remain unchanged. These are subsets of
the same solver runs, not independent integration-step convergence studies.
The audit does not establish a rigorous bound between output times.

### Observed Fine-Startup Result

Run `radial_320_640_fine_20260914T235112394264Z`, PyBaMM 26.8.0.0:

| Quantity | Densest sampled maximum | Peak time |
|---|---|---|
| Voltage difference | 0.55696939 mV | 0.0285 s |
| Negative surface concentration difference | 1.6719641 mol/m3 | 0.0045 s |
| Positive surface concentration difference | 13.288261 mol/m3 | 0.0265 s |

All three satisfy the existing limits. The 0.05-second grid missed part of each
peak. Refining the output grid from 0.001 to 0.0005 seconds changes the voltage
maximum by 0.00002655 mV, negative surface maximum by 0.002898 mol/m3, and positive
surface maximum by 0.0008779 mol/m3. This supports closing the identified
first-output-sample concern for these quantities at the present working limits.
It is not a formal continuous-time bound or certification of other fields.
The current exported x80/r80 HDF5 is unchanged; selection and export of a new
working reference remain separate steps.
