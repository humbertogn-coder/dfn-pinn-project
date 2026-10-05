# Frozen short-probe gradient diagnosis

Input: `results/solid_potential_probe_20260928T190218279084Z`.
Output: `results/potential_probe_gradients_20260928T190551760912Z/report.json`.
Command: `python scripts/diagnose_potential_probe.py`.

No optimization or reference simulation ran. Source and checkpoint hashes,
amplitude metadata and exact final metric replay were checked for both arms.
Input artifact hashes and model tensors remained unchanged, and parameter
grad buffers were not populated. All 34 term gradients were evaluated on
training and fresh samples; their sum matches the total-loss gradient.
Four focused tests passed in 6.29 s, including an analytic affine-field
check of the new potential-value/spatial-slope Jacobians.

## Findings

The following cosines use only the shared positive-solid-potential parameters,
on held-out samples. Negative cosine means opposing Euclidean descent
directions locally; zero means orthogonal. These are not Adam step directions.

| Pair | Original | Ohmic |
| --- | ---: | ---: |
| Local solid charge vs kinetics | -0.537929 | -0.536858 |
| Collector current vs kinetics | 0.140380 | 0.165490 |
| Collector current vs local solid charge | 0.000566 | -0.061735 |

Thus the strongest selected opposition is charge versus kinetics, not
collector versus kinetics. Training-sample signs and values are similar.
Do not explain the preceding results as a strong collector/kinetics conflict.

Held-out branch gradient norms change from 0.070018 to 2.892042 for kinetics,
0.011126 to 1.992745 for local charge, and 0.008255 to 1.517418 for collector
current. Such norms depend on parameterization; bigger is not inherently better.

At 32 fixed collector times, the normalized-current Jacobian RMS row norm is
0.004166 for the original versus 0.726969 for the candidate. Despite this
greater sensitivity, predicted positive-collector current has the wrong sign:
original [-0.000515, -0.000244], candidate [-0.063962, -0.023800], target +1.
Sensitivity alone is not current accuracy.

Neither short-run positive branch has sampled hidden activations above 0.99
in absolute value at the collector. Final hidden-layer mean tanh derivatives
are 0.9220 and 0.9270. The saturation seen in the historical full-budget runs
is not yet present in these 20-step snapshots. This does not rule out later
saturation or saturation at unsampled locations.

Potential-value versus global-X-slope Jacobian row cosines range approximately
0.125 to 0.265 (original) and 0.114 to 0.269 (candidate). Their sensitivities
are neither identical nor fully independent. This observation alone does not
justify a new architecture or show why an optimizer chooses a particular path.

## Decision

Do not launch full training or introduce an architecture change on these
snapshots alone. A useful next bounded measurement is the first-order effect
of the actual next Adam direction, reconstructed from each saved initial
state and its 20-step history because optimizer state was not saved in the
probe checkpoints. Verify exact final replay before using that direction.
Alternatively, a clearly labeled Euclidean directional derivative can be
computed without replay; it must not be presented as an Adam prediction.
Keep the physics, samples and amplitudes frozen during this diagnosis.
