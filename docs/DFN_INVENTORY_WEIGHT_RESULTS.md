# Joint inventory-weight comparison

## Completed scope

Command: `python scripts/train_inventory_weight_pair.py`.
Run: `results/inventory_weight_pair_20260929T222006194779Z`.
Status: PAIR_COMPLETE_NOT_VALIDATED. Runtime after preparation: 111.88 s.
Both arms completed 200 Adam steps. Three focused tests passed in 11.15 s.
No extra training, weight search, seed sweep or full acceptance audit ran.

Both direct-kinetics models started from the pinned joint step-zero checkpoint,
with fresh Adam, identical training points and all fields trainable. The sole
objective change is inventory MSE weight 1 versus 1e8. The two inventory
residuals are multiplied by sqrt(weight) inside the objective only; common
physical metrics remain unweighted. Raw and weighted losses are saved separately.
Historical physics modules and checkpoints were not modified.

The control reproduced every historical training-history entry exactly.
Historical milestone model tensors, Adam states and common metrics also matched
exactly. Both arms passed milestone reload and disposable next-step model/Adam
replay. Sampled concentration bounds and hard solid-boundary identities passed.
These checks establish technical reproducibility, not a physical solution.

On September 30, after interruption, all 51 recorded input/source hash entries,
the saved sample hash and every milestone checkpoint hash were verified again.
No training was repeated. This document closes the previously unfinished write.

## Common evaluation results

The additional evaluation set uses seed 20261002, with 128 interior points per
region and 64 boundary times; it never entered training. Other evaluation sets
are reused diagnostics. All values below are normalized.

| Metric | Control | Weight 1e8 | Required limit |
| --- | ---: | ---: | ---: |
| Negative inventory maximum | 3.89476e-5 | 1.72547e-5 | 1e-6 |
| Positive inventory maximum | 6.60865e-5 | 3.30296e-5 | 1e-6 |
| Negative particle PDE RMS | 0.260001 | 0.459906 | 0.01 |
| Positive particle PDE RMS | 0.498442 | 0.293706 | 0.01 |
| Negative particle PDE maximum | 2.69887 | 4.90432 | 0.1 |
| Positive particle PDE maximum | 4.75748 | 2.36775 | 0.1 |
| Negative surface-flux maximum | 0.084653 | 0.063043 | 0.01 |
| Positive surface-flux maximum | 0.013819 | 0.085785 | 0.01 |
| Negative kinetic maximum | 0.951052 | 0.951113 | 0.01 |
| Positive kinetic maximum | 0.505650 | 0.499597 | 0.01 |
| Negative electrolyte-charge RMS | 1.70195 | 1.70207 | 0.01 |
| Positive electrolyte-charge RMS | 1.99000 | 1.99055 | 0.01 |
| Negative interior total-current maximum | 0.893559 | 0.893558 | 0.01 |
| Positive interior total-current maximum | 0.874263 | 0.874503 | 0.01 |

Inventory improves but remains about 17 and 33 times above its maximum-error
limit. Negative diffusion worsens while positive diffusion improves. Positive
flux RMS worsens from 0.00517 to 0.05171; its maximum is over six times the
control maximum. Charge and negative kinetics remain almost unchanged.
No joint improvement or acceptable DFN solution is established.

The control's positive diffusion RMS is 0.4984 on the new set versus 0.1103
on the previous joint-check set, despite identical checkpoint tensors. This
shows sampling sensitivity, not a changed control trajectory. Neither sample
provides a resolved continuous-domain bound.

## Decision

Close this fixed-weight experiment. Do not select the weighted arm as a
validated baseline, increase the weight or extend the budget automatically.
The result demonstrates an inventory tradeoff without repairing the dominant
electrolyte-charge/kinetic failure. It does not prove all weighting approaches
fail. Preserve both outcomes and the original acceptance gates.

The next intervention requires a strategy decision focused on unresolved
charge/kinetic coupling, not another automatic inventory-weight trial.
Independent reference comparisons, global lithium balances and resolved
quadrature remain absent for these models. Full validation is incomplete and
sampled criteria already fail. Technical completion is not physical PASS.
