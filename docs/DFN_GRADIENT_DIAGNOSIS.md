# Frozen DFN Gradient Diagnosis

Status: diagnostic evidence only; no training, parameter update or physical
acceptance. Report: results/dfn_gradient_diagnosis_20260928T003146341355Z/report.json.

## Reproduction and Scope

```bat
python scripts/diagnose_dfn_gradients.py
```

The script examines the saved C0 and direct-kinetics hard-current F full runs.
It verifies training sources, checkpoint/config/reference identity, exact
prediction replay, unchanged model tensors and unchanged checkpoint hashes.
Autograd computes parameter derivatives without populating parameter .grad
buffers. No optimizer is constructed and no parameters are perturbed.

All 34 original normalized MSE terms are differentiated separately. Results
include norms by network branch, selected pairwise gradient cosines and the
norm of the total gradient relative to the sum of individual norms. Their
vector sum is checked against direct differentiation of the total loss.
Zero-gradient cosines are null, not artificially assigned alignment zero.

Two sample sets are used: the actual saved training samples and a new fixed
sample set with the same counts/distribution and seed 20260928. The latter is
saved as diagnostic_samples.npz with its hash. It is a sampling sensitivity
check, not a volume/time-weighted independent acceptance audit. Gradients are
Euclidean derivatives in the present parameterization, not Adam/L-BFGS update
directions or a measurement of global optimization conditioning.

Collector sensitivity uses 32 unique uniform/log-spaced positive times in
[1e-6, 1] s. Current is normalized by imposed geometric current. Hidden-layer
statistics below apply only at those collector/time queries, not everywhere
in the electrode.

## Main Evidence

| Quantity | C0 | Hard-current F |
| --- | ---: | ---: |
| Positive collector loss, training samples | 0.9999891 | 0.9999614 |
| Gradient norm of that loss | 3.246880e-5 | 1.088010e-4 |
| Positive kinetic-loss gradient norm | 0.03277529 | 0.1111341 |
| Positive collector current / imposed current, sampled range | 1.23e-6 to 8.37e-6 | 4.26e-6 to 2.97e-5 |
| Positive current Jacobian RMS row norm | 1.804012e-5 | 6.058375e-5 |
| Positive final hidden-layer mean tanh derivative | 4.177192e-4 | 2.786065e-4 |
| Fraction of final hidden activations with absolute value > 0.99 | 100% | 100% |

The large collector error produces a weak parameter-gradient signal. Its
loss gradient is about three orders of magnitude smaller than the positive
kinetic-loss gradient. Both trained positive-potential branches exhibit
saturation at the sampled collector. Negative-potential branches have no
activations above the same 0.99 criterion at their sampled collectors and
mean final hidden-layer derivatives near 0.85 (C0) and 0.82 (F).

The conductivity-dependent coefficient relating normalized solid-current to
normalized potential slope is 656.603 for the negative electrode and 0.549714
for the positive electrode, a ratio of about 1194. Both potential networks use
the same output correction amplitude 0.01. The required collector slopes in
global x/L coordinates are therefore -0.001523 and -1.819127, respectively.
This is a physical scaling asymmetry within the present parameterization,
not a sign or units error established by this diagnostic.

These observations support investigating potential representation/scaling.
They do NOT prove that saturation alone caused the failure or that changing
an amplitude will cure the coupled DFN. No alternative architecture was tested.

## Gradient Conflicts and Sampling

In F, the positive collector-current and positive solid-charge losses have
gradient cosine about -0.8375 on their shared positive-potential branch. This
persists on the alternate sample set. Their local first-order directions
oppose one another at the checkpoint; this is not a trajectory-level proof
that one prevents convergence. Kinetics versus collector gradients on that
branch have cosine only about -0.011 on training points, so it would be
incorrect to describe all losses as strongly opposed.

F total-gradient norm rises from 0.12346 on training points to 27.61724 on the
alternate set. Positive electrolyte charge is the largest alternate gradient
contributor (about 27.22); its sampled MSE rises from approximately 0.04 to
1.21. The summed-gradient norm ratio changes from 0.0726 to 0.8689. Thus a
small aggregate gradient at the saved training points does not establish
stationarity on other points. C0 total-gradient norms are 0.55151 and 1.02653.

Some apparent particle-flux/PDE conflicts change sign between sample sets.
They must not be treated as robust global conflicts from one sampled cosine.
The existing physical-measure PDE audits remain the accuracy evidence.

## Verification and Next Decision

Three focused analytical tests passed in 10.74 s: gradient sum/unused
parameters and cosine handling, nonfinite derivative rejection, and linear
solid-potential current sensitivity including physical scaling. Both real
checkpoints replayed exactly and remained unchanged. No full suite was run.

Next propose a separately labeled, controlled potential-representation/scaling
check before another full training or attributing the problem solely to direct
Butler-Volmer. Verify required collector slopes and gradient sensitivity on
simple analytic fields first. Preserve C0/F as historical comparisons; if a
new representation is later used for the kinetic ablation, apply it consistently
to both direct and inverse arms. Do not simultaneously change sampling,
weights, kinetics and architecture and call it a single-factor experiment.

This diagnostic does not authorize automatic retraining, threshold changes,
new seeds, Grace jobs, commits or pushes. No such action occurred here.
