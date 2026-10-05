# Startup particle wiring smoke

## Technical scope

New explicit startup-particle wiring preserves all non-particle fields from
the frozen boundary step-200 checkpoint. Particle networks are fresh four-input
networks (12,12 hidden widths, seed 42), not historical three-input weights.
Only `cs.*` parameters are trainable. The eight particle subproblem terms and
their unit weights are unchanged. A fresh Adam optimizer uses learning rate
0.001 for three steps. This is not a matched performance comparison.

The startup checkpoint has a separate schema and explicit scales, beta,
Fourier-time scale, initial values, representation, seed and trainable-prefix
metadata. Restoration validates these against settings and checks fixed buffers.
No historical factory or source file was changed.

## Verification and results

`python -m pytest tests/test_particle_startup_variant.py -q -p no:cacheprovider`

**4 passed in 7.74 s**: non-particle tensor preservation, strict round trip and
rejection of altered scales, initial buffers and trainable-prefix metadata.

`python scripts/smoke_particle_startup_variant.py`

Run: `results/particle_startup_smoke_20260929T073928375126Z`.
Status: STARTUP_SMOKE_ONLY. Pre-step particle objective values are approximately
343442, 316156 and 290003. These large values must not be hidden by reporting
only their decrease.

Final residual RMS on the historical held-out diagnostic samples:

| Term | Negative | Positive |
| --- | ---: | ---: |
| Particle diffusion | 9.915792 | 548.562832 |
| Surface flux | 0.968583 | 0.977854 |
| Particle inventory | 1.05398e-4 | 6.47272e-4 |

On a separate range grid (17 radii, five x positions, zero time plus nine
positive log-spaced times), final normalized concentration ranges are
[0.7999362, 0.8000877] and [0.4, 0.4014887]. All four sampled checkpoints
stay within [0,1] and reproduce the exact initial concentration. This is a
sampled check, not a global bound or a conservation certificate.

Frozen branches stay exactly unchanged. Final metrics replay on training and
historical diagnostic samples, saved Adam state matches, and a disposable
continuation step reproduces model and optimizer state exactly. No fourth step
is accepted or saved. Original input/source hashes remain unchanged.

## Decision

The technical integration succeeds, but the very large diffusion residual
blocks any claim of physical improvement. Three steps cannot establish eventual
trainability, and the fresh initialization differs from the preceding warm-start
experiment. Do not automatically proceed to 200 steps on this evidence.

A specific representation risk merits checking first: multiplying a generic
raw network by sqrt(time) permits an O(sqrt(time)) correction throughout the
particle, not only in the near-surface layer. A nonzero interior raw value can
therefore produce an O(1/sqrt(time)) time derivative without a matching diffusion
term. This is an analytic risk, not yet a demonstrated decomposition of this
checkpoint's error.

Next bounded check: localize the early diffusion residual and separate its time
and radial contributions on the frozen checkpoint. If a bulk startup term is
responsible, consider confining the square-root contribution to a surface layer
while letting the bulk correction start at O(time). Treat that as a new declared
representation change, verify derivatives and initial/surface behavior before
training, and preserve this technically successful but physically poor smoke.
