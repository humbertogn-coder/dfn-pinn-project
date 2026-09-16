# Five-Seed Single-Particle Pilot

Run `python scripts/check_single_particle_seeds.py`. It sequentially launches
the existing trainer with seeds 0, 1, 2, 3 and 42, 1000 Adam steps and 150 maximum
L-BFGS iterations. Architecture, losses, sample counts and evaluation stay fixed.
Each run is a fresh Python process; no model is reused. There is a 600-second
per-run timeout. This is CPU work; no HPRC or GPU resources are requested.

The existing seed jointly controls network initialization and interior sample
locations. The study therefore measures their combined variation, not the
isolated effect of initialization. Five seeds are a pilot, not a large-sample
robustness certificate. No acceptance threshold is chosen after seeing results.

Each timestamped study directory contains five logs, per-seed model/report/plot
folders, progress.json, runs.csv and summary.json. Reports include all attempted
runs, failure counts, means, medians, sample standard deviations and ranges.
Failed runs are excluded from descriptive statistics but never hidden. Any run
failure produces a nonzero final exit code. Interrupted studies retain progress;
rerunning starts a new study rather than silently resuming partial work.

Source SHA256 hashes are saved, and per-run reports retain Python/PyTorch
versions. Reported trainer time excludes imports and plotting; process wall time
includes them. Neither is a claim of acceleration over a numerical solver.
No best-seed selection, full DFN validation or new physics is introduced here.

## Recorded Study

Study `single_particle_seeds_20260916T184830999621Z`: five of five completed;
all 100 project tests passed. Default architecture and training settings unchanged.

| Seed | Maximum concentration error | Maximum mean drift |
|---|---:|---:|
| 0 | 1.312185e-4 | 3.503993e-5 |
| 1 | 8.042676e-5 | 1.250908e-5 |
| 2 | 1.050644e-4 | 2.162911e-5 |
| 3 | 9.361488e-5 | 2.582364e-5 |
| 42 | 8.728282e-5 | 2.001338e-5 |

All quantities above are normalized. Median maximum concentration error was
9.361488e-5; median RMS concentration error was 2.576459e-5. These are
descriptive results, not acceptance thresholds. Seed 42 reproduced the previous
pilot's printed metrics. Process wall times ranged from 21.79 to 26.10 seconds
per run (median 22.78 seconds). Tests ran concurrently with the early part of
the study, so these timings are not controlled performance benchmarks.
