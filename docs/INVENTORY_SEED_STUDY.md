# Resumable Inventory Study

Run `python scripts/check_inventory_seeds.py` from the project root. The default
persistent directory is `results/inventory_seed_study_v1`. The SAME command
resumes that study. Use `--prepare-only` to initialize and verify reused seed 42
without training seeds 0,1,2,3. Use `--study PATH` for a distinct study.

On creation, the latest existing full-radius paired control study and projected
seed-42 run are pinned. Explicit `--controls PATH` and `--seed42 PATH` override
that choice only at creation. The runner checks control and completed-artifact
SHA256 hashes on resume. Source hashes are also pinned: changed source requires
restoring the matching version or creating a new study, not silently mixing
implementations. Keep the code unchanged while the study runs.

Seed 42 is audited and reused; only seeds 0,1,2,3 require fresh training.
Settings remain inventory/full_radius, 64 projection nodes, 2000 Adam steps,
300 L-BFGS maximum iterations and no mass penalty. Completed runs are audited
against their matching full-radius controls, including original training-point
equality, checkpoint reload, all metrics and independent PDE/mass quadratures.

Each attempt has separate training/audit logs and a new directory. The runner
never deletes or overwrites earlier training attempts. Progress and summary
JSON files are replaced atomically after each transition. An OS-level lock
prevents concurrent writers on the same machine and is released on process
exit. Do not run the same OneDrive-synced study on multiple machines at once.

A partial training has no saved optimizer state: resuming repeats THAT seed
from the beginning, not the last Adam step. Completed model artifacts waiting
for audit are reused. Previously completed and verified seeds are skipped.
Failures and interrupted attempts remain recorded; a failed seed is not
silently counted as a pass. Each subprocess has a four-hour safety timeout.
Keep the computer awake and connected to power. A power failure during file
sync can still damage artifacts; hashes detect changes, but are not backups.

Final summary includes each paired projected-minus-control difference, counts
passing the existing exploratory targets, and worst completed values. All five
seeds must be complete before acceptance can pass. The log-time PDE regression
is retained alongside the physical PDE metric. This is a robustness extension
on previously studied seeds, not independent confirmation or DFN validation.

Expected new training time is roughly 80-100 minutes on the measured CPU,
plus auditing overhead; actual duration may differ. No long training is launched
by the preparation command.

## Preparation Verification

The default study was prepared on 2026-09-21. Existing seed 42 was audited
against its pinned seed-42 full-radius control. A second --prepare-only run
verified hashes and skipped that seed without training or repeating the audit.
State: 1/5 completed, overall INCOMPLETE; seeds 0,1,2,3 remain pending.
Six focused runner tests and all 154 repository tests passed. No four-seed
training result is claimed by this preparation check.
