# First Controlled DFN Baseline

Status: SPECIFICATION_ONLY_NOT_TRAINING_READY. September 25 2026.
This step freezes an exploratory protocol and training samples. The independent
whole-cell auditor and trainer are not implemented by this preparation step.
No new training, seed sweep, reference simulation or Grace job is authorized
automatically by running the preparation command.

## Scientific Question

Can the existing mixed-current DFN representation learn the matched one-second
problem while satisfying independent local and global screens? This baseline
is not the proposed conservation-projection contribution: inventory remains a
soft residual, and the constant-flux pilot's projection is NOT silently assumed
to extend to this coupled model. Establish the baseline before an ablation.

The reference remains the isothermal Chen2020 synthetic initial state at 5 A,
negative stoichiometry 0.8, positive 0.4, electrolyte concentration 1000 mol/m3
and temperature 298.15 K. It is not the older initial-SOC=1 discharge problem.
Configuration: configs/dfn_baseline_v1.json. Reference HDF5 is pinned by SHA256;
the preparation also checks its recorded source hashes and assembly settings.

## Fixed Training Design

Keep the existing 12 branches, two hidden layers of width 12, output mappings,
constitutive mode and physical scales. Start fresh with seed 42, float64 CPU,
one thread. Do not warm-start from the smoke checkpoint. Increase the number of
fixed interior points to 128 per region and positive boundary times to 64.
Use order 32 radial and current-integration quadrature in the training inventory
term. The existing architecture is an exploratory baseline, not a claim that
its capacity or initial amplitude is sufficient.

For each region, half of the times are uniform in physical time and half are
logarithmically distributed between 1e-6 and 1 s. x and particle radius are
uniform within their respective domains. Seed 20260925 fixes sampling.
Boundary times additionally include both interval endpoints. Model inputs
remain GLOBAL x/L, r/R and t/3600, not local x or t/1. Concentration ICs at
t=0 are audited separately; the incompatible surface-flux corner is excluded.
This design does not claim resolution below 1e-6 s.

Planned optimizer budget: 2000 Adam steps at 1e-3 followed by L-BFGS with
max_iter=200, max_eval=250, history_size=50 and strong Wolfe line search.
Every existing normalized residual MSE has weight one. No reference labels
enter any loss. Unequal physical scales can still produce unequal optimization
influence; these weights define a reproducible baseline, not optimal balancing.
The 34 terms and four separate diagnostics retain the smoke contract.

The future trainer must save model, optimizer, step, frozen samples, config,
source/reference hashes and wall-time accounting at least every 100 Adam steps,
before L-BFGS and after completion. Stop on nonfinite values or a two-hour wall
cap checked at optimizer/closure boundaries; label interrupted work INCOMPLETE.
The cap is a resource policy, not a runtime prediction. A measured short cost
and replay check is required before launching this budget. No extra seed or
budget extension follows automatically from a failure.

## Independent Evaluation

Use every saved native reference point for field errors, with no spatial
interpolation and no fitted gauge shift. Independently sample residuals at
41 positions per region, 41 radial locations and the union of 201 uniform
and 121 logarithmic positive times. Include exact interfaces, collectors,
particle surfaces and center as separate boundary traces; interior PDE samples
must exclude interfaces. Record initial conditions separately. Process grids
in chunks without changing pointwise derivatives.

Physical PDE RMS uses region dx and uniform physical time measures; particle
averages additionally use 3*rho^2*d_rho. Compute each equation's RMS separately,
never average different equations into one acceptance metric. Use independent
Gauss-Legendre orders 32 and 64 for spatial/radial/time integrations. If their
metric gap exceeds 5% of that metric's limit, evaluate order 128 and require
the 64/128 gap to pass. Otherwise mark quadrature unresolved, not accepted.
This numerical check is not a rigorous error bound. Report sampled maxima too.

## Exploratory Limits and Units

All limits are defined before the future full-budget result. The existing
three-step smoke is not a preregistered confirmatory experiment. No threshold
was selected to make that smoke checkpoint pass.

| Quantity | Required limit |
| --- | ---: |
| Voltage and each potential error | 0.005 V |
| Electrolyte concentration error | 10 mol/m3 |
| Particle and surface error divided by electrode c_max | 0.001 |
| Reaction-current error divided by electrode j_ref | 0.01 |
| Initial normalized concentration error and center gradient | 1e-12 |
| Local normalized particle mean minus integrated-current target | 1e-6 |
| Total phase-current error divided by i_app | 0.01 |
| Integrated electrode reaction current error divided by I | 0.01 |
| Total lithium drift divided by initial total lithium | 1e-7 |
| Electrolyte lithium drift divided by initial electrolyte lithium | 1e-5 |
| Normalized surface flux and kinetics residual maxima | 0.01 |
| Each normalized boundary or interface residual maximum | 0.01 |
| Normalized gauge residual maximum | 0.01 |
| Each normalized PDE physical RMS | 0.01 |
| Each normalized PDE sampled maximum | 0.1 |

The field screens retain the preceding comparison's limits. A 1% relative
current limit corresponds to 0.05 A at cell level. The particle mean threshold
is absolute in c/c_max; total-lithium thresholds are relative to initial
inventories, not to the much smaller amount transferred in one second. Report
both absolute mol and transferred-charge fractions as diagnostics so a small
relative drift does not hide a large fraction of the transferred lithium.
These are engineering screening choices, not literature-derived universal
accuracy requirements or experimental measurement tolerances.

Normalization follows the frozen operators, not automatic per-batch scales:

- i_ref = I/A; j_ref,k = i_ref/(a_k*L_k); phi_ref = R_g*T/F.
- Charge PDE divided by i_ref/L; particle PDE divided by c_max/t_ref;
  electrolyte mass PDE divided by c_e_ref/t_ref, with t_ref=3600 s.
- Interface concentration divided by c_e_ref, potential by phi_ref, phase
  current by i_ref, diffusive salt flux by c_e_ref*L/t_ref.
- Surface flux expressed in current units and divided by j_ref; kinetics
  current mismatch divided by that same j_ref. Gauge is phi_s_n(0)/phi_ref.
- Center condition is d(c_s/c_max)/d(r/R)=0. No solid-potential continuity
  across the separator is imposed, and no second potential gauge is introduced.

For global lithium use physical area, electrode thickness, solid fractions and
porosities: N_s,k=A*integral_x(epsilon_s,k*c_max,k*mean_r(theta_k)) dx;
N_e=A*sum_regions integral_x(epsilon_e*c_e) dx. Compare N_s,n+N_s,p+N_e
with its independently integrated initial value. Also test electrode reaction
integrals A*integral(a*j)dx against +I and -I and total phase current in the
separator as well as both electrodes and collectors. These must be actual
field/current quadratures, not balances copied from a prescribed target.

Require finite fields, positive electrolyte concentration and strictly interior
solid stoichiometries at all audited points. Report local signed reaction-current
ranges without imposing an unjustified pointwise sign constraint on the full
cell. Report any concentration overshoots instead of clipping them.

## Decision Policy and Next Work

Every required check must pass to report SAMPLED_EXPLORATORY_TARGETS_PASS.
Missing or nonfinite metrics, interrupted training, or unresolved integration
give INCOMPLETE or FAIL, never PASS. Even a complete pass is not continuous-time
certification, protocol generalization, experimental validation or novelty.
The fine reference's internal-field uncertainty is still provisional; a close
comparison alone cannot certify physical accuracy below that uncertainty.

Next implement the independent auditor and check its response to deliberately
wrong current/flux fields. Then measure a short trainer dry run with replay,
without changing this configuration. Only after those launch gates are met
should the single specified full-budget run begin. Do not rerun earlier pilot
studies or the full repository suite merely to prepare this experiment.

Preparation command, which does NOT train:

```bat
python scripts/prepare_dfn_baseline.py
```

Generated samples, configuration snapshot and manifest are saved in a new
results/dfn_baseline_preparation_* directory. Git excludes those artifacts;
the committed configuration and fixed sampling seed allow regeneration, but
the pinned HDF5 reference must be backed up separately.

## Preparation Evidence

Run: results/dfn_baseline_preparation_20260925T052056924182Z.
The pinned reference data and source identities matched. Saved samples reloaded
exactly: 128 (X,tau) points per region, 128 (rho,X,tau) points per electrode,
and 64 shared boundary times. Three focused preparation tests passed, covering
deterministic coordinates/scales, invalid sampling and changed reference data
or settings. No optimizer, reference solver or full test suite was run.
