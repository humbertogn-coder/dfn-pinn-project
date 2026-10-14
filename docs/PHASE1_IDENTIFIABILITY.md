# Phase 1: how many discharge curves does a Li-SPAN cathode need?

Results of Phase 1 of the identifiability proposal (October 2026). All numbers come from the files in
`results/lispan/phase1/`; figures from `report/phase1/make_figures.py`.

## Bottom line

* **One C/10 curve is not enough to predict other rates.** On the measured Simanjuntak cell, five calibrations on
  0.1 C fit equally well (15.2-15.3 mV) but predict 1 C with errors from 32 to 318 mV, depending on the contact
  resistance that was assumed or fitted. On synthetic data with a known answer the same happens: the C/10 fits match
  the true 0.1 C curve within 1 mV and miss 1 C by 146-278 mV (2 C by 236-306 mV).
* **A fast second rate fixes it; a second slow rate does not on real data.** Calibrating on 0.1 C + 1 C recovers
  every parameter within 2 % (synthetic; U0_1 within 0.6 mV) and predicts 2 C to 1.3 mV rms. A second LOW rate (0.1 C + 0.2 C) works in
  theory (synthetic: 1 C to 6.2 mV rms) but fails on the measured cell (1 C off by 217 mV): the voltage difference
  between two slow rates (tens of mV) is of the same size as the model's own rate-dependent error.
* **Correction to the proposal's hypothesis.** For an irreversible step the rate constant and the step potential
  enter the rate only as k0 exp(F U0 / 2RT), at EVERY current (k0 x f is the same as U0 + (2RT/F) ln f), so no set of
  discharge rates separates them (U0_2, U0_3 also set the initial S2- equilibrium, a negligible effect: their
  sensitivities are collinear with those of ln k0 to a correlation of 0.9995 or better). Only the combination k0 exp(F U0 / 2RT) is identifiable, and it is all a prediction needs. What a
  C/10 curve leaves open is the contact resistance, the kinetics of the reversible first step and (partly) these
  combinations.
* **Electrolyte transport is not identified by any set of rates** (posterior/prior sd 0.46-1.0), but in this thin,
  porous electrode it hardly changes the predictions.

## 1. Linearized analysis (scripts/lispan_phase1_fisher.py)

Truth: the calibrated Simanjuntak cell (classical fit to their measured 0.1 + 1 C points,
`results/lispan/fit_experiment/fit_0.1+1C_fdabs.json`). Sensitivities by central finite differences of the
finite-volume model for 14 parameters at 0.05, 0.1, 0.2, 0.5 and 1 C (100 points per curve on 2-95 % of the
discharge time). Gaussian noise 5 mV (and 15 mV as a check); prior: each parameter known within a factor of 2,
step potentials within 50 mV. Posterior covariance (F + P^-1)^-1; prediction uncertainty by linear propagation.

| measurement set | Z_CC (post/prior sd) | step 2-3 combinations | 1 C prediction sd [mV] | 0.05 C prediction sd [mV] |
| --- | --- | --- | --- | --- |
| C/10 | 0.82 | 0.31 | 170 | 9.6 |
| C/10 + C/5 | 0.13 | 0.034 | 7.9 | 1.3 |
| 1 C | 0.16 | 0.55-0.56 | (measured) | 32 |
| C/10 + C/2 | 0.09 | 0.027-0.030 | 2.5 | 1.1 |
| C/10 + 1 C | 0.05 | 0.026-0.031 | (measured) | 1.2 |
| all five rates | 0.04 | 0.02 | (measured) | (measured) |

At 15 mV noise the 1 C prediction sd is 171 mV (C/10), 20.5 mV (C/10 + C/5) and 5.8 mV (C/10 + C/2). k0_2, U0_2,
k0_3, U0_3 alone stay at 0.43-0.88 of the prior in every set, with correlations of -0.996 or stronger once two
rates are used. 2 C was left out of this analysis: the solver stalls at 2 C for some perturbed parameter sets
(b_2 +-5 %), where this cell reaches the cut-off at about a quarter of its capacity.

## 2. Calibrate-then-predict tests (scripts/lispan_phase1_fit.py)

Free parameters: k0_1..3, b_1..3, U0_1 and Z_CC (U0_2, U0_3 fixed because of the exact degeneracy); FV model,
least squares with absolute finite-difference steps. Queue: `scripts/run_phase1_fits.sh`, `run_phase1_fits_2.sh`.

Measured data (Simanjuntak et al. 2024, Fig. 4b, digitized points):

| calibration | Z_CC [Ohm m2] | 0.1 C misfit [mV] | 1 C error [mV] |
| --- | --- | --- | --- |
| 0.1 C, Z_CC = 0 (assumed) | 0 | 15.2 | 318 (prediction) |
| 0.1 C, Z_CC = 0.0125 (assumed) | 0.0125 | 15.2 | 207 (prediction) |
| 0.1 C, Z_CC fitted | 0.0107 | 15.3 | 217 (prediction) |
| 0.1 C, Z_CC = 0.035 (assumed; paper's text value) | 0.035 | 15.2 | 32 (prediction) |
| 0.1 C, Z_CC = 0.05 (assumed) | 0.05 | 15.2 | 134 (prediction) |
| 0.1 C + 0.2 C | 0.0079 | 15.6 | 217 (prediction) |
| 0.1 C + 1 C | 0.0331 | 20 | 16 (fitted) |

The 0.1 + 1 C calibration predicts 0.05 C and 0.2 C within 30 and 31 mV, the model's own error across rates.
The only C/10-only calibration that predicts 1 C well used Z_CC = 0.035, a value the original authors obtained by
comparing their model with the 1 C data.

Synthetic data (truth = calibrated cell, 5 mV noise on the calibration curves, 100 points per curve):

| calibration | 0.1 C vs truth [mV] | 1 C vs truth [mV] | 2 C vs truth [mV] |
| --- | --- | --- | --- |
| C/10, Z_CC = 1/2 truth (assumed) | 0.7 | 146 | 304 |
| C/10, Z_CC fitted (converged to 1/2 truth) | 0.7 | 148 | 306 |
| C/10, Z_CC = 2 x truth (assumed) | 0.7 | 278 | 236 |
| C/10 + C/5 | 0.5 | 6.2 | not computed |
| C/10 + 1 C | 0.2 | 0.7 (fitted) | 1.3 |

## 3. Consequences for the manuscript and for experiments

1. Minimal protocol: one slow discharge (C/10) plus one fast one (1 C, or C/2). Two slow rates are not enough on
   real data.
2. The degeneracy of irreversible steps is (practically) exact and harmless for prediction; report the combination
   k0 exp(F U0 / 2RT), not k0 and U0 separately. Absolute step potentials from discharge data alone are not
   meaningful for irreversible steps.
3. For SPAN500 (Wang et al., one rate only), the step potentials quoted earlier are conditional on the assumed
   kinetics; differences between materials and cycles remain meaningful.
4. Open: the rate-dependent model error (+18 / -20 mV at 0.05 / 0.2 C after the 0.1 + 1 C calibration) is now the
   limit, not the data; the literature multi-rate SPAN datasets (Phase 2) will show whether it is specific to this
   cell.

Figures: `report/phase1/figures/p1_measured.png`, `p2_ranking.png` (and `p2_ranking_sigma15mV.png`),
`p3_synthetic.png`.
