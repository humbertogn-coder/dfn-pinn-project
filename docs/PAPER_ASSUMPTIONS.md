# Explicit assumptions of the Li-SPAN PINN paper (step 4)

Status 2026-10-08. Numbers in the tables come from `scripts/paper_assumption_checks.py`
(`results/paper/assumption_checks.json`, `.md`, `.png`); everything else is cited to the section of
`docs/LI_SPAN_MODEL_FORMULATION.md` (LS) where it was settled. Voltage differences are taken at equal delivered
charge (the axis of every discharge plot), from the charge passed at t = 100 s to 98 % of the smaller capacity.

What the paper claims, and only this: (i) a finite-volume (FV) reference of the Simanjuntak et al. (2024) Li-SPAN
model with every ambiguity closed and documented; (ii) a PINN that reproduces that reference (benchmark form below)
to a fraction of a millivolt; (iii) what discharge curves can and cannot identify, and a multi-rate inverse that
reaches the accuracy of an ideal estimator on synthetic data; (iv) calibrations to two measured data sets with
their limits; (v) an honest cost comparison. It does not claim a new SPAN model or predictive parameters for a
specific commercial cell.

## 1. Inherited from the model of Simanjuntak et al. (not tested here)

| # | assumption | why acceptable for this paper |
| --- | --- | --- |
| A1 | one-dimensional cell, isothermal | the paper's own model; temperature is not stated, 298.15 K assumed. A 10 K error through the RT/F terms alone moves the voltage 0.8 / 2.9 mV rms (0.1 / 1 C); kinetic Arrhenius effects are not modelled |
| A2 | SPAN as covalently bound S4 chains, three sequential one-electron steps (6 e- per chain = 1.5 e- per S), no solid-state diffusion | the paper's structural hypothesis; it sets the capacity 1254 mAh/g_S and the 1/3-1/3-1/3 split of Fig. 5a |
| A3 | OCV of each step linear in its reaction extent, U_m = U0_m - b_m (1 - c_m/c_ref) (Table 3) | fitted by the authors to their measured OCV |
| A4 | dilute-solution Nernst-Planck transport with the measured 1 M salt diffusivity, transference number and conductivity, used as constants (the paper cites concentration-dependent Lundgren 2014 correlations without giving them) | largest salt excursion of the benchmark 1.8 % (0.1 C) and 13.5 % (1 C); bracketing checks: conductivity -10 % -> 0.32 / 3.0 mV rms, salt diffusivity -20 % -> 0.12 / 0.96 mV rms |
| A5 | Li metal anode as a planar Butler-Volmer surface; no polysulfide shuttle (carbonate electrolyte, covalent sulfur) | as in the paper |
| A6 | Li2S precipitation with a solubility product, active area a_SPAN eps_Li2S (empirical) | as in the paper; Li2S kinetics are practically invisible in the discharge voltage (identifiability, LS 6.5) |
| A7 | contact resistance lumped into one series resistance Z_CC | as in the paper; it is a pure ohmic shift (checked: Z_CC 0.025 -> 0.035 Ohm m2 gives exactly 10 / 100 mV at 0.1 / 1 C) |

## 2. Closures where the paper and SI are silent or inconsistent (FV reference, LS 5.1-5.2)

| # | choice | evidence / effect |
| --- | --- | --- |
| B1 | faradaic current includes reaction 1: i_F = F a (r1 + r2 + r3) (SI S29 counts only S2- production) | reaction 1 carries one third of the capacity in the paper's Fig. 5a; S29 as printed cannot produce it |
| B2 | effective rate constants k0 = (2.95, 2.39, 2.34) x 1e-8 mol/m2/s fitted to Fig. 6a (Table 3 lists 1e-2, 1e-2, 1e-4) | with Table 3 the reactions sit at equilibrium and show no rate dependence, while Fig. 6a shows a Tafel response (0.17 V between 0.1 and 1 C); only a_SPAN k0 enters. Fit 16 / 18 mV rms to Fig. 6a |
| B3 | reaction 3 irreversible during discharge | with the literal mass-action reverse term (all three reversible) the voltage changes by 53 mV rms, 190 mV max at 0.1 C (a spurious oxidation of SLi by S2- during phases 1-2), contradicting Fig. 5a; at 1 C that variant does not even finish in reasonable solver time |
| B4 | S2- references: c_S,ref = 0.01 mol/m3 in the SPAN kinetics, c_sat = 1e-5 mol/m3 at K_sp = 10 | needed for the OCV break points (Fig. 3) and the saturation plateau (Figs. 5b, 7e); a 10x change of either moves the voltage by <= 0.01 mV |
| B5 | ion diffusivities from D_salt and t+ (the SI's S20-S21 swap t+ and 1 - t+) | dimensionally and physically consistent form; transport effects are small (A4) |
| B6 | permanent Li2S nucleation seed eps_seed = 1e-5 (the paper's initial value) | without it the seed dissolves within microseconds and Li2S never nucleates |
| B7 | capacity normalised by the sulfur implied by the SPAN concentrations (0.767 mg/cm2), not the 0.6 mg/cm2 of the text | reproduces the paper's capacity axes (1 C = 1 mA/cm2 = 0.96 mAh/cm2) |
| B8 | Z_CC = 0.025 Ohm m2 (Table 2) rather than 0.035 (text) | Fig. 4b is reproduced with 0.025; the measured-data fit independently returns 0.033 (LS 6.8) |

Validation of the closed model against the paper: LS 5.3 (Fig. 6a 16-18 mV rms; capacities at 0.05-1 C within 3.1 %
of Fig. 4b; species trajectories of Fig. 5a).

## 3. PINN benchmark versus the FV reference (what the PINN is asked to reproduce)

The PINN is compared with an FV solution of exactly the same equations; the benchmark differs from the FV
reference of section 2 in three numerical choices, each quantified (dV rms / max [mV], change of capacity to
1.0 V [mAh/g_S]):

| # | benchmark choice | instead of | 0.1 C | 1 C |
| --- | --- | --- | --- | --- |
| C1 | reaction 2 also irreversible (reversibility T, F, F) | reaction 2 reversible (T, T, F) | 2.66 / 5.63, +0.0 | 0.12 / 0.30, +0.0 |
| C2 | double layer removed (c_DL = 1e-6 F/m2) | c_DL = 0.1 F/m2 | 4.58 / 11.73, +6.0 | 3.37 / 9.91, +5.1 |
| C3 | 30 s current ramp, I tanh(t / 30 s) | 1 s ramp | 0.01 / 0.20, +0.0 | 0.06 / 0.95, +0.0 |
| C4 | FV grid 40 / 20 volumes for the reference | 20 / 10 (paper) | 0.01 / 0.01 | 0.08 / 0.09 |
| C5 | training window ends at 0.98 t_end (before the cut-off collapse) | full discharge | - | - |
| C6 | PINN only: dissolved S2- not represented (< 1e-8 of the sulfide), the S2- released by reactions 2-3 forms Li2S at once (eps_Li2S = eps_L0 + V_m c_S4,0 (xi_2 + xi_3)) | FV with S2- transport and precipitation kinetics (k0_L = 200 mol/m2/s) | included in the PINN error | included in the PINN error |

With reactions 2 and 3 irreversible the S2- activity no longer enters the SPAN kinetics, so C6 only removes the
(fast) precipitation step; because the PINN is always compared with the full FV solution, its reported error already
contains C6 (LS 6.1 estimated 0.03 mV / 0.04 mol/m3). C2 is the only choice above 3 mV: with a_SPAN = 1e7 1/m the paper's c_DL gives 100 F per m2 of cell, whose
discharge (Delta phi falls by ~1.6 V) releases ~160 C/m2 = 6 mAh/g_S and shifts the plateau transitions; it is kept
out of the benchmark because it adds a stiff, fast state (time constant of seconds) to the PINN without changing any
conclusion. All benchmark effects are at least 3x smaller than the model-vs-experiment misfit (18 mV, LS 6.8) and
far larger than the PINN error (0.1-0.2 mV), so the PINN accuracy is a statement about the benchmark equations,
not about the cell.

## 4. Inverse problems and data

| # | assumption | effect / check |
| --- | --- | --- |
| D1 | synthetic data: benchmark FV model (40 / 20 volumes) + white Gaussian noise 1 mV, 112 / 90 points at 0.1 / 1 C (one per 300 / 30 s, inside the PINN window) | the ideal estimator (nonlinear least squares with the same noise realisation, `lispan_fit_experiment.py --data nominal`) is the yardstick, not the truth |
| D2 | eight parameters estimated (k0 x3, b x3, U0_1, Z_CC) as log-multipliers / offsets; U0_2, U0_3 fixed | U0_2, U0_3 are exactly confounded with k0_2, k0_3 in the Tafel regime (only k0 exp(F U0 / 2RT) is identifiable, LS 6.5) |
| D3 | everything else (transport, Li2S, anode) fixed at nominal values | practically invisible in the discharge voltage (LS 6.5) |
| D4 | measured data 1: circles of Simanjuntak Fig. 4b digitized (12-22 points per rate), converted with the paper's capacity convention | digitization error a few mV; model misfit 18 mV dominates |
| D5 | measured data 2: SPAN500 C/10 cycle 3 of Wang et al. (Nat. Mater. 2026), one rate only; Simanjuntak geometry and transport kept; Z_CC from the EIS high-frequency intercept (2.2e-4 Ohm m2); effective sulfur fraction w_S fitted from the capacity | at ~1 A/m2 transport and series resistance contribute < 1 mV; the kinetic constants cannot be separated from U0 at one rate, so two kinetic hypotheses are fitted and their 1 C predictions compared (`scripts/lispan_span500.py`); SPAN500 has shorter sulfur chains than the S4 chain of A2, so its fitted U0 / b are phenomenological |

## 5. Not claimed

* That Table 3 of Simanjuntak et al. is wrong: only that the published curves are consistent with the effective
  constants of B2 under our reading of the equations; the authors' implementation (DLR) would settle it.
* Charging, cycling, ageing, or the 2-3 V region beyond the initial spike.
* Transferability of the fitted SPAN500 parameters to other rates (section 4, D5).
