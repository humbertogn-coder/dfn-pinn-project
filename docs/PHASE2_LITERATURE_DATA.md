# Phase 2: published multi-rate SPAN data - assessment and pilot

October 2026. Five files supplied by the owner were assessed for the predictive test of the identifiability proposal
(calibrate on one or two discharge rates, predict the others). Curves extracted: `data/literature_span/` (README there).
Pilot fits: `scripts/lispan_lit_rates.py`, results `results/lispan/literature_rates/`, figures `report/phase2/`.

## 1. What each paper offers

| paper | material, cell | multi-rate voltage curves | other useful content | verdict |
| --- | --- | --- | --- | --- |
| Chen et al., Nat. Commun. 10, 1021 (2019) | Se0.06SPAN (Se-doped), ~1 mg/cm2, carbonate (1 M LiPF6 EC/DMC) AND ether (LiTFSI DOL/DME + LiNO3) | Fig. 3b,d: 0.4, 1, 2, 4, 6, 10 A/g in both electrolytes (+ first cycle at 0.2 A/g); vector graphics, extracted exactly | same material in two electrolytes; first-cycle curves; DFT Li-migration barriers (1.33 vs 0.39 eV) | **use** (best data quality; Se doping makes parameters effective) |
| Ma et al., Small 22, e74169 (2026) | SPAN and SPAN aerogel, ~1.9 mg S/cm2, carbonate (1 M LiPF6 EC/DMC/EMC) | Fig. 6c,d: 0.1, 0.2, 0.3, 0.5, 1, 2 C for both materials (first cycle at each rate); digitized | EIS of both materials (Fig. 4f,g: charge-transfer arcs ~20 (SPAN-A) vs ~63 (SPAN) Ohm after CV, ~48 vs ~120 Ohm before, on 1.13 cm2), CV vs scan rate, GITT diffusion coefficients, Tafel plots | **use** (closest chemistry to the model: plain SPAN in carbonate) |
| Moschner et al., Adv. Sci. 12, 2415436 (2025) | commercial SPAN (ADEKA SAM-1, 38 wt% S), 3.5 mAh/cm2, 102 um, porosity 59 -> 49 % by calendering, ether (LiTFSI/LiNO3 in diglyme) | only C/50 and C/10 (44 % compacted electrode) and C/10 for the others; rate tests shown as capacities only (SI Fig. S8) | written with the authors of the Li-SPAN model (Danner, Simanjuntak, Latz); full model parameter tables for this cell (SI Tables S4-S6); EIS + electronic conductivity measurements; simulations showing that a thin, electronically poor layer at the current collector shifts the voltage with little change in capacity (a contact-resistance signature), while an ionically poor layer cuts capacity | **not for the rate test, but the most useful for interpretation**; its raw C-rate curves (Fig. S8) would be the ideal Phase 2 dataset (known geometry, porosity, conductivity, same model family) - ask the authors |
| Hao et al., ACS Nano 19, 25385 (2025) | FeS2/SPAN composite and SPAN (53 wt% S), 1.5-2 mg/cm2, carbonate (1 M LiPF6 EC/DEC/FEC) | Fig. 3d: 0.1-2 C for FeS2/SPAN only (plain SPAN in SI Fig. S9, not available); low-resolution raster | GITT of FeS2/SPAN; high-loading (6.1 mg/cm2) single-rate data | **low priority**: FeS2 conversion (~15 % of the capacity) is not in the model, and its rate dependence is small (the 2 C curve lies 0.12-0.15 V below 0.2 C at equal capacity; 781 -> 649 mAh/g from 0.1 to 2 C) |

Side notes:
* Moschner et al. apply the Simanjuntak model to a diglyme electrolyte with LiPF6 transport properties (SI Table S5:
  transference number and thermodynamic factor of the Simanjuntak LiPF6/EC:DEC set; conductivity 0.988 S/m and
  diffusivity 3.12e-10 m2/s from Horwitz et al., ACS Omega 2018) - close to the "effective electrolyte" approach used
  here for the Nature Materials ether data. Their kinetic
  table uses a different normalisation of the rate constants, and the step-3 equilibrium potential appears to repeat the
  step-2 value (1.9 V; likely a typo), so their parameters need mapping before a direct comparison.
* Moschner et al. cite two SPAN papers by Perez Beltran and Balbuena (J. Phys. Chem. C 125, 13185 (2021); ACS Appl.
  Mater. Interfaces 13, 491 (2021)), for sulfur content and electrolyte compatibility. Their content was not checked
  here (no web access); if they report lithiation voltages, the "DFT bridge" idea needs no new calculations.
* Resistances measured by EIS on cycled, uncompressed electrodes are tens of Ohm cm2 (Moschner, symmetric cells, for
  the uncalendered and 17 % electrodes: Rbulk ~8, Rint1 8-11, pore resistance 26-29 Ohm cm2; Ma after CV:
  charge-transfer arcs ~23 and ~71 Ohm cm2). Fresh or compressed electrodes reach more (Ma before CV ~54 and
  ~136 Ohm cm2; Moschner's 62 % electrode up to 1280 Ohm cm2 after 12 h). The contact resistance fitted to the
  Simanjuntak cell is 330 Ohm cm2 (0.033 Ohm m2), so the fitted Z_CC is an effective lump, not only a contact
  resistance.

## 2. Pilot: calibrate on one or two rates, predict the others

Model and mapping as for the SPAN500 calibration (Simanjuntak geometry and electrolyte; the paper's capacity basis
mapped to model sulfur by an effective factor w_S; fitted: w_S, U0_1..3, b_1..3, k0_1, Z_CC). Fits on the coarse
grid (10/5 volumes; <= 0.2 mV rms from the reported 20/10 grid). rms errors in mV against the published curves:

Se0.06SPAN, carbonate (Chen 2019), rates 0.4 / 1 / 2 / 4 / 6 / 10 A/g (c = calibration rate):

| calibration | 0.4 | 1 | 2 | 4 | 6 | 10 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4 only | 10 c | 70 | 145 | 259 | 401 | 433 |
| 0.4 + 2 | 20 c | 29 | 20 c | 81 | 92 | 147 |
| 0.4 + 10 | 24 c | 39 | 50 | 48 | 35 | 30 c |
| all six (model limit) | 28 c | 35 c | 38 c | 42 c | 26 c | 34 c |

Se0.06SPAN, ether (Chen 2019):

| calibration | 0.4 | 1 | 2 | 4 | 6 | 10 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4 only | 7 c | 58 | 101 | 212 | 303 | 305 |
| 0.4 + 2 | 11 c | 25 | 11 c | 37 | 70 | 137 |
| 0.4 + 10 | 24 c | 38 | 33 | 41 | 34 | 25 c |
| all six (model limit) | 26 c | 24 c | 15 c | 25 c | 25 c | 32 c |

SPAN aerogel and SPAN, carbonate (Ma 2026), rates 0.1 (first cycle) / 0.2 / 0.3 / 0.5 / 1 / 2 C:

| material, calibration | 0.1 | 0.2 | 0.3 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- |
| SPAN-A, 0.2 only | 138 | 8 c | 93 | 181 | 223 | 209 |
| SPAN-A, 0.2 + 1 | 138 | 56 c | 94 | 115 | 73 c | 227 |
| SPAN, 0.2 only | 49 | 8 c | 100 | 175 | 216 | 198 |
| SPAN, 0.2 + 1 | 76 | 49 c | 95 | 135 | 49 c | 166 |

Findings:
1. **One slow curve never predicts the faster ones** (all four datasets: 58-433 mV), although it is fitted to 7-10 mV.
   This confirms Phase 1 on independent data.
2. **Where the model is adequate (Chen), two rates predict what lies between them, not beyond.** Calibrated on the
   slowest and the fastest rate (a 25x range), the model predicts the four intermediate rates within 33-50 mV,
   1.3-1.8x its calibration misfit (27 / 24 mV) and close to the best the model can do on all six rates at once
   (34 / 25 mV rms; the all-rates fits stopped at the evaluation limit, near convergence). Calibrated on
   0.4 + 2 A/g, it interpolates 1 A/g (25-29 mV) but misses 4-10 A/g by 37-147 mV. Reason (report/phase2/figures/l3_split.png): two rates fix the total polarization between them, not its
   split into a logarithmic (kinetic) and a linear (ohmic) part. The 103-137 mV drop from 0.4 to 2 A/g extrapolates
   to 205-274 mV at 10 A/g if purely kinetic, to 615-822 mV if purely ohmic; measured: 489-631 mV. A log + linear
   description of all six rates (36/31 mV per e-fold + 40/56 mV per A/g for carbonate/ether) reproduces the voltage at
   400 mAh/g with rms residuals of 10/3 mV (max 18/5 mV).
3. **Where the model's transport is wrong for the cell (Ma), the second rate reveals it; one rate hides it.** One
   0.2 C curve is fitted to 8 mV, but the two-rate calibration cannot fit 0.2 C and 1 C together (49-73 mV). In these
   cells a higher rate mainly shrinks the capacity (digitized: SPAN-A 1248 -> 686 mAh/g at 1 C; SPAN 793 -> 321;
   paper: 1262 -> 692 and 812 -> 322) while the voltage drops by 0.23-0.43 V, whereas the model with the Simanjuntak
   transport keeps most of its capacity (570-1140 mAh/g at 1 C). A forward check with the 0.2 + 1 C parameters
   (results/lispan/literature_rates/transport_check_ma2026_SPAN-A.json) shows that a much slower transport time scale
   produces exactly this signature: salt diffusivity x0.1 / x0.03 cuts the 1 C capacity from 1037 to 899 / 486 mAh/g
   at an almost unchanged voltage (1.64 / 1.62 V at 300 mAh/g, vs 1.64 V) and leaves 0.2 C unchanged; the measured
   686 mAh/g lies between the two. So these cells behave as transport-limited on a time scale ~10-30x slower than
   the Simanjuntak cell - either in the electrolyte of a thicker,
   more tortuous electrode (thickness and porosity are not reported) or inside the SPAN particles (not in the model;
   Ma et al. credit the aerogel's porous 3D framework, with faster Li+ diffusion from CV and GITT and lower
   charge-transfer resistance). A C/10 curve carries no information on this time scale. Fits with the salt
   diffusivity free were attempted but stopped: salt-depleted solves are too stiff for the current solver (NOTE.txt).
4. **Electrolyte comparison (Chen, same material).** At 0.4 A/g the ether cell is ~30 mV higher at 400 mAh/g and
   gives ~10 % more capacity; at high rate it polarizes faster (56 vs 40 mV per A/g). Both are described by the same
   model with different effective parameters (w_S 1.19 vs 1.12 in the 0.4 + 2 A/g fits; 1.19-1.23 vs 1.12-1.19 across
   calibrations).

Consequences for the proposal: the minimal protocol is "the slowest rate plus the fastest rate of interest", not
"C/10 + 1 C" (a calibrated model should not be extrapolated beyond its fastest rate); a third rate is needed to split
kinetic and ohmic losses; the second rate doubles as a test of whether the model's transport is right for the cell;
and the solver needs to handle salt-depleted (transport-limited) cases before transport parameters can be fitted.
