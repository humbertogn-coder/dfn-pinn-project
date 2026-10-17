# Phase 2: published multi-rate SPAN data - assessment and pilot

October 2026. Five files supplied by the owner were assessed for the predictive test of the identifiability proposal
(calibrate on one or two discharge rates, predict the others); the Supporting Information of Hao et al. followed on
2026-10-16 (section 1b) and added a plain-SPAN rate test. Curves extracted: `data/literature_span/` (README there).
Pilot fits: `scripts/lispan_lit_rates.py`, results `results/lispan/literature_rates/`, figures `report/phase2/`.

## 1. What each paper offers

| paper | material, cell | multi-rate voltage curves | other useful content | verdict |
| --- | --- | --- | --- | --- |
| Chen et al., Nat. Commun. 10, 1021 (2019) | Se0.06SPAN (Se-doped), ~1 mg/cm2, carbonate (1 M LiPF6 EC/DMC) AND ether (LiTFSI DOL/DME + LiNO3) | Fig. 3b,d: 0.4, 1, 2, 4, 6, 10 A/g in both electrolytes (+ first cycle at 0.2 A/g); vector graphics, extracted exactly | same material in two electrolytes; first-cycle curves; DFT Li-migration barriers (1.33 vs 0.39 eV) | **use** (best data quality; Se doping makes parameters effective) |
| Ma et al., Small 22, e74169 (2026) | SPAN and SPAN aerogel, ~1.9 mg S/cm2, carbonate (1 M LiPF6 EC/DMC/EMC) | Fig. 6c,d: 0.1, 0.2, 0.3, 0.5, 1, 2 C for both materials (first cycle at each rate); digitized | EIS of both materials (Fig. 4f,g: charge-transfer arcs ~20 (SPAN-A) vs ~63 (SPAN) Ohm after CV, ~48 vs ~120 Ohm before, on 1.13 cm2), CV vs scan rate, GITT diffusion coefficients, Tafel plots | **use** (closest chemistry to the model: plain SPAN in carbonate) |
| Moschner et al., Adv. Sci. 12, 2415436 (2025) | commercial SPAN (ADEKA SAM-1, 38 wt% S), 3.5 mAh/cm2, 102 um, porosity 59 -> 49 % by calendering, ether (LiTFSI/LiNO3 in diglyme) | only C/50 and C/10 (44 % compacted electrode) and C/10 for the others; rate tests shown as capacities only (SI Fig. S8) | written with the authors of the Li-SPAN model (Danner, Simanjuntak, Latz); full model parameter tables for this cell (SI Tables S4-S6); EIS + electronic conductivity measurements; simulations showing that a thin, electronically poor layer at the current collector shifts the voltage with little change in capacity (a contact-resistance signature), while an ionically poor layer cuts capacity | **not for the rate test, but the most useful for interpretation**; its raw C-rate curves (Fig. S8) would be the ideal Phase 2 dataset (known geometry, porosity, conductivity, same model family) - ask the authors |
| Hao et al., ACS Nano 19, 25385 (2025) + SI | FeS2/SPAN composite and plain SPAN (53 wt% S), 1.5-2 mg/cm2 active, carbonate (1 M LiPF6 EC/DEC/FEC) | main text Fig. 3d: FeS2/SPAN only; **SI Fig. S9: plain SPAN at 0.1, 0.2, 0.5, 1, 2 C** (digitized); SI Fig. S12b: plain SPAN at 0.5 C, cycles 1/2/10/100 | SI: GITT of both materials (Fig. S11), EIS before/after cycling (Tables S2-S3), CV vs scan rate, high-loading SPAN (Fig. S18) | main text alone: low priority (FeS2 conversion is not in the model); **with the SI: use** (plain SPAN in an electrolyte close to the model's, plus EIS for an independent resistance check and one-rate cycling curves; section 1b) |

Side notes:
* Moschner et al. apply the Simanjuntak model to a diglyme electrolyte with LiPF6 transport properties (SI Table S5:
  transference number and thermodynamic factor of the Simanjuntak LiPF6/EC:DEC set; conductivity 0.988 S/m and
  diffusivity 3.12e-10 m2/s from Horwitz et al., ACS Omega 2018) - close to the "effective electrolyte" approach used
  here for the Nature Materials ether data. Their kinetic
  table uses a different normalisation of the rate constants, and the step-3 equilibrium potential appears to repeat the
  step-2 value (1.9 V; likely a typo), so their parameters need mapping before a direct comparison.
* Moschner et al. cite two SPAN papers by Perez Beltran and Balbuena (J. Phys. Chem. C 125, 13185 (2021); ACS Appl.
  Mater. Interfaces 13, 491 (2021)), for sulfur content and electrolyte compatibility. The Simanjuntak model paper
  (refs. 48-49) cites the first by title, "Sulfurized polyacrylonitrile (SPAN): changes in mechanical properties during
  electrochemical lithiation" (doi 10.1021/acs.jpcc.1c02966), together with a third one, "A solid electrolyte
  interphase to protect the sulfurized polyacrylonitrile (SPAN) composite for Li-S batteries: computational approach
  addressing the electrolyte/SPAN interfacial reactivity" (J. Mater. Chem. A 9, 7888 (2021), doi 10.1039/D1TA00110H),
  among the DFT studies of SPAN. Owner's assessment (2026-10-16): these are theoretical studies (mechanical
  properties, interfacial reactivity) of the owner's group and give no input for this work; not used. Balbuena is
  also a co-author of the Nature Materials SPAN-structure paper (Wang et al., 2026, doi 10.1038/s41563-026-02484-y).
* Resistances measured by EIS on cycled, uncompressed electrodes are tens of Ohm cm2 (Moschner, symmetric cells, for
  the uncalendered and 17 % electrodes: Rbulk ~8, Rint1 8-11, pore resistance 26-29 Ohm cm2; Ma after CV:
  charge-transfer arcs ~23 and ~71 Ohm cm2). Fresh or compressed electrodes reach more (Ma before CV ~54 and
  ~136 Ohm cm2; Moschner's 62 % electrode up to 1280 Ohm cm2 after 12 h). The contact resistance fitted to the
  Simanjuntak cell is 330 Ohm cm2 (0.033 Ohm m2), so the fitted Z_CC is an effective lump, not only a contact
  resistance.

## 1b. Hao et al. (2025), Supporting Information (supplied 2026-10-16)

25 figures and 3 tables; all figures are rasters (~200 ppi). What matters here:

| item | content | use |
| --- | --- | --- |
| Fig. S9 | plain SPAN, discharge-charge at 0.1, 0.2, 0.5, 1, 2 C (one profile per rate) | **rate test, used** (section 2; digitized end capacities 745 / 695 / 654 / 623 / 592 mAh/g vs 743.89 / 692.95 / 652.11 / 622.04 / 591.53 in the text, +0.1-0.3 %) |
| Fig. S12b | plain SPAN, cycles 1, 2, 10, 100 at 0.5 C (cycle 1 is probably the 0.1 C activation cycle: cycle 2 ends at 724 mAh/g, the paper's "initial" 0.5 C capacity 720.83) | **one-rate aging check, used** (finding 7) |
| Tables S2-S3 | EIS of SPAN: Rs 2.20 / Rct 271.7 Ohm before cycling; Rs 2.78 / Rct 7.70 / Rint 7.09 Ohm after 10 cycles at 0.2 C | **independent resistance check, used** (finding 5) |
| Fig. S11 | GITT (30 min pulses at 0.05 C, 2 h rests) of SPAN and FeS2/SPAN, discharge and charge, with apparent D | read semi-quantitatively (`scripts/analyze_hao2025_gitt.py`, ~3 mAh/g and ~10 mV per pixel): pulse size 18.0 mAh/g (SPAN) and 20.0 (FeS2/SPAN), so 1 C = 0.72 / 0.80 A/g, close to each material's 0.1 C capacity (744 / 781 mAh/g); the voltage recovers by 30-100 mV (mean 65) in a discharge rest, but rested charge and rested discharge voltages stay 160-180 mV apart at 300-500 mAh/g: most of the charge/discharge gap is not removed by 2 h of rest (path-dependent, consistent with the model's irreversible steps 2-3); apparent D 10^-14.7 to 10^-12.5 cm2/s (median 10^-13.6), which depends on an assumed diffusion length and cannot be used as a model input as is |
| Table S1, Fig. S1 | 53.11 wt% S in this SPAN (made at 300 C from PAN:S 1:8); its XRD shows sharp peaks of crystalline sulfur | some unbound sulfur is present; a reason to leave out the first cycle (cycle 1 in Fig. S12b: 913 mAh/g with a lower plateau) |
| Fig. S18 | SPAN at 5.4 mg/cm2 (caption: sulfur loading; Methods: cathode load 5.4-6.5 mg/cm2), 0.2 C, compaction 1.1 g/cm3: ~2.3 mAh/cm2 in cycles 2-3 (~430-450 mAh/g; ~480 in cycles 4-10) vs 693 at 1.5-2 mg/cm2 | a loading (thickness) effect at constant C-rate: a possible later test of the model's transport, but the thickness of the low-loading electrode is not reported |
| Figs. S10, S13, S15, S25 | CV vs scan rate (b-values 0.82-0.85, "pseudocapacitive" share 87-93 %); dQ/dV at cycles 10/50/100; SEM after 100 cycles (cracked SPAN electrode); TEM of the CEI after 20 cycles (thicker, uneven on SPAN; ~10 nm, uniform on FeS2/SPAN) | qualitative only (the model is galvanostatic and has no capacitive storage) |
| Figs. S2-S8, S14, S16-S17, S19-S24 | DFT of Li2Sx on FeS2, morphology, FeS2/SPAN and SPAN-CNT data, polysulfide adsorption in ether, in-situ XRD, XPS | not relevant here |

1 C is not defined in the paper; the fits use 1 C = 0.71 A/g SPAN (from an earlier, cruder reading of the GITT pulse
size; 0.72 from the final reading, 0.74 if 1 C is the 0.1 C capacity; a 2-5 % difference, negligible next to the
misfits below).

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

Plain SPAN, carbonate (Hao 2025, SI Fig. S9), rates 0.1 / 0.2 / 0.5 / 1 / 2 C (1 C = 0.71 A/g SPAN, section 1b):

| calibration | 0.1 | 0.2 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
| 0.1 only | 9 c | 70 | 130 | 246 | 470 |
| 0.1 + 0.5 | 19 c | 43 | 30 c | 49 | 74 |
| 0.1 + 2 | 35 c | 70 | 46 | 31 | 36 c |
| 0.2 + 2 | 41 | 47 c | 41 | 39 | 45 c |
| all five (model limit) | 27 c | 47 c | 32 c | 36 c | 50 c |

Findings:
1. **One slow curve never predicts the faster ones** (all five datasets: 58-470 mV), although it is fitted to 7-10 mV.
   This confirms Phase 1 on independent data from three groups.
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
   **The plain SPAN of Hao shows the same signature, milder.** It loses 20 % of its capacity from 0.1 to 2 C (745 ->
   592 mAh/g), the calibrated model ~10 % (744 -> 667 in the 0.1 + 2 C fit), so the two-rate calibrations that include
   2 C do not get below 35 mV, nor the all-rates calibration below 39 mV, and the largest deviations sit mostly in the last ~20 % of the discharge (up to
   168-232 mV at 2 C). Forward check
   with the 0.1 + 2 C parameters (transport_check_hao2025_SPAN.json): salt diffusivity x0.3 / x0.1 gives 664 / 549
   mAh/g at 2 C (measured 592) at an almost unchanged voltage (1.750 / 1.740 V at 300 mAh/g, vs 1.755; x0.03 failed
   in the solver), i.e. a time scale ~3-10x slower. But even x0.1 leaves 0.5 C at 710 mAh/g (measured 654): the
   measured loss is gradual over all rates, while one slower electrolyte time scale only cuts the highest rate. A
   spread of time scales, such as Li transport inside SPAN particles of different sizes (not in the model; both Hao
   and Ma argue from GITT that faster in-particle Li+ diffusion explains their better materials), would fit that
   pattern better (not simulated here). Neither paper reports the electrode thickness, and Hao does not report the
   separator, so the electrolyte cannot be ruled out.
4. **Electrolyte comparison (Chen, same material).** At 0.4 A/g the ether cell is ~30 mV higher at 400 mAh/g and
   gives ~10 % more capacity; at high rate it polarizes faster (56 vs 40 mV per A/g). Both are described by the same
   model with different effective parameters (w_S 1.19 vs 1.12 in the 0.4 + 2 A/g fits; 1.19-1.23 vs 1.12-1.19 across
   calibrations).
5. **An independent check: the resistance found by each calibration vs the cell's EIS (Hao).** The fitted contact
   resistance converts to a real-cell equivalent R = Z_CC m_S,model / (w_S L) (m_S,model = 7.67 g/m2, loading L =
   1.5-2 mg/cm2). EIS after 10 cycles gives Rs + Rct + Rint = 17.6 Ohm, i.e. 19.9 Ohm cm2 for a 12 mm disc (disc size
   not reported; before cycling 310 Ohm cm2). Calibrated on 0.1 C only: 165-220 Ohm cm2 (8-11x the cycled-cell EIS);
   on 0.1 + 0.5 C: 0.1-0.2 Ohm cm2 (at the lower bound: the ohmic part is lost, which is why 2 C is then
   mispredicted); on 0.1 + 2 C: 14-19; on 0.2 + 2 C: 31-41; on all five: 29-39 Ohm cm2. Only calibrations that include
   the fastest rate land within ~2x of the measured value (report/phase2/figures/l4_hao.png b). Caveats: EIS Rct is a
   linearized charge-transfer resistance, so the EIS total is not purely ohmic, and the fitted Z_CC is an effective
   lump; the agreement is in order of magnitude, not a parameter validation.
6. **The cell changes during the first cycles, by as much as one rate step (Hao).** At 0.5 C the voltage rises by
   ~45-50 mV from cycle 2 to cycle 10 (SI Fig. S12b; cycle 2 coincides with cycle 100 below ~370 mAh/g, cycle 10 lies
   higher), and Rct falls from 272 Ohm before cycling to 7.7 Ohm after 10 cycles. In the rate test the 0.1 C curve
   (the first step in the rate order given in the main text) coincides with 0.2 C within the line width (~18 mV)
   up to 525 mAh/g, ~30 mV below the trend of the other four rates if that trend continues below 0.2 C (log + linear
   fit to the voltage at 300 mAh/g, 0.2-2 C: 42 mV per e-fold + 35 mV per C, rms 3 mV).
   Steps between neighbouring rates are 39-69 mV, so a rate test run on a cell that is still activating mixes
   activation into the "rate dependence". The same issue appeared for SPAN500 (cycle 2 vs 3: 47 mV) and for Ma
   (0.1 C = first cycle). In Hao the 0.2 + 2 C calibration is only marginally better than 0.1 + 2 C (rms over the
   five rates 43 vs 46 mV): finding 3 dominates.
7. **Exploratory: aging seen at one rate only (Hao, SI Fig. S12b; scripts/lispan_lit_aging_check.py).** Starting
   from the all-rates calibration, each cycle at 0.5 C was fitted with two parameters under two hypotheses: active
   sulfur loss plus a larger contact/ohmic resistance, or active sulfur loss plus slower kinetics of steps 2-3 (a
   common shift of U0_2 and U0_3, equivalent to a change of k0). At cycle 100 both fit equally (23 / 24 mV; w_S
   0.656 / 0.659, i.e. ~8-9 % less active sulfur than cycle 10; Z_CC 0.0102 Ohm m2, or a 35 mV shift = k0 x0.5 relative
   to the base) but predict 2 C curves 68 mV apart (1.635 vs 1.703 V at 300 mAh/g). At cycle 2 (activation) the two
   readings predict 2 C 150 mV apart (fits 16 / 25 mV). At cycle 10 they agree within 13 mV (smaller changes from
   the base: Z_CC -22 % or a +20 mV kinetic shift). So one-rate cycling data cannot say whether a SPAN cell ages by
   resistance growth or by slower kinetics; a periodic check-up at a second rate should be able to (implied by the
   68-150 mV split against 16-36 mV misfits, not demonstrated). Caveats: the base comes from a different cell (the rate test), and the
   one-rate fits are 16-36 mV.

Consequences for the proposal: the minimal protocol is "the slowest rate plus the fastest rate of interest", not
"C/10 + 1 C" (a calibrated model should not be extrapolated beyond its fastest rate); a third rate is needed to split
kinetic and ohmic losses; the second rate doubles as a test of whether the model's transport is right for the cell;
and the solver needs to handle salt-depleted (transport-limited) cases before transport parameters can be fitted.
Added by the Hao data: (i) the rate test should be run on a conditioned cell (Hao's 0.5 C curves were still changing
between cycles 2 and 10; when it stabilizes is not shown), with the slow rate repeated at the end to check for drift
(Hao did: 685 vs 744 mAh/g at 0.1 C, -8 %); (ii) an EIS spectrum is a cheap independent check of the fitted resistance;
(iii) for aging studies (the follow-on), one-rate cycling data are ambiguous and a periodic two-rate check-up is
needed; (iv) both plain-SPAN rate tests from other labs (Ma, Hao) lose capacity with rate faster than the model, which
points to a missing transport process, plausibly inside the SPAN particles; a model extension with in-particle
transport (one time constant or a distribution of them; not simulated here) would be the next model step, and the
second rate is what would identify it.
