// Chapter 4: results
module.exports = (h) => {
  const { P, H1, H2, H3, UL, EQ, FIGURE, ANALOGY, KEY, CAUTION, SLIDE, TABLE } = h;
  return [
    H1("4. Results"),
    P("All numbers in this chapter are produced by `scripts/paper_build.py` from saved results (`paper/numbers.json`), and every figure is a figure of the draft manuscript. Errors are given as **root-mean-square (rms)** differences over the discharge, from 100 s after the current is switched on to 98 % of the discharge, and as the **maximum** difference over the same window:"),
    EQ("rms"),

    // ------------------------------------------------------------- 4.1
    H2("4.1 The reference model reproduces the published curves"),
    ...FIGURE("fig1_reference_vs_paper.png", "(a) Our FV solver (lines) and the curves of Simanjuntak et al. Fig. 6a (circles) at 0.1 C and 1 C without contact resistance. (b) The FV solver at the four C-rates of their Fig. 4b, with their experimental points (circles).", 6.0),
    P("With the effective rate constants and the closures of Table {tabclos}, the FV solver reproduces the published model curves to **16.0 mV rms at 0.1 C and 18.4 mV rms at 1 C** (largest deviations 32 and 45 mV). The delivered capacities at 0.05, 0.1, 0.2 and 1 C (1254, 1240, 1212 and 1025 mAh g_{S}^{-1}) agree with the paper within 3.1 %, and the evolution of the four sulfur species and of Li_{2}S follows their Fig. 5a (not shown here). The remaining difference is expected: the paper uses concentration-dependent electrolyte properties that it does not give, and we digitized the curves from the figure."),
    KEY("4.1 in one line", ["We have a trustworthy numerical copy of the published model, with every interpretation documented."]),

    // ------------------------------------------------------------- 4.2
    H2("4.2 The PINN solves the model to a fraction of a millivolt"),
    ...FIGURE("fig2_forward_voltage.png", "Top: cell voltage from the FV solver (thick grey) and the PINN (seed 0, dashed red) at 0.1 C and 1 C. Bottom: PINN minus FV for the three seeds. The grey band marks the first 100 s (30 s current ramp and its transient), excluded from the error statistics.", 5.6),
    ...TABLE(["C-rate", "Seed", "V rms [mV]", "V max [mV]", "SPAN species rms [mol m^{-3}]", "Electrolyte rms / max [mol m^{-3}]"], [
      ["0.1 C", "0", "0.164", "0.74", "0.12-0.56", "0.04 / 0.10"],
      ["0.1 C", "1", "0.435", "3.54", "0.38-0.94", "0.05 / 0.11"],
      ["0.1 C", "2", "0.177", "0.81", "0.12-0.57", "0.04 / 0.12"],
      ["1 C", "0", "0.225", "0.77", "0.19-0.51", "1.51 / 7.8"],
      ["1 C", "1", "0.137", "0.63", "0.16-0.47", "0.87 / 7.0"],
      ["1 C", "2", "0.179", "0.70", "0.15-0.45", "1.14 / 7.0"],
      ["**mean**", "", "**0.26 (0.1 C), 0.18 (1 C)**", "", "", ""],
    ], [1100, 800, 1700, 1500, 2100, 2160], "Forward accuracy of the final recipe, trained from scratch, three seeds per C-rate (PINN vs FV).", "tabfwd"),
    P("All six runs reproduce the FV voltage to **better than 0.5 mV rms** (0.14-0.44 mV). The evaluated voltage window spans 1.16 V at 0.1 C and 0.80 V at 1 C, so this is an error of about 0.01-0.04 % of the voltage range, well below the millivolt-level noise of a typical battery cycler. One run (0.1 C, seed 1) has a narrow spike of 3.5 mV exactly where the third plateau starts (around 915 mAh g^{-1}), where the open-circuit potential changes slope abruptly; without that spike its error is 0.35 mV rms. The other five runs stay below 0.81 mV everywhere."),
    P("The internal states are also right (Figure {fig3}): the four sulfur species agree within 1 mol m^{-3} out of 598, the Li_{2}S volume fraction within 10^{-5}, and the electrolyte concentration within 0.05 mol m^{-3} at 0.1 C and 0.9-1.5 mol m^{-3} rms at 1 C (out of 1000; the 1 C electrolyte is the hardest field, with a maximum error of about 7-8 mol m^{-3})."),
    ...FIGURE("fig3_forward_species.png", "Cathode-averaged concentrations of the four sulfur species (left axis) and Li_{2}S volume fraction (grey, right axis): FV (solid, light) and PINN seed 0 (dashed).", 5.8),
    P("At 0.1 C the error first falls below 1 mV after 12 500 of the 30 000 steps (seed 0), and all three seeds stay below 1 mV from step 17 500 on; at 1 C it falls below 1 mV after 15 000-17 500 steps. Earlier results (0.21 / 0.12 mV) came from networks that continued previous runs; the numbers above are from random initialization and are the ones the paper will report."),
    ANALOGY("three students, one exam", ["Training the same network three times with different random starting points is like giving the same exam to three students with the same preparation. If all three pass with similar grades, the method (not luck) is responsible. Here all six passed (below 0.5 mV); one student made a single slip on one question (the 3.5 mV spike)."]),
    SLIDE(["Figure {fig2} plus the mean row of Table {tabfwd}: 'PINN vs reference: 0.26 mV (0.1 C) and 0.18 mV (1 C), three seeds, from scratch'."]),

    // ------------------------------------------------------------- 4.3
    H2("4.3 How much do our modeling choices matter?"),
    P("The PINN is compared with the FV solution of exactly the same equations. That benchmark differs from the FV reference of section 4.1 in a few numerical choices (step 2 irreversible, no double-layer charging, a 30 s current ramp), and the reference itself relies on assumptions inherited from the paper (constant electrolyte properties, 25 °C). We quantified every one by changing it alone and recomputing the discharge (Table {tabass}; the curves are Figure S1 of the draft)."),
    ...TABLE(["Choice changed", "0.1 C: rms / max [mV]", "1 C: rms / max [mV]"], [
      ["step 2 reversible (as in the FV reference)", "2.66 / 5.63", "0.12 / 0.30"],
      ["all three steps reversible (literal printed model)", "52.7 / 190", "not computed (solver too stiff)"],
      ["double layer as in the paper (c_{DL} = 0.1 F m^{-2})", "4.58 / 11.7 (+6 mAh g^{-1})", "3.37 / 9.9 (+5 mAh g^{-1})"],
      ["1 s instead of 30 s current ramp", "0.01 / 0.20", "0.06 / 0.95"],
      ["coarser grid (20 / 10 volumes)", "0.01 / 0.01", "0.08 / 0.09"],
      ["contact resistance 0.035 instead of 0.025 Ω m^{2}", "10.0 (constant shift)", "100 (constant shift)"],
      ["S^{2-} reference concentrations × 10", "≤ 0.01", "≤ 0.01"],
      ["electrolyte conductivity -10 %", "0.32 / 0.33", "3.04 / 3.19"],
      ["salt diffusivity -20 %", "0.12 / 0.19", "0.96 / 1.66"],
      ["temperature +10 K (no Arrhenius)", "0.78 / 2.91", "2.93 / 3.49"],
    ], [4560, 2400, 2400], "Effect of each modeling choice on the voltage (vs the PINN benchmark).", "tabass"),
    P("The only benchmark simplification above 3 mV rms is the **double layer** (4.6 / 3.4 mV rms): with the large SPAN surface area the electrode stores about 160 C m^{-2} capacitively, adding about 6 mAh g^{-1} and shifting the plateau transitions by up to 12 mV. Treating step 2 as irreversible changes the voltage by 2.7 mV rms, but up to 5.6 mV at the plateau transitions. The choices span 0.01-5 mV rms: most are larger than the PINN error (0.14-0.44 mV), a few (grid, ramp, S^{2-} references) are smaller, and all are well below the model-experiment misfit (18 mV), except two that are not benchmark simplifications: the contact resistance quoted in the paper's text (a constant shift of 10 / 100 mV) and the literal printed model. The PINN accuracy is therefore a statement about the equations, not about the cell. Closure B3 is essential: the literal model, with the backward terms of steps 2 and 3 switched on, differs by 53 mV rms."),

    // ------------------------------------------------------------- 4.4
    H2("4.4 What a discharge curve can reveal"),
    ...FIGURE("fig4_identifiability.png", "Cramer-Rao lower bounds (smallest achievable standard error, 1 mV noise) of the eight parameters of the inverse problem, from 0.1 C alone, 1 C alone and both together. Logarithmic scale; the line marks 1 %.", 4.8),
    P("Figure {fig4} answers the question 'what can we hope to measure?'. Three findings decide the design of the inverse problem:"),
    ...UL([
      "**With one C-rate, the kinetics, the first equilibrium potential and the contact resistance cannot be separated.** At 0.1 C the bound of k_{0,2} is 144 % and that of Z_{CC} 296 %: they all shift the voltage almost uniformly. Even at 0.1 C alone the open-circuit slopes b_{m} are well determined (below 0.5 %).",
      "**Two C-rates solve it.** The ohmic drop grows linearly with the current, the kinetic overpotential with its logarithm, so 0.1 C and 1 C together bound all eight parameters to **1.04 % or better** (Z_{CC} 0.09 %, U_{0,1} 0.44 mV).",
      "**Some parameters are invisible.** The electrolyte conductivity, transference number and solid conductivity act through an ohmic-like drop that looks like Z_{CC} (correlation 0.94; bounds 15-72 % even with four C-rates), and the Li_{2}S kinetics move the voltage by less than 0.1 mV per e-fold change. They must be taken from independent measurements.",
    ]),
    P("A fourth, exact result explains why U_{0,2} and U_{0,3} are not in the list: for a step that only runs forward (Tafel regime), the rate depends on the product of the rate constant and an exponential of the equilibrium potential, so multiplying k_{0} by 1000 is *exactly* the same as raising U_{0} by 0.355 V:"),
    EQ("tafel"),
    KEY("4.4 in one line", ["One discharge rate tells you the open-circuit shape; you need at least two rates to measure the kinetics, and some properties no discharge curve will ever reveal."]),

    // ------------------------------------------------------------- 4.5
    H2("4.5 The inverse PINN is as good as an ideal estimator"),
    P("The test: synthetic discharge curves at 0.1 C and 1 C (FV solver, true parameters = nominal, 1 mV random noise, 112 and 90 points), and a deliberately bad starting guess: k_{0} × 2 / 0.5 / 1.5, b × 1.1 / 0.9 / 1.1, U_{0,1} + 20 mV and Z_{CC} × 1.5 (84 mV initial misfit)."),
    ...FIGURE("fig5_inverse_synthetic.png", "(a) Parameter errors during training with Adam only; (b) with the Levenberg-Marquardt step every 250 steps; (c) final errors of the PINN (blue) and of the ideal estimator on the same data (orange) with its Cramer-Rao bound, and the PINN result predicted from its forward error (crosses).", 6.5),
    P("With Adam alone the parameters crawl along the correlated directions and are still 9.8 % off in k_{0,1} after 7000 steps (Figure {fig5}a). With the LM step the first parameter update brings the misfit from 84 mV to 2.6 mV, and after 750 steps all parameters are within 1-2 % (Figure {fig5}b). Table {tabinv} compares the final PINN estimates with the ideal estimator."),
    ...TABLE(["Parameter", "Initial", "PINN + LM", "Ideal (FV) estimator", "CRLB", "PINN - ideal: actual / predicted"], [
      ["k_{0,1} [%]", "+100", "+1.96", "+1.58", "1.02", "+0.37 / +0.60"],
      ["k_{0,2} [%]", "-50", "-0.31", "+1.10", "0.71", "-1.41 / -1.08"],
      ["k_{0,3} [%]", "+50", "+0.31", "+0.07", "0.72", "+0.23 / +0.18"],
      ["b_{1} [%]", "+10", "-0.14", "+0.03", "0.16", "-0.17 / -0.15"],
      ["b_{2} [%]", "-10", "+0.05", "+0.30", "0.20", "-0.24 / -0.22"],
      ["b_{3} [%]", "+10", "+0.06", "+0.02", "0.11", "+0.05 / +0.01"],
      ["U_{0,1} [mV]", "+20", "-0.99", "-0.69", "0.43", "-0.31 / -0.27"],
      ["Z_{CC} [%]", "+50", "+0.09", "+0.04", "0.09", "+0.06 / +0.09"],
    ], [1500, 1000, 1300, 1800, 1000, 2760], "Eight-parameter inverse from 0.1 C + 1 C synthetic data: errors with respect to the true values.", "tabinv"),
    P("Even the ideal estimator misses the truth (k_{0,1} by 1.6 %) because of the noise, so it is the right yardstick. The CRLB column is computed for the actual data points of this test (for example 1.02 % for k_{0,1}); Figure {fig4} uses the 100-point design of the identifiability study (1.04 %). The PINN differs from the ideal estimator by at most **2.0 Cramer-Rao bounds** (k_{0,2}; median 0.7), and the difference is **predictable**: the PINN voltage at its final parameters differs from the FV voltage by 0.15 mV (0.1 C) and 0.71 mV (1 C), and propagating this forward error through the sensitivities reproduces the PINN-minus-ideal differences parameter by parameter (last column). The inverse PINN is the ideal estimator plus the bias of its own forward error, mostly from the 1 C electrolyte."),
    ANALOGY("a slightly bent ruler", ["If your ruler is bent by a known small amount, every measurement you make with it is off by a predictable amount. The PINN is that ruler: its tiny voltage error (0.15-0.7 mV) bends the parameter estimates by a predictable amount (up to 1.4 % in k_{0,2}). Knowing the bend, you can correct for it or straighten the ruler."]),

    // ------------------------------------------------------------- 4.6
    H2("4.6 Measured data"),
    ...FIGURE("fig6_measured.png", "(a) Simanjuntak et al. cell: experimental points (circles), FV least-squares fit (solid) and PINN inverse (dashed, FV model run at the PINN's parameters), fitted at 0.1 + 1 C, predicted at 0.05 and 0.2 C. (b) SPAN500 cell (Wang et al. 2026): C/10 cycle 3 (thick grey) and cycle 2 (dotted), fits with two kinetic hypotheses (solid) and their 1 C predictions (dotted), and the PINN inverse (dashed).", 6.2),
    H3("Simanjuntak et al. cell (Fig. 4b points)"),
    P("Fitting the same eight parameters to the digitized experimental points at 0.1 C and 1 C reduces the misfit from 50 / 46 mV (nominal model) to **19.8 / 16.1 mV** (18.4 mV overall) and improves the predictions at the two rates not used in the fit from 31 / 61 mV to 29.8 / 31.1 mV (Table {tabmeas}). The contact resistance comes out as **0.033 Ω m**^{2}, close to the 0.035 that the authors themselves quote as their best value. The other parameters are loosely determined (different starts give different parameters with the same misfit), because the remaining 18 mV is a *model-form* error, not noise. The PINN inverse lands in the same valley: the FV model at its parameters misses the data by 22.7 / 15.8 mV (20.2 mV overall, 1.8 mV above the FV optimum), with Z_{CC} = 0.034 Ω m^{2}."),
    H3("SPAN500 (Nature Materials 2026)"),
    P("These are our own group's data, but they were not measured for modeling: one C-rate (C/10), a different electrolyte (ether instead of carbonate), a coin cell, and no reported sulfur content or electrode thickness. We kept the Simanjuntak geometry and transport (at about 1 A m^{-2} they contribute less than 1 mV), took the series resistance from the high-frequency end of the published impedance spectrum, and fitted the effective sulfur fraction together with the open-circuit parameters U_{0,1-3} and b_{1-3}."),
    ...UL([
      "The third discharge is reproduced to **10.0 mV rms** (nominal model: 80 mV), with an effective sulfur fraction of **0.38 g S per g SPAN**, close to the 40-50 wt% the paper gives for typical SPAN.",
      "But the same cell changes by **54 mV rms between cycles 2 and 3**, and the fitted model misses cycle 2 by 55 mV. The calibration describes one cycle, not the material.",
      "With all three rate constants raised a thousandfold the fit is as good (11.3 mV; the fit compensates by lowering U_{0,2} and U_{0,3} by 0.35 and 0.43 V, as the equation of section 4.4 predicts), but the two calibrations **predict 1 C curves that differ by up to 135 mV** in the first plateau. One C-rate cannot fix the kinetics, exactly as section 4.4 predicted.",
      "The PINN inverse on the same data (same sulfur fraction, data up to 413 mAh g^{-1} so that the window stops before the end-of-discharge collapse) reaches parameters where the FV model misses the data by **8.9 mV**, against **8.2 mV** for the FV optimum on the same window.",
    ]),
    ...TABLE(["Data", "Model / method", "Misfit [mV rms]", "Notes"], [
      ["Simanjuntak 0.1 C / 1 C (fitted)", "FV least squares", "19.8 / 16.1", "Z_{CC} = 0.033 Ω m^{2}"],
      ["Simanjuntak 0.05 C / 0.2 C (predicted)", "FV least squares", "29.8 / 31.1", "nominal model 31 / 61"],
      ["Simanjuntak 0.1 C / 1 C (fitted)", "PINN inverse (FV at its parameters)", "22.7 / 15.8", "Z_{CC} = 0.034 Ω m^{2}"],
      ["SPAN500 cycle 3, C/10", "FV, paper kinetics", "10.0", "w_{S} = 0.376; cycle 2: 55 mV"],
      ["SPAN500 cycle 3, C/10", "FV, kinetics × 1000", "11.3", "1 C prediction differs by up to 135 mV"],
      ["SPAN500 cycle 3, Q ≤ 413 mAh g^{-1}", "PINN inverse vs FV optimum", "8.9 vs 8.2", "same valley"],
    ], [2900, 2600, 1500, 2360], "Fits to measured data.", "tabmeas"),
    CAUTION("what the real-data results mean", ["On real data the limit is the model, not the PINN: 8-20 mV of misfit is the price of the model's simplifications (and, for SPAN500, of using a carbonate-electrolyte model with S_{4} chains for a short-chain SPAN in an ether electrolyte). The fitted parameters are therefore *effective* values. The real-data cases demonstrate the workflow and its limits; they are not a calibration of SPAN500."]),

    // ------------------------------------------------------------- 4.7
    H2("4.7 Computing cost"),
    ...FIGURE("figB_cost.png", "Computing time on one CPU core (log scale). FV: finite-volume solver. PINN: training of the final recipe (CPU time from measured time per step) and, for the inverse, the logged time of the inverse runs plus the forward training they start from.", 5.6),
    P("An FV discharge takes **4-7 s**; training the PINN takes **1.9 h** of CPU time, about a thousand times more. Once trained, the PINN returns a whole discharge curve in about 60 ms, but only for the parameters and current it was trained for. For the eight-parameter inverse, the FV least-squares fit needs 68 discharges per C-rate (**14 min**), the PINN route about 1.9 h on top of about 5 h of forward training. For this one-dimensional model the PINN is therefore **not** a faster solver, and the paper will say so. Its value lies elsewhere (chapter 5)."),
    SLIDE(["Figure {figB} with one sentence: 'For a 1-D cell a stiff solver takes seconds; the PINN is a different tool, not a faster one'."]),
  ];
};
