// Chapter 3: results (simple: what is plotted, what it shows, why the small differences)
module.exports = (h, N) => {
  const { P, H1, H2, UL, FIGURE, HOWTO, SHOWS, TABLE } = h;
  const T = N.inverse_table;
  const meaning = {
    k0_1: ["k_{0,1}", "how fast step 1 runs", "mol m^{-2} s^{-1}"],
    k0_2: ["k_{0,2}", "how fast step 2 runs", "mol m^{-2} s^{-1}"],
    k0_3: ["k_{0,3}", "how fast step 3 runs", "mol m^{-2} s^{-1}"],
    b_1: ["b_{1}", "how much the step-1 voltage drops as the step progresses", "V"],
    b_2: ["b_{2}", "same for step 2", "V"],
    b_3: ["b_{3}", "same for step 3", "V"],
    U0_1: ["U_{0,1}", "voltage at which step 1 starts", "V"],
    Z_CC: ["Z_{CC}", "contact resistance electrode / collector", "Ω m^{2}"],
  };
  const fmt = (k, v) => (k.startsWith("k0") ? (v * 1e8).toFixed(3) + "×10^{-8}" : k === "Z_CC" ? v.toFixed(4) : v.toFixed(4));
  const err = (k, e) => (k === "U0_1" ? `${e >= 0 ? "+" : ""}${e.toFixed(1)} mV` : `${e >= 0 ? "+" : ""}${e.toFixed(2)} %`);
  const rows = T.map((r) => [meaning[r.param][0], meaning[r.param][1], meaning[r.param][2], fmt(r.param, r.true),
    fmt(r.param, r.start), fmt(r.param, r.adam), fmt(r.param, r.lm), err(r.param, r.err_lm),
    r.param === "U0_1" ? `${r.noise_limit.toFixed(1)} mV` : `${r.noise_limit.toFixed(2)} %`]);
  return [
    H1("3. Results"),
    P("Each figure is explained in three parts: what is plotted, what it shows, and why the small differences exist. Unless stated otherwise, **error** means the difference between the PINN and the reference simulator, averaged over the discharge (root mean square), in millivolts. For comparison, a good battery tester measures the voltage with a noise of about 1 mV."),

    H2("3.1 Classic DFN: the PINN reproduces the reference simulator"),
    ...FIGURE("r1_dfn_vs_pybamm.png", "LG M50 cell (parameters of Chen et al. 2020), 1C discharge (5 A). Reference: PyBaMM.", 5.4),
    ...HOWTO([`(a) Cell voltage during the first 50 minutes of the discharge: thick grey = PyBaMM, dashed blue = PINN. (b) Error (PINN minus PyBaMM) for two trainings started from different random numbers.`]),
    ...SHOWS([`The two curves lie on top of each other: the error is ${N.dfn_fwd_1} and ${N.dfn_fwd_2} mV on average, about a quarter of the noise of a battery tester. It is largest in the first seconds and the last minute, where the voltage changes fastest.`]),

    H2("3.2 Classic DFN: parameters and aging from one discharge"),
    ...FIGURE("r2_dfn_inverse_aging.png", "Inverse problem with synthetic LG M50 data. (a) New cell: three parameters from one noisy 1C discharge. (b) Cell after 200 cycles (PyBaMM degradation models): how much lithium and active material are left in each electrode.", 5.4),
    ...HOWTO(["Each bar is the error of one parameter found by the PINN, in percent of its true value (zero = perfect)."]),
    ...SHOWS([`The errors are at most about 0.6 % for the new cell and 1.6 % for the aged cell. The PINN reads not only material parameters but also how much lithium and active material are left in each electrode of an aged cell, from a single discharge curve. The two largest errors of (b) (lithium in graphite ${N.dfn_age_theta_n0} %, active graphite ${N.dfn_age_eps_am_n} %) belong to a pair that changes the voltage in almost the same way; a slow discharge should separate them.`]),

    H2("3.3 Our Li-SPAN model reproduces the published curves"),
    P("No code of the Simanjuntak model is public, so we wrote our own simulator (a finite-volume solver). Before using it as the reference for the PINN we compared it with the paper."),
    ...FIGURE("r3_model_vs_paper.png", "(a) Circles: simulated curves from Simanjuntak et al. 2024 (their Fig. 6a); lines: our model. (b) Circles: their measured cell (their Fig. 4b) at four rates; lines: our model.", 5.4),
    ...HOWTO(["x: capacity delivered per gram of sulfur; y: cell voltage. Circles are read (digitized) from the figures of the paper; lines are our simulator."]),
    ...SHOWS([`(a) Our model differs from theirs by ${N.ref_rms_01} mV (0.1 C) and ${N.ref_rms_1} mV (1 C) on average. (b) With the parameters printed in the paper, the model follows the trend of the measured cell (higher rate, lower voltage and capacity) but misses the points by ${N.ref_vs_exp_min}-${N.ref_vs_exp_max} mV: it is below them at the slow rates and above them at 1 C, where the measured cell loses more voltage (the paper attributes this to the contact resistance). Fitting the parameters to these points (inverse PINN, chapter 4) brings the misfit to ${N.sim_pinn_rms_1}-${N.sim_pinn_rms_01} mV.`,
      "Why the differences: the curves were read from a printed figure (a few mV), and the paper does not give some inputs (for example how the electrolyte properties change with concentration). We had to choose these in eight places, and every choice is documented."]),

    H2("3.4 The Li-SPAN PINN reproduces the model"),
    ...FIGURE("r4_pinn_forward.png", "Li-SPAN PINN against our model at 0.1 C and 1 C (three trainings each, all started from random numbers).", 5.4),
    ...HOWTO(["Top: cell voltage; thick grey = our model, dashed red = PINN. Bottom: error for each of the three trainings."]),
    ...SHOWS([`The error is ${N["fwd_0.1_1"]}, ${N["fwd_0.1_2"]} and ${N["fwd_0.1_3"]} mV at 0.1 C and ${N.fwd_1_1}, ${N.fwd_1_2} and ${N.fwd_1_3} mV at 1 C (averages ${N["fwd_0.1_mean"]} and ${N.fwd_1_mean} mV). It stays roughly constant along the whole discharge, below 1 mV almost everywhere. Three trainings with the same recipe give a similar quality, so the result is reproducible and not luck.`,
      "Why the small bumps: they sit where one reduction step ends and the next begins and the curve bends sharply; networks find sharp bends hardest. One training has a narrow 3.5 mV spike at the start of the third step (about 915 mAh/g)."]),
    ...FIGURE("r5_species.png", "Concentration of the four sulfur states during the discharge (average over the electrode): thick = our model, dashed = PINN.", 5.4),
    ...SHOWS(["The PINN gets the chemistry inside the electrode right, not only the voltage: the four sulfur states agree within about 1 mol/m^{3} (out of 598). The curves show the mechanism of Figure {figa2}: PAN-S_{4} is used up first, PAN-S_{3}Li rises and falls, PAN-S_{2}Li rises and falls, and PAN-SLi (the end product) grows. At 1 C the green PAN-S_{2}Li curve does not return to zero: the discharge stops before step 3 is finished (chapter 4)."]),

    H2("3.5 Inverse problem: which parameters are easy to find"),
    P("Test: discharge curves computed with our model at 0.1 C and 1 C, plus 1 mV of random noise, and a starting guess far from the truth (rate constants ×2, ×0.5, ×1.5, etc.). Two ways of training: **Adam only** (the standard training, 7000 steps) and **Adam + LM** (the same plus a Newton-type correction of the parameters, Levenberg-Marquardt, every 250 steps)."),
    ...TABLE(["Parameter", "What it is", "Unit", "True", "Start", "Adam only", "Adam + LM", "Error (Adam + LM)", "Noise limit"],
      rows, [700, 2300, 950, 950, 950, 950, 950, 950, 800],
      "Eight parameters found from two synthetic discharge curves (0.1 C and 1 C, 1 mV noise). Noise limit: the typical error (one standard deviation) that this amount of noise causes even for the best possible method; a single test can land closer or further.", "tabinv"),
    ...SHOWS(["**Easiest:** the voltage slopes b and the contact resistance Z_{CC} (errors 0.05-0.14 % and 0.09 %). **Hardest:** the reaction rates k_{0} (0.3-2 %), because a slower reaction and a larger contact resistance lower the voltage in a similar way; only the combination of a slow and a fast discharge separates them. With Adam alone the rate of step 1 was still 10 % off after 7000 steps; with the LM correction every parameter is within 2 % of the truth and within 2.3 noise limits."]),

    H2("3.6 Independent real data: SPAN500 (Wang et al., Nature Materials 2026)"),
    P("These are published data from a paper our group co-authored: a SPAN made at 500 °C, in a coin cell with an **ether** electrolyte (the Simanjuntak model was built for a **carbonate** electrolyte), discharged at C/10, cycles 1-3. The sulfur content and the electrode thickness are not reported, so we first fit an *effective sulfur content* with our simulator (it sets the current of the cell per gram of sulfur); the PINN then finds the voltages and slopes of the three steps."),
    ...FIGURE("r6_span500.png", "SPAN500 at C/10 (data from Wang et al., Nature Materials 2026, source data of Fig. 4a).", 5.5),
    ...HOWTO(["(a) x: capacity per gram of SPAN (as in the paper). Thick grey = measured, cycle 3; dotted = cycle 2 of the same cell; blue = our model fitted to cycle 3 with our simulator; dashed orange = our model run with the parameters found by the PINN.",
      "(b) Thin lines: two C/10 fits that assume slow reactions (the Simanjuntak rates) or 1000 times faster reactions; thick lines: what each fit predicts for a 1 C discharge."]),
    ...SHOWS([`(a) The model reproduces cycle 3 to ${N.s5_rms_paper} mV on average. With the PINN's parameters the error is ${N.s5_win_pinn} mV, against ${N.s5_win_fv} mV for the best possible parameters on the same part of the curve: the PINN finds practically the best answer. The same cell changes by ${N.s5_c2c} mV between cycles 2 and 3, so the remaining error is smaller than the cell's own variation. The effective sulfur content is ${Math.round(100 * N.s5_wS)} wt% (the paper gives 40-50 wt% for typical SPAN).`,
      `(b) Both assumptions fit C/10 equally well, but their 1 C predictions differ by up to ${N.s5_1C_diff} mV. One discharge rate cannot tell how fast the reactions are; one 1 C discharge of the same cell would decide.`,
      "Why the error: the model's three steps make small kinks (near 200 and 320 mAh/g) that the real cell does not have. The model assumes identical S_{4} chains, while SPAN500 has a mixture of short chains (the Nature Materials finding), and it was built for a different electrolyte. The fitted values are therefore *effective* parameters of this cell."]),
  ];
};
