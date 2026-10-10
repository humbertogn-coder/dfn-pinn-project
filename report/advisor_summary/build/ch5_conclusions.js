// Chapter 5: analysis and conclusions, limitations, outlook
module.exports = (h, N) => {
  const { P, H1, H2, UL, OL, KEY, CAUTION, TABLE } = h;
  const c = N.chem || {};
  const d12 = c.s500_c1 && c.s500_c2 ? c.s500_c1.U0.map((u, i) => Math.round(1000 * (c.s500_c2.U0[i] - u))) : null;
  const d35 = c.si300_c2 && c.si500_c2 ? c.si300_c2.U0.map((u, i) => Math.round(1000 * (u - c.si500_c2.U0[i]))) : null;
  return [
    H1("5. Analysis and conclusions"),
    H2("5.1 The chain of evidence"),
    P(`Each step was checked against something trusted before the next one used it: the DFN PINN against PyBaMM (${N.dfn_fwd_2}-${N.dfn_fwd_1} mV), our Li-SPAN model against the published curves (${N.ref_rms_01} / ${N.ref_rms_1} mV), the Li-SPAN PINN against our model (${N["fwd_0.1_mean"]} / ${N.fwd_1_mean} mV), and the inverse PINN against known synthetic answers (within 2 %), against the authors' estimate on their measured cell (Z_{CC} ${N.sim_Zcc_pinn} vs 0.025-0.035 Ω m^{2}) and against the best possible fit on independent SPAN500 data (${N.s5_win_pinn} vs ${N.s5_win_fv} mV). The errors form a clear hierarchy: PINN vs model about **0.2 mV**, the numerical simplifications of our benchmark **0.01-5 mV** (manuscript), model vs real cells **8-20 mV**. The PINN is never the limiting part; what we can say about a real cell is limited by the model and the data.`),

    H2("5.2 What we learned about the chemistry"),
    ...UL([
      `**Simanjuntak cell at 1 C:** the voltage loss is mostly contact resistance (${N.rate_loss_zcc} of ${N.rate_loss_total} mV) and the capacity loss is an unfinished third reduction step; Li_{2}S forms evenly. The main conclusion agrees with the original paper; the split between the losses is a trend.`,
      ...(d12 ? [`**SPAN500, first cycle:** the first discharge happens ${Math.min(...d12)}-${Math.max(...d12)} mV lower than the second (most at its beginning), and its extra capacity appears as extra 'sulfur', close to the measured first-cycle loss.`] : []),
      ...(d35 ? [`**SPAN300 to SPAN500:** all three reduction steps move down by ${Math.min(...d35)}-${Math.max(...d35)} mV and the effective sulfur content falls from ${Math.round(100 * c.si300_c2.w_S)} to ${Math.round(100 * c.si500_c2.w_S)} wt%, separating the chain-length and sulfur-content effects.`] : []),
      "**Experiments:** one C/10 curve fixes the equilibrium properties but not the reaction speed; one more discharge rate of the same cell is the most valuable next measurement.",
    ]),

    H2("5.3 Limitations"),
    ...UL([
      "**The model:** identical S_{4} chains, a carbonate electrolyte with constant properties, no polysulfide shuttle, no first-cycle chemistry, constant temperature, discharge only. Parameters fitted to SPAN500 or to other electrolytes are *effective* values.",
      "**The data:** some curves were read from published figures; the Nature Materials cells were measured at one rate, and their sulfur content and electrode thickness are not reported; the LG M50 curves are simulated.",
      "**The method:** training a PINN takes about two hours, while our classic solver computes one Li-SPAN discharge in a few seconds (see the box below).",
    ]),
    ...KEY("why a PINN if the classic solver is faster?", [
      "For a one-dimensional cell the classic solver is faster, both for one simulation and for a parameter fit (about 140 simulations of a few seconds), and the manuscript will say so. The PINN is worth it because one tool gives the solution, the fit to data and every field inside the cell; because it suits cases where each classical simulation is expensive (2-D/3-D electrodes, microstructures) or where the unknown is a whole function; and because a *parametric* PINN (parameters as inputs) is trained once and then fits new data in seconds. This work shows that the method is accurate and reliable on a new, multi-reaction chemistry; the last two points are the next steps.",
    ]),
    H2("5.4 Next steps"),
    P("(1) Write the manuscript for the Journal of The Electrochemical Society, with the mathematical details, error metrics and checks of every step there and in the supplementary information. (2) Obtain discharge curves of one SPAN cell at two or more rates, with its sulfur content and electrode thickness, to calibrate the SPAN kinetics (for SPAN500, a single 1 C discharge would decide between the two assumptions of section 3.6). (3) Later: fit measured LG M50 curves, and build a parametric PINN for fast fitting of many cells."),
  ];
};
