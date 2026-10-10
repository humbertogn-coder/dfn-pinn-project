// Chapter 4: using the tool to learn about the chemistry
module.exports = (h, N) => {
  const { P, H1, H2, UL, FIGURE, HOWTO, SHOWS, CAUTION } = h;
  const c = N.chem || {};
  const mv = (x) => Math.round(1000 * x);
  const pct = (x) => Math.round(100 * x);
  const out = [
    H1("4. Using the tool: what it tells us about the chemistry"),
    P("Once the model has been calibrated by the inverse PINN, it shows things that are hard to measure directly: what happens inside the electrode and which process costs the voltage. Every result in this chapter is limited by the assumptions of the model (Figure {figa3}); where possible we compare it with what the two papers report."),

    H2("4.1 What limits the Simanjuntak cell at higher rate?"),
    ...FIGURE("c1_rates.png", "Simanjuntak cell, model calibrated by the inverse PINN on the measured points of their Fig. 4b (0.1 C and 1 C).", 5.7),
    ...HOWTO(["(a) Lines: the calibrated model at four rates; circles: the measured cell (data from Simanjuntak et al. 2024, Fig. 4b). (b) Voltage lost at 1 C compared with a very slow discharge (0.05 C), split by removing one loss at a time from the model. (c) Share of the sulfur chains still in the PAN-S_{2}Li state when the cell reaches its 1.0 V cut-off."]),
    ...SHOWS([
      `**Most of the 1 C loss is not chemistry.** At 1 C the cell loses about ${N.rate_loss_total} mV compared with 0.05 C. About ${N.rate_loss_zcc} mV of it is the contact resistance between the electrode and the current collector (the PINN found Z_{CC} = ${N.sim_Zcc_pinn} Ω m^{2}; the paper quotes 0.035 as its best agreement in the text and uses 0.025 in its table), ${N.rate_loss_kin} mV is the speed of the SPAN reactions and only ${N.rate_loss_el} mV is ion transport in the electrolyte. (The three parts add up to more than the total because they overlap.)`,
      `**The capacity lost at 1 C is the third reduction step.** The capacity falls from ${N["rate_Q_0.05"]} to ${N.rate_Q_1} mAh per g of sulfur. At the cut-off, ${N["rate_left_1.0"]} % of the chains are still PAN-S_{2}Li: the cell runs out of voltage before step 3 (PAN-S_{2}Li → PAN-SLi + Li_{2}S) can finish. Without the contact resistance the model would deliver ${N.rate_Q_1_noZ} mAh/g; with 100 times faster reactions ${N.rate_Q_1_fastk}; with a perfect electrolyte only ${N.rate_Q_1_el}.`,
      `**The electrode works evenly.** Li_{2}S forms almost uniformly across the electrode (${N.rate_li2s_spread_1} % difference between front and back at 1 C) and the salt concentration changes by at most ${N.rate_salt_dev_1} %.`,
    ]),
    P(`**What the literature says.** Simanjuntak et al., changing parameters by hand, concluded that the large overpotential at high current comes from the contact resistance and not from the SPAN kinetics, and that ion transport is not limiting in their highly porous electrodes. Our calibration agrees that the contact resistance dominates and transport is minor, but gives the reactions a visible share; the unfinished third step is new, more chemical information. These numbers come from about 30 digitized points and a model that misses them by about 18 mV, so the contact resistance is firmly determined and the other shares are trends. For this electrode design, a better contact with the collector would help at 1 C more than faster SPAN chemistry. (The PINN's own fields agree with our simulator at these parameters within ${N.rate_pinn_V_1}-${N.rate_pinn_V_01} mV and ${N.rate_pinn_species_max} mol/m^{3}.)`),
  ];

  // ---------------------------------------------------------------- 4.2 first cycle
  const stepNames = ["step 1", "step 2", "step 3"];
  const newSets = ["s500_c1", "s500_c2", "si300_c2", "si320_c2", "si500_c2"].filter((k) => c[k] && c[k].tool === "PINN");
  if (newSets.length) {
    const gap = Math.max(...newSets.map((k) => c[k].rms_window_pinn - c[k].rms_window_fv));
    const dU = Math.ceil(Math.max(...newSets.map((k) => Math.max(...c[k].U0.map((u, i) => Math.abs(1000 * (u - c[k].U0_fv_window[i])))))));
    const lo = Math.min(...newSets.map((k) => c[k].rms_window_pinn)), hi = Math.max(...newSets.map((k) => c[k].rms_window_pinn));
    out.push(P(`For sections 4.2 and 4.3 the inverse PINN was run on ${newSets.length} more measured curves with the same method as in section 3.6 (allowing larger parameter steps, because these curves need larger shifts). On every one it reached a misfit within ${gap.toFixed(1)} mV of the best possible fit with our simulator (${lo.toFixed(1)}-${hi.toFixed(1)} mV), with step potentials within ${dU} mV of it. As in section 3.6, the effective sulfur content is fitted with our simulator and the PINN finds the step potentials and slopes.`));
  }
  const largest = (d) => d.indexOf(Math.max(...d));
  if (c.s500_c1 && c.s500_c2 && c.s500_c3) {
    const d12 = c.s500_c1.U0.map((u, i) => mv(c.s500_c2.U0[i] - u));
    const d23 = c.s500_c2.U0.map((u, i) => mv(c.s500_c3.U0[i] - u));
    out.push(
      H2("4.2 SPAN500: what changes in the first cycle?"),
      P(`SPAN loses about a fifth of its capacity in the first cycle (Nature Materials: about 20 %, explained by lithium that replaces residual protons at the edges of the SPAN domains and stays trapped). For SPAN500 the measured capacity falls from ${N.cap_500_c1} (cycle 1) to ${N.cap_500_c2} and ${N.cap_500_c3} mAh/g (cycles 2 and 3), and the first-cycle Coulombic efficiency is ${N.ce_500_c1} %. We calibrated the model separately on each of the three discharges.`),
      ...FIGURE("c2_first_cycle.png", "SPAN500, cycles 1-3 at C/10 (data from Wang et al., Nature Materials 2026, source data of Fig. 4a), each calibrated separately.", 5.7),
      ...HOWTO(["(a) Thick = measured discharge of each cycle; dashed = calibrated model. (b) Equilibrium potential of each reduction step found for each cycle (bars), with the values of the Simanjuntak model for a carbonate electrolyte (black marks). (c) Effective sulfur content the model needs to reproduce the capacity of each cycle."]),
      ...SHOWS([
        `**The first discharge happens at a much lower voltage.** The model places steps 1, 2 and 3 ${d12[0]}, ${d12[1]} and ${d12[2]} mV lower in cycle 1 than in cycle 2; the largest shift is in ${stepNames[largest(d12)]}, at the beginning of the discharge. From cycle 2 to cycle 3 steps 1 and 2 move up by only ${d23[0]} and ${d23[1]} mV, while step 3 keeps rising (${d23[2]} mV): after the first discharge the material is activated, and its end of discharge is still settling.`,
        `**The first discharge needs more 'sulfur'.** The effective sulfur content goes from ${pct(c.s500_c1.w_S)} wt% (cycle 1) to ${pct(c.s500_c2.w_S)} and ${pct(c.s500_c3.w_S)} wt%. The model has no term for lithium that is trapped, so it reads the extra first-cycle capacity as extra active sulfur. This number follows the capacity, so it is a consistency check rather than a new measurement: the ${pct(1 - c.s500_c2.w_S / c.s500_c1.w_S)} % drop is close to the ${100 - Number(N.ce_500_c1)} % first-cycle loss implied by the Coulombic efficiency (${N.ce_500_c1} %), and of the same order as the ~20 % of sulfur in S=C environments that the Nature Materials paper estimates from the SPAN structure.`,
      ]),
      ...CAUTION("what the model can and cannot say here", ["The model describes the first cycle with *effective* values (lower potentials and more 'sulfur'). It measures how large the activation is and in which part of the discharge it happens, but the mechanism comes from the structural measurements of the Nature Materials paper."]),
    );
  }

  // ---------------------------------------------------------------- 4.3 synthesis temperature
  if (c.si300_c2 && c.si320_c2 && c.si500_c2) {
    const d = c.si300_c2.U0.map((u, i) => mv(u - c.si500_c2.U0[i]));
    const dsrc = c.s500_c2 ? Math.max(...c.si500_c2.U0.map((u, i) => Math.abs(mv(u - c.s500_c2.U0[i])))) : null;
    out.push(
      H2("4.3 SPAN300, SPAN320, SPAN500: what does the synthesis temperature change?"),
      P("The Nature Materials paper shows that a higher synthesis temperature shortens the sulfur chains in SPAN, and that the discharge plateau and the capacity become lower (long chains such as S_{8} or S_{6}^{2-} reduce near 2.4 V, short chains such as S_{4}^{2-} near 2.1 V). We calibrated the model on the second-cycle discharge of each material."),
      ...FIGURE("c3_synthesis.png", "SPAN made at 300, 320 and 500 °C, second discharge at C/10 (data from Wang et al., Nature Materials 2026, digitized from Supplementary Fig. 21).", 5.7),
      ...HOWTO(["(a) Thick = measured; dashed = calibrated model. (b) Equilibrium potential of each step for each material, with the Simanjuntak values (black marks). (c) Effective sulfur content."]),
      ...SHOWS([
        `**From SPAN300 to SPAN500 all three steps move down.** The steps of SPAN500 sit ${d[0]}, ${d[1]} and ${d[2]} mV below those of SPAN300; the largest shift is in ${stepNames[largest(d)]}${largest(d) === 2 ? ", the end of the discharge" : ""}. This puts numbers, step by step, on the lower plateau that the paper links to shorter chains.`,
        `**Less sulfur takes part.** The effective sulfur content is ${pct(c.si300_c2.w_S)} wt% (SPAN300), ${pct(c.si320_c2.w_S)} wt% (SPAN320) and ${pct(c.si500_c2.w_S)} wt% (SPAN500). The model thus separates two effects that the paper discusses together ("shorter sulfur chains and/or lower sulfur content"): *where* the steps happen (the potentials) and *how much* sulfur takes part (the effective content). It cannot tell whether the missing sulfur is absent or present but inactive.`,
      ]),
      ...CAUTION("digitized curves, one rate, effective parameters", [`The three curves were read from a figure (about 6 mV and 2 mAh/g per pixel).${dsrc !== null ? ` As a digitization check, the SPAN500 curve of this figure and the source-data curve of section 4.2 give step potentials within ${dsrc} mV.` : ""} All three materials are fitted with the same S_{4}-chain model, so the parameters are effective values that describe the differences between materials, not the real chain lengths.`]),
    );
  }

  return out;
};
