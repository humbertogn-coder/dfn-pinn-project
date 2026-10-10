// Summary + chapter 1 (background) + chapter 2 (what we did)
module.exports = (h, N) => {
  const { P, H1, H2, UL, OL, FIGURE, KEY, CAUTION, ANALOGY } = h;
  const c = N.chem || {};
  return [
    H1("In short"),
    P("We trained a **physics-informed neural network (PINN)** on the classic lithium-ion battery model and checked it where the answer is known; we then moved it to the **lithium-SPAN** cell model, checked it against that model's own paper, tested it on independent data from another publication, and finally **used it to learn about SPAN chemistry**. The main results:"),
    ...OL([
      `**Classic model (graphite | NMC, LG M50 cell).** The PINN reproduces the reference simulator to ${N.dfn_fwd_2}-${N.dfn_fwd_1} mV and recovers battery parameters and the aging state from one discharge curve within about 0.6 % and 1.6 % (synthetic data).`,
      `**Li-SPAN model.** Our own copy of the published model reproduces the authors' curves (${N.ref_rms_01} / ${N.ref_rms_1} mV), and the PINN reproduces our model to **${N["fwd_0.1_mean"]} mV at 0.1 C and ${N.fwd_1_mean} mV at 1 C** (average of three independent trainings), including the sulfur species inside the electrode.`,
      "**Inverse problem.** From two synthetic discharge curves (0.1 C and 1 C, with noise) the PINN finds eight parameters of the model within 2 % (most within about 0.3 %).",
      `**Independent data (SPAN500, Wang et al., Nature Materials 2026, a paper our group co-authored; different electrolyte).** The model fits the measured curve to ${N.s5_win_pinn}-${N.s5_rms_paper} mV, less than the change of the same cell between two consecutive cycles (${N.s5_c2c} mV).`,
      `**Chemistry.** In the Simanjuntak cell most of the voltage lost at 1 C is contact resistance (${N.rate_loss_zcc} of ${N.rate_loss_total} mV), and the lost capacity is an unfinished third reduction step. For SPAN500 the tool measures the first-cycle activation, and across SPAN300, SPAN320 and SPAN500 how the reduction potentials and the active sulfur change with the synthesis temperature.`,
    ]),
    P("The detailed equations, error metrics and numerical settings of every step will be in the manuscript and its supplementary information; this document explains what each result means."),

    H1("1. Background"),
    H2("1.1 Why lithium-SPAN, and why a model"),
    P("Lithium-sulfur batteries promise more energy per kilogram than lithium-ion batteries, because sulfur is light and can take up two electrons per atom. Their weak point is that, in most electrolytes, the sulfur dissolves as polysulfide molecules that travel to the lithium anode and are lost (the *shuttle*). In **sulfurized polyacrylonitrile (SPAN)** the sulfur is chemically bound to a carbon-nitrogen polymer backbone. In carbonate electrolytes it reacts without dissolving, so the shuttle largely disappears and the cell can cycle for hundreds of cycles. Our group, together with collaborators at Brookhaven National Laboratory and other institutions, studies how the SPAN structure controls this behavior (Wang et al., *Nature Materials* 2026)."),
    P("A **continuum model** of the cell turns the chemistry into equations that predict the voltage during a discharge, the concentration of each species inside the electrode, and how these change with the current. Such a model lets us ask *why* a cell behaves as it does and *which material property* we should change."),

    H2("1.2 The classic battery model (DFN) and the PINN"),
    P("The **Doyle-Fuller-Newman (DFN)** model is the standard physics model of a lithium-ion cell. It treats the cell as three thin layers (negative electrode, separator, positive electrode) and follows five effects at the same time (Figure {figa1})."),
    ...FIGURE("a1_dfn_effects.png", "What the classic DFN model describes (drawing, not to scale). Every effect costs some voltage; the cell voltage is what is left.", 4.4),
    P("A **PINN** is a neural network that is trained to *obey these equations* instead of copying data: at thousands of random points in the cell and in time, the network's answer is put into the equations, and the training reduces how badly they are violated. The same network can be used in two directions. In the **forward problem** the parameters are known and the network predicts the voltage (and everything inside the cell). In the **inverse problem** the voltage is measured and some parameters are unknown: they become extra numbers that the training adjusts until the network obeys the equations *and* matches the data."),
    P("**How we checked it.** With *synthetic data*, i.e. voltage curves computed with **PyBaMM** (the standard open-source battery simulator), where the exact answer is known. The cell is the **LG M50**, a commercial 5 Ah graphite | NMC cell whose parameters were measured by Chen et al. (2020): forward problem, inverse problem (three parameters of a new cell) and aged cells (lithium and active material lost after 200 cycles, from PyBaMM's degradation models)."),
    CAUTION("LG M50: real parameters, simulated curves", ["The LG M50 tests use the parameters measured on the real cell, but all voltage curves (also for the aged cells) were computed with PyBaMM. We have not yet fitted *measured* LG M50 curves; the first measured data in this project are the Li-SPAN ones (chapter 3)."]),

    H2("1.3 The Li-SPAN model of Simanjuntak et al. (2024)"),
    P("The first continuum model of a Li-SPAN cell was published by **Simanjuntak, Danner, Wang, Buchmeiser and Latz (Electrochimica Acta 497, 144571, 2024)**. In it, the sulfur sits in short chains of four atoms (S_{4}) that bridge the polymer backbone. During discharge each chain is reduced in **three steps, one after the other** (Figure {figa2}). Each step adds two electrons per chain and happens at its own voltage (near 2.2, 1.9 and 1.66 V), which is why the model's discharge curve has three sloping segments (measured SPAN curves are smoother). Steps 2 and 3 also release sulfide (S^{2-}), which meets Li^{+} from the electrolyte and precipitates as solid Li_{2}S."),
    ...FIGURE("a2_lispan_mechanism.png", "The reaction mechanism of the Li-SPAN model (drawing). Three steps of two electrons each give the theoretical capacity of 1254 mAh per gram of sulfur.", 5.0),
    P("Figure {figa1b} shows the cell as this model sees it: a lithium-metal foil, a thick glass-fibre separator and a porous SPAN electrode."),
    ...FIGURE("a1b_lispan_cell.png", "The Li-SPAN cell as described by the Simanjuntak model (drawing, not to scale).", 4.4),
    P("Compared with the classic DFN, the Li-SPAN model is simpler in one way (nothing diffuses inside the solid: the sulfur is fixed on the polymer) and more complex in another (three reactions in sequence and a solid product that grows). Figure {figa3} compares the two and lists what the Li-SPAN model leaves out. These are its **limitations**, and they explain most of the differences we see when we compare it with real cells."),
    ...FIGURE("a3_dfn_vs_lispan.png", "What each model includes, and what the Li-SPAN model does not include.", 4.9),

    H1("2. What we did"),
    P("Figure {figa4} shows the seven steps of the project. Blue: the classic model; orange: the Li-SPAN model with the data of its own paper; green: the Li-SPAN model with independent data, and its use to learn about the chemistry."),
    ...FIGURE("a4_workflow.png", "The steps of the project, in order (drawing).", 4.8),
    H2("Why start with the classic DFN, and not directly with Li-SPAN?"),
    P("Because a new tool must first be checked where the answer is known (Figure {figa5}):"),
    ...UL([
      "**A trusted reference exists for the DFN.** PyBaMM solves the DFN model reliably, so when the PINN disagrees, the PINN is wrong. For Li-SPAN no public simulator existed; we had to write our own. Starting there would have meant two unknowns at once: is the PINN wrong, or our implementation of the model?",
      "**We learned how to train this kind of network.** The DFN taught us the design rules: build into the network everything the physics fixes exactly, balance the different equations during training, add global conservation checks, and use two discharge rates for the inverse problem. A first attempt without these rules (September 2026) never got below 23 mV; with them the Li-SPAN PINN reached below 1 mV within days.",
      "**Credibility.** The graphite | NMC DFN is the usual test case for PINNs in battery modeling. Reproducing it is what makes the new result (Li-SPAN) believable.",
    ]),
    ...FIGURE("a5_why_dfn_first.png", "Why the classic model came first (drawing).", 4.6),
  ];
};
