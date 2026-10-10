// Chapters 5-7: analysis, conclusions, the paper
module.exports = (h) => {
  const { P, H1, H2, UL, OL, ANALOGY, KEY, CAUTION, TABLE } = h;
  return [
    H1("5. Analysis and discussion"),
    H2("5.1 What the results mean together"),
    P("Read one after the other, the results tell a single story with three levels of error, each roughly an order of magnitude smaller than the one above:"),
    ...TABLE(["Level", "Size", "What it is", "Where"], [
      ["model vs experiment", "8-20 mV (fits), about 30 mV (predictions)", "the physics of the model is simpler than the cell", "4.6"],
      ["modeling choices", "0.01-5 mV (12 mV max)", "closures, double layer, ramp, constant properties", "4.3"],
      ["PINN vs model", "0.14-0.44 mV", "the network's numerical error", "4.2"],
    ], [2100, 2400, 3300, 1560], "The hierarchy of errors in this work."),
    P("This hierarchy is the central argument of the paper. Because the PINN error is about an order of magnitude below the typical modeling choice, and the modeling choices are well below the model-experiment misfit, **the PINN does not limit anything we can say about the cell**. Every conclusion about real data is a conclusion about the model, and every improvement must come from the physics (or from better data), not from a better network."),
    ANALOGY("a microscope that is better than the sample", ["If your microscope resolves details ten times finer than the scratches on your slide, buying a better microscope will not help you see the cell; cleaning the slide will. Our PINN is the microscope, the model is the slide."]),
    H2("5.2 Strengths"),
    ...UL([
      "**Accuracy and reproducibility from scratch.** Six independent trainings, all below 0.5 mV, with the same recipe at two C-rates whose physics is quite different (at 1 C the electrolyte matters, at 0.1 C it does not).",
      "**Design rules that transfer.** Building the reaction ordering, the current and the kinetics into the network, and adding global inventories, worked for both the DFN and the Li-SPAN model; each fix is tied to a specific, diagnosed failure (table of section 3.2), which makes the paper useful to others.",
      "**An honest yardstick for inverse problems.** Comparing with the ideal estimator on the same data, and predicting the difference from the forward error, is stronger than the usual 'we recovered the true parameters within x %'. It separates what the noise allows from what the method loses.",
      "**Identifiability first.** The Fisher analysis explained in advance why one C-rate is not enough and which parameters must be measured independently; the SPAN500 case then showed exactly that behavior on real data.",
      "**Traceability.** Every number is generated from saved files with fingerprints; this already caught one invalid run and three wrong notes.",
    ]),
    H2("5.3 Weaknesses and how the paper deals with them"),
    ...UL([
      "**The PINN is slower than the FV solver** (about 1000x for a forward solve). We state it plainly and argue the value of the method elsewhere: uniform treatment of forward and inverse problems, a differentiable model with gradients for free, and a path to settings where meshes are expensive (not tested here).",
      "**The real-data cases are limited.** The Simanjuntak points are digitized; the SPAN500 data are one C-rate, a different electrolyte and a different chain structure. We present them as demonstrations with explicit limits rather than as calibrations.",
      "**The benchmark is our reading of the published model.** The closures are documented and quantified; a confirmation from the original authors would remove the doubt (optional step, chapter 9).",
      "**One seed has a 3.5 mV spike** at a phase transition. It is reported, located and explained (abrupt change of the open-circuit slope); the rms error of that run is still 0.44 mV.",
    ]),
    H2("5.4 Where a PINN is worth its cost"),
    P("For a single forward simulation of a one-dimensional cell, never. The PINN becomes interesting when (i) the same object must solve the model, absorb data and return gradients with respect to parameters, as in our inverse problems; (ii) many evaluations of a fixed model are needed after a one-time training (60 ms per discharge curve once trained); or (iii) the geometry or coupling makes mesh-based solvers expensive (two- or three-dimensional electrodes, parametric networks that take the parameters as inputs). Points (ii) and (iii) are outlook, not results of this work."),
    KEY("the honest message", ["A carefully designed PINN can solve a multi-reaction conversion-cathode model to sub-millivolt accuracy and invert it as well as an ideal estimator; on a 1-D cell it is a methodological tool, not a speed-up; and for Li-SPAN the bottleneck is the model and the data, not the solver."]),

    H1("6. Conclusions"),
    ...OL([
      "The Li-SPAN model of Simanjuntak et al. reproduces its published curves only with documented closures (faradaic current of step 1, effective rate constants in the Tafel regime, step 3 irreversible during discharge); with them our FV reference matches the published curves to 16 / 18 mV rms.",
      "A PINN with the reaction ordering, the current balance and the kinetics built in, plus a global charge-inventory residual, reproduces the reference to 0.26 mV (0.1 C) and 0.18 mV (1 C) rms, mean of three seeds trained from scratch.",
      "One discharge rate cannot separate kinetics from contact resistance and open-circuit parameters; 0.1 C and 1 C together determine eight parameters to about 1 % or better; transport and Li_{2}S parameters are practically invisible in discharge curves.",
      "With a Levenberg-Marquardt step on the parameters, the multi-rate inverse PINN reaches the accuracy of an ideal least-squares estimator up to a bias of at most 2 Cramer-Rao bounds, which is predicted by the PINN's own forward error.",
      "On measured data the misfit (8-20 mV) is a model-form error; for SPAN500 it is smaller than the change between consecutive cycles, and the kinetics cannot be calibrated from one C-rate.",
      "For this 1-D model the PINN costs about a thousand times more CPU time than the FV solver; its value is methodological.",
    ]),

    H1("7. The paper: message, structure and status"),
    H2("7.1 What is new (to be confirmed by the literature review)"),
    P("To our knowledge (this must be checked against the literature before submission, chapter 9): (i) no PINN has been published for a SPAN or other multi-reaction conversion-cathode continuum model; (ii) the published Li-SPAN model has not been independently re-implemented with its ambiguities documented; (iii) inverse PINNs for battery models are usually judged against the true parameters, not against an ideal estimator with a predictive bias model."),
    H2("7.2 Structure and status of the draft"),
    P("The draft (`paper/manuscript/manuscript.md`) has all sections: introduction; model and closures (the closure table, Table {tabclos} of this document, still has to be added); FV reference (Fig. 1); PINN formulation; identifiability and inverse method; results on forward accuracy (Figs. 2-3, Table 2), identifiability (Fig. 4), synthetic inverse (Fig. 5, Table 3), measured data (Fig. 6, Table 5) and cost (Table 4); conclusions; supplementary assumption checks (Fig. S1, Table S1). It has about 3400 words (roughly 7-8 journal pages with the figures), is written in Markdown with LaTeX equations for easy conversion to the JES template, and its 61 placeholders (55 distinct numbers) are all filled automatically from the saved results. The reference list is still incomplete."),
    CAUTION("decisions the advisor should make", ["(1) Is the framing 'methods paper with an honest cost statement' acceptable, or should the paper emphasize the identifiability and inverse results more and the PINN less? (2) Is the SPAN500 case worth including with its limits, or should it move to the supplementary material until multi-rate data exist? (3) Authorship and the role of the DFN results (main text, supplementary, or a separate paper)."]),
  ];
};
