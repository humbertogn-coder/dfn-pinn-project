// Chapter 3: phases of the work and problems solved
module.exports = (h) => {
  const { P, H1, H2, UL, KEY, CAUTION, TABLE } = h;
  return [
    H1("3. Phases of the work"),
    H2("3.1 Timeline"),
    P("The Li-SPAN work started after the DFN part of the project (a PINN for the lithium-ion DFN model of the LG M50 cell, with 0.23-0.25 mV accuracy, three-parameter inverse problems within 0.1-0.6 % and an aging study; see the student guide, chapters 5-6). The DFN part gave us the design rules (build the physics in; balance the residual groups) that the Li-SPAN work reused; the second-order parameter step for correlated parameters was developed during the Li-SPAN work itself."),
    ...TABLE(["Phase", "When", "What was produced", "Main outcome"], [
      ["1 Reference model", "Oct 5-6", "FV solver, digitized paper curves, closures B1-B8", "16 / 18 mV rms from the paper at 0.1 / 1 C; capacities within 3.1 %"],
      ["2 Forward PINN", "Oct 6-7", "PINN with built-in structure; charge inventory; separator scaling", "first 24 mV, then 1 mV, then 0.2 mV (0.1 C) and 0.12 mV (1 C), warm-started"],
      ["3 Identifiability", "Oct 7", "sensitivities of 17 parameters at 4 C-rates", "8 parameters identifiable with 0.1 C + 1 C; others invisible or confounded"],
      ["4 Inverse (synthetic)", "Oct 7", "multi-rate inverse PINN with LM step", "8 parameters within 2.3 standard errors of the truth"],
      ["5 Measured data", "Oct 7", "fits to Simanjuntak Fig. 4b (FV and PINN)", "18 mV model-form error; contact resistance 0.033 Ω m^{2} (paper 0.035)"],
      ["6 Paper steps 1-5", "Oct 8-9", "3 seeds from scratch; cost table; traceable build; assumption checks; SPAN500 case", "0.26 / 0.18 mV mean; this document"],
    ], [1700, 900, 3400, 3360], "Phases of the Li-SPAN work (October 2026)."),
    H2("3.2 Problems found and how they were solved"),
    P("Much of the value of the project is in the problems it ran into, because each one is a lesson for anybody who builds a PINN for a battery model. We list them honestly, including our own mistakes; the negative results are part of what the paper will report."),
    ...TABLE(["Problem", "Symptom", "Fix", "Effect"], [
      ["step 2 reversible is stiff for a PINN", "backward rate 20x the demand from a tiny representation error", "PINN benchmark treats step 2 as irreversible (quantified)", "2.7 mV change at 0.1 C, 0.1 mV at 1 C"],
      ["single-precision cancellation in 1 - ξ", "residuals stuck, 150 mV voltage error", "species computed from the remaining fractions e_{m}", "residuals down 100x in 500 steps"],
      ["extents drift over the 10 h discharge", "error wanders between 0.7 and 1.5 mV", "global charge-inventory residual", "0.1 C error about 0.2 mV"],
      ["separator salt residual on the wrong scale", "1 C separator salt gradient about 10 % too small; a residual that looked converged was a 25 % error of the separator dynamics", "separator residual scaled with (1 - t_{+}) I / (F L_{sep}), its own transient scale", "1 C error 1.97 → 0.12 mV"],
      ["inverse: Adam crawls along correlated parameters", "k_{0,1} still 10 % off after 7000 steps", "Levenberg-Marquardt step on the parameters", "within 1-2 % after 750 steps"],
      ["real data: data bend the physics", "PINN reports 14 mV, FV at its parameters 20 mV", "physics-only fields + trust region", "physics-consistent, 1.8 mV above the FV optimum"],
      ["protocol bug (ours)", "the saved protocol was read back with the wrong key: a resumed 1 C run and one re-evaluation silently used the 0.1 C current", "protocol read with both key names, error otherwise; run repeated", "exposed by the re-evaluation (257 mV instead of 0.12 mV); invalid run kept with a note"],
      ["finite-difference step too small (ours)", "least-squares fits slow or stalled", "absolute 1 % / 1 mV steps", "SPAN500 fit 12.2 → 10.0 mV; Fig. 4b fit unchanged (18.4 mV)"],
      ["SPAN500 PINN: data window, warm start, bounds", "trust-region steps rejected or stuck", "window before the cut-off; re-adaptation stage; wider bounds", "8.9 mV vs 8.2 mV for the FV optimum"],
    ], [2300, 2500, 2500, 2060], "Problems met during the Li-SPAN work and their solutions."),
    CAUTION("three wrong numbers in our own notes", ["Building the paper numbers from the saved files showed three values in our working notes that were wrong: the bounds of the transport parameters with four C-rates are 15-72 %, not 70-190 %; the largest salt-concentration change at 1 C is 13.5 %, not 3 %; and the two SPAN500 kinetic hypotheses differ at 1 C by up to 135 mV, not 95 mV (95 mV is the value at 50 mAh g^{-1}). All three are corrected everywhere; none changes a conclusion."]),
    KEY("why we tell this", ["A reviewer trusts a paper more when it shows how it checked itself. The traceable build is not bureaucracy: in two days it exposed one code bug (and the invalid training run it caused) and three wrong numbers."]),
    P("A practical note: the computations ran on a two-core cloud machine that was reset several times when the session was idle, so all long jobs were made resumable; training costs are reported as CPU time per step (measured time per step × number of steps), which is less sensitive to such interruptions."),
  ];
};
