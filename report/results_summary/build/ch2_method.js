// Chapter 2: methodology
module.exports = (h) => {
  const { P, H1, H2, H3, UL, OL, EQ, FIGURE, ANALOGY, KEY, CAUTION, TABLE } = h;
  return [
    H1("2. Methodology"),
    P("The work followed six phases (Figure {figA}). Each phase produced a tool that the next one needed: the reference model is the ruler for the PINN, the PINN is the engine of the inverse problem, the identifiability study says which parameters the inverse problem can hope to find, and the measured data test everything against reality. This chapter explains the tools; chapter 3 tells the story of the phases and the problems we met; chapter 4 gives the results."),
    ...FIGURE("figA_workflow.png", "The six phases of the Li-SPAN work (drawing). Blue: research phases; green: the five consolidation steps agreed for the paper (reproducibility, cost, traceability, assumptions, real-data case).", 5.4),

    H2("2.1 The reference model: a finite-volume solver"),
    P("PyBaMM, the standard open-source battery simulator, has no Li-SPAN model, so we wrote our own. The electrode and the separator are cut into small volumes (40 in the cathode and 20 in the separator for the PINN benchmark and the synthetic data; 20 and 10 for the comparison with the paper, the fits, the identifiability study and SPAN500); inside each volume the species and potentials are averaged, and the equations become a large set of ordinary differential equations in time, solved with a stiff integrator (backward differentiation formula, tolerance 10^{-6}). This is called a **finite-volume (FV)** method. One full discharge takes about 4-5 s at 0.1 C and 5-7 s at 1 C on one processor core (coarse and fine grid)."),
    ANALOGY("the ruler", ["Before you can say that a new measuring tape is accurate, you need a ruler you trust. The FV solver is our ruler: we checked it against the paper's curves (chapter 4.1), against a different grid (20 / 10 and 40 / 20 volumes give voltages within 0.1 mV), and against conservation of charge and sulfur. Every PINN number in this document is a comparison with this ruler."]),
    H3("Closures: where the paper had to be interpreted"),
    P("Reproducing the published curves was not possible with the equations and table values exactly as printed. In eight places we had to choose (Table {tabclos}). The three that matter most: we count **all three steps** in the electric current (the printed definition omits step 1, which carries a third of the capacity); we use **effective rate constants** fitted to the paper's curves (the printed values make the voltage almost independent of the current, unlike the paper's own Fig. 6a); and we treat **step 3 as irreversible** during discharge (switching on the printed backward terms of steps 2 and 3 together changes the voltage by 53 mV rms, against 2.7 mV for step 2 alone, and contradicts the paper's species curves)."),
    ...TABLE(["#", "Choice", "Why"], [
      ["B1", "faradaic current includes step 1", "step 1 carries one third of the capacity (paper Fig. 5a)"],
      ["B2", "effective rate constants fitted to paper Fig. 6a", "printed values give no rate dependence; 16 / 18 mV rms after the fit"],
      ["B3", "step 3 irreversible during discharge", "literal model (steps 2 and 3 reversible): 53 mV rms change and wrong species curves"],
      ["B4", "S^{2-} reference concentrations (0.01 and 10^{-5} mol m^{-3})", "needed for the open-circuit break points; a 10x change moves V by ≤ 0.01 mV"],
      ["B5", "ion diffusivities from the salt diffusivity and t_{+}", "the printed formula swaps t_{+} and 1 - t_{+}"],
      ["B6", "a permanent Li_{2}S nucleation seed (10^{-5})", "without it Li_{2}S never nucleates"],
      ["B7", "capacity normalized by the sulfur implied by the model (0.767 mg cm^{-2})", "reproduces the paper's axes (text says 0.6)"],
      ["B8", "contact resistance 0.025 Ω m^{2} (Table 2)", "the text quotes 0.035; Fig. 4b matches 0.025"],
    ], [700, 4300, 4360], "Closures of the published model (details: docs/PAPER_ASSUMPTIONS.md, docs/LI_SPAN_MODEL_FORMULATION.md section 5.2).", "tabclos"),

    H2("2.2 The PINN: build the physics in, penalize the rest"),
    P("The main lesson of the whole project (DFN and Li-SPAN) is that a PINN works well when **everything the physics fixes exactly is built into the network's structure**, and only the remaining equations are left as residuals to be minimized. For Li-SPAN this meant four built-in features:"),
    ...UL([
      "**Ordered reaction extents.** The network outputs three positive numbers P_{m} at each point; from them we build the fraction of each step that is still to be done, e_{m}, and the extents ξ_{m} (equation below). By construction the steps happen in order (step 3 can never be ahead of step 2), start at zero and never exceed one, so sulfur is conserved exactly.",
      "**Exact current.** A second network decides *where* in the electrode the reaction happens, but its output is normalized so that the total reaction rate always equals the current the cycler imposes.",
      "**Exact kinetics.** The potential difference Δφ is not a network output; at each point it is computed by solving the Butler-Volmer equation for the rate the network asks for (a small root-finding step inside the network).",
      "**Exact initial state** for the electrolyte, and the anode reaction built into the electrolyte potential.",
    ]),
    EQ("extents"),
    P("What is left (the extent equations, Ohm's law in solid and electrolyte, the salt balance and its boundary conditions) becomes the loss function, a weighted sum of the mean squared residuals of each group; the weights adapt during training so that no group dominates:"),
    EQ("loss"),
    H3("The global charge inventory"),
    P("One extra residual turned out to be decisive. Every electron that leaves through the external circuit corresponds to one step of the reaction somewhere in the electrode, so the integral of the extents over the electrode must equal the charge passed:"),
    EQ("inventory"),
    P("This equation follows from the extent equations, so in theory it adds nothing. In practice the extent equations are satisfied only up to a small mean error, and over a ten-hour discharge that error accumulates into a few mol m^{-3} of missing or extra product, which in the last plateau shows up directly as millivolts of voltage error. Adding the inventory as its own residual removed this drift (0.1 C error from about 1 mV to about 0.2 mV)."),
    ANALOGY("the bank statement", ["You can track your money by recording every purchase (the extent equations). Small rounding errors in each entry add up over a year. The bank statement (the charge passed, which we know exactly) lets you check the total at any time. The inventory residual is that monthly check."]),
    H3("Training recipe"),
    P("Four networks of 4 layers × 96 neurons, 30 000 optimization steps (Adam, learning rate decreasing from 10^{-3} to 10^{-5}), new random collocation points at every step, the inventory residual switched on after 10 000 steps (switched on from the start, it fights the extent equations before they are learned), one CPU thread. The same recipe is used for both C-rates and for all seeds."),

    H2("2.3 Identifiability: which parameters can a voltage curve reveal?"),
    P("Before trying to estimate parameters, we asked which ones the data can determine at all. For each parameter θ_{i} we compute, with the FV solver, how much the voltage curve changes when θ_{i} changes slightly (the sensitivity S). From the sensitivities and the measurement noise σ we build the **Fisher information matrix** F; its inverse gives the **Cramer-Rao lower bound (CRLB)**, the smallest standard error any unbiased method can achieve with that data:"),
    EQ("crlb"),
    ANALOGY("two people pushing a car", ["If two people push a stuck car from the same side, you can measure how far the car moved, but not who pushed harder: only their sum is visible. Two parameters with the same effect on the voltage are like that, and the Fisher matrix detects it (their bounds become huge and their correlation close to ±1). Pushing from different angles (a second C-rate) separates them."]),

    H2("2.4 The inverse problem and the ideal estimator"),
    P("For the inverse problem one PINN per C-rate is trained, all sharing the same unknown parameters. Plain gradient descent on network weights and parameters together turned out to be very slow, because the parameters are correlated (they sit in a long narrow valley of the misfit). The fix is a **Levenberg-Marquardt (LM) step** on the parameters every 250 training steps, with the networks frozen for that step:"),
    EQ("lm"),
    P("Here r is the vector of data residuals, J its derivative with respect to the parameters and λ a damping factor. For measured data, which the model cannot match exactly, a *trust-region* version keeps the networks physics-only and accepts a parameter step only if the misfit really decreases after the networks re-adjust."),
    P("Noise makes even a perfect method miss the true parameters, so we judge the PINN against the **ideal estimator**: a least-squares fit with the FV solver itself on exactly the same noisy data. When the two differ, a linear error model predicts the difference from the PINN's forward error e (PINN minus FV voltage at the same parameters):"),
    EQ("bias"),

    H2("2.5 Real data used"),
    P("Two measured data sets were used (details in section 4.6): the experimental points of Simanjuntak et al. Fig. 4b at 0.05-1 C, digitized from the figure (we fit 0.1 + 1 C and *predict* 0.05 and 0.2 C), and the C/10 discharges of the SPAN500 coin cell of Wang et al. (Nat. Mater. 2026), our group's own data, at a single rate and without reported sulfur content."),
    H2("2.6 Reproducibility and traceability"),
    P("Because the paper must be correct, we followed four rules: (1) the final recipe is trained from scratch with **three seeds per C-rate** and the spread is reported, not the best run; (2) **no number is typed by hand**: `scripts/paper_build.py` regenerates every figure, table and number from saved results, fingerprints every input file, and fills the placeholders of the manuscript; (3) the build **re-evaluates the saved networks** instead of copying numbers from training logs; (4) **failed runs are kept** with a note explaining why they stopped."),
    KEY("these rules found real errors", ["When the build re-evaluated a saved 1 C network it reported 257 mV instead of 0.12 mV: the code that reads a run's protocol back from its configuration silently fell back to the 0.1 C current. The same bug had made a resumed 1 C training run train at the wrong current; that run was kept with a note and repeated. The traceable numbers also caught three wrong values in our own notes (section 3.2). Without these rules they would have reached the manuscript."]),
  ];
};
