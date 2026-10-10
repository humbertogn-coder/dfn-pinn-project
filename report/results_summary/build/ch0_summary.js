// Executive summary
module.exports = (h) => {
  const { P, H1, H2, UL, KEY, CAUTION, SLIDE, TABLE } = h;
  return [
    H1("Executive summary", true),
    P("This document summarizes, explains and analyzes the results of the **Li-SPAN part of the PINN-DFN project**, so that they can be presented to our advisor and she can decide whether we start writing the manuscript for the *Journal of The Electrochemical Society* (JES). It uses the same plain language as the student guide (October 2026): every method is explained with an analogy first, then with the equation, then with what we found. It does not repeat the background of the guide (batteries, the DFN model, neural networks); read chapters 2-4 of the guide if those are new to you."),
    H2("What we did, in five sentences"),
    ...UL([
      "We rebuilt the published Li-SPAN cell model (Simanjuntak et al. 2024) as a finite-volume (FV) solver, fixed its ambiguities, and checked it against the paper.",
      "A physics-informed neural network (PINN) trained from scratch reproduces the FV voltage to **0.26 mV rms at 0.1 C and 0.18 mV at 1 C** (mean of three seeds).",
      "One C-rate cannot separate the kinetics from the contact resistance; **0.1 C + 1 C determine eight parameters to about 1 %**.",
      "The inverse PINN recovers them within **2 Cramer-Rao bounds of an ideal estimator**, and its small bias is predicted by its own forward error.",
      "On two measured data sets the model fits to **8-20 mV**, limited by the model; for this 1-D cell the PINN is about **1000 times slower** than the FV solver.",
    ]),
    ...TABLE(["Question", "Answer", "Evidence"], [
      ["Is our reference model a faithful copy of the published one?", "Yes, after eight documented closures; 16 / 18 mV rms from the paper's curves at 0.1 / 1 C", "Figure {fig1}, Table {tabclos}"],
      ["Can a PINN solve the Li-SPAN model accurately?", "Yes: 0.14-0.44 mV rms, all six runs below 0.5 mV", "Figure {fig2}, Table {tabfwd}"],
      ["Is it reproducible?", "Yes: three seeds per rate from random initialization; one seed has a narrow 3.5 mV spike at a phase transition", "Table {tabfwd}"],
      ["Which parameters can a voltage curve reveal?", "Eight (k_{0,1-3}, b_{1-3}, U_{0,1}, Z_{CC}) with two C-rates; transport and Li_{2}S parameters no", "Figure {fig4}"],
      ["Does the inverse PINN work?", "Yes, as well as an ideal estimator up to a bias that we can predict", "Figure {fig5}, Table {tabinv}"],
      ["Does it work on real data?", "The fits are as good as the model allows (8-20 mV); the real-data cases are demonstrations with clear limits", "Figure {fig6}, Table {tabmeas}"],
      ["Is the PINN faster than a classical solver?", "No (seconds vs hours). Its value is methodological", "Figure {figB}"],
    ], [3000, 4200, 2160], "The project in seven questions."),
    KEY("is the current work enough to start writing?", [
      "**Yes, for the decision to start writing**: every result the paper needs exists, is reproducible from saved files, and a first full draft of the manuscript already exists (`paper/manuscript/`). Before submission we still need (i) the literature positioning and reference list, (ii) an agreed framing of the cost result, and (iii) the supplementary material. Two optional items would make the paper stronger: confirmation of our model closures by the original authors, and multi-rate measurements of a SPAN cell from our group. Chapter 9 gives the full checklist.",
    ]),
    CAUTION("what this work does NOT claim", [
      "It does not propose a new SPAN model, it does not claim that the PINN is faster than a finite-volume solver, and it does not claim that the parameters fitted to the SPAN500 data are material properties valid at other rates. These limits are part of the message, not a weakness to hide (chapter 8).",
    ]),
  ];
};
