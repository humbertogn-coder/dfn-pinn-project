// Chapter 1: context and goal
module.exports = (h) => {
  const { P, H1, H2, UL, EQ, ANALOGY, KEY, CAUTION, TABLE } = h;
  return [
    H1("1. Context and goal"),
    H2("1.1 Why lithium-SPAN, and why a model"),
    P("Lithium-sulfur batteries promise more energy per kilogram than lithium-ion batteries, because sulfur is light and can take up two electrons per atom. Their weak point is that, in most electrolytes, the sulfur dissolves as polysulfide molecules that travel to the lithium anode and are lost (the *shuttle*). In **sulfurized polyacrylonitrile (SPAN)** the sulfur is chemically bound to a carbon-nitrogen polymer backbone. In carbonate electrolytes it reacts without dissolving, so the shuttle largely disappears and the cell can cycle for hundreds of cycles. Our group, together with collaborators at Brookhaven National Laboratory and other institutions, studies how the SPAN structure controls this behavior (Wang et al., *Nature Materials* 2026)."),
    P("A **continuum model** of the cell turns the chemistry into equations that predict the voltage during a discharge, the concentration of each species inside the electrode, and how these change with the current. Such a model lets us ask *why* a cell behaves as it does and *which material property* we should change. The first continuum model of a Li-SPAN cell was published by **Simanjuntak, Danner, Wang, Buchmeiser and Latz (Electrochimica Acta 497, 144571, 2024)**. Our project uses that model as its physics."),
    H2("1.2 The Simanjuntak model in one page"),
    P("The cathode contains SPAN in which sulfur sits in short chains of four atoms (S_{4}) bound to the polymer. During discharge each chain is reduced in **three sequential steps**, each transferring two electrons per S_{4} chain:"),
    ...UL(["step 1: S_{4} → S_{3}Li + SLi (the chain is cut);",
      "step 2: S_{3}Li → S_{2}Li + released sulfide S^{2-};",
      "step 3: S_{2}Li → SLi + released sulfide S^{2-};",
      "the released S^{2-} meets Li^{+} in the electrolyte and precipitates as solid Li_{2}S."]),
    P("There is no diffusion inside the solid (the sulfur does not move), so at every point of the electrode the four sulfur species just change in time according to the three reaction rates r_{1}, r_{2}, r_{3}:"),
    EQ("species"),
    P("Each rate follows a Butler-Volmer law. Its driving force is the difference between the electrode potential and an equilibrium potential U that decreases linearly as the step progresses (the slope b_{m} was fitted by the authors to measured open-circuit voltages):"),
    EQ("rate"),
    P("Here Δφ is the potential difference between solid and electrolyte, ζ_{m} is how far step m has progressed, and δ_{m} switches the backward (oxidation) term on or off. The electrolyte (1 M LiPF_{6} in carbonate) is described by Nernst-Planck transport, the lithium metal anode by its own Butler-Volmer law, and the cell voltage is what the cathode current collector sees minus a contact-resistance drop:"),
    EQ("voltage"),
    ANALOGY("a three-stage assembly line", ["Picture a factory where each sulfur chain passes three workstations in order. Station 1 cuts the chain, station 2 and station 3 each remove one more piece. Each station works faster when the 'pressure' (the electrode potential) is higher than the station's own threshold (U), and each station's threshold drops as it runs out of work. The **cell voltage** is the pressure the factory needs to keep all stations busy at the speed the customer (the current) demands. The three plateaus of a SPAN discharge curve are the three stations taking turns."]),
    H2("1.3 What a PINN is"),
    P("A **physics-informed neural network** is trained on the model equations instead of data: its outputs (concentrations, potentials) must satisfy the equations at thousands of random points. When the equation errors (*residuals*) are near zero, the network is a solution of the model; the same network can then be asked to match measured data with some parameters left free (the **inverse problem**). Chapter 4 of the student guide explains this in detail."),
    H2("1.4 Goal of the paper"),
    P("The paper we want to write is a **short, correct methods paper** for JES, whose readers are modelers in the Newman tradition. It should show four things and state their limits:"),
    ...UL(["a faithful, documented numerical reference of the Li-SPAN model (the published model is ambiguous in several places);",
      "a PINN that reproduces that reference to a fraction of a millivolt, with the design choices that made it possible;",
      "what discharge curves can and cannot tell us about the parameters, and an inverse PINN that is as accurate as an ideal estimator;",
      "two applications to measured curves, and an honest comparison of the computing cost."]),
    P("Scope agreed on October 7: Li-SPAN only (no Li-S); real data limited to what is available (the digitized Simanjuntak Fig. 4b points and the SPAN500 data of our Nat. Mater. 2026 paper); a first full draft of the manuscript now exists, with every number generated from saved results."),
    KEY("why this matters for the group", ["A validated Li-SPAN model plus a tool that reads parameters from voltage curves is the bridge between our atomistic and structural studies of SPAN (what the material *is*) and the performance of a cell (what the material *does*)."]),
    CAUTION("our model is the published model, not a new one", ["Everything in this work rests on the physics of Simanjuntak et al. Where we had to make choices (section 2.1) we say so and quantify them. A reviewer may ask whether the authors agree with our reading; contacting T. Danner (DLR) is listed as an optional step in chapter 9."]),
  ];
};
