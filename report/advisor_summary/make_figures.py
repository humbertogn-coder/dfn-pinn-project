"""Figures of the short advisor summary (report/advisor_summary/PINN_LiSPAN_advisor_summary.docx).

    python report/advisor_summary/make_figures.py [name ...]     -> report/advisor_summary/figures/*.png

Drawings (a*) are labelled as drawings. Every data figure (r*, c*) is regenerated from saved runs and results of
the repository; the data source of every experimental curve is written on the figure.
"""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
RES = ROOT / "results"
RUNS = RES / "lispan_runs"

# validated categorical palette (dataviz reference instance), fixed order
C1, C2, C3, C4, C5, C6, C7, C8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DATA = "#7a7974"          # measured / published data (grey)
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.axisbelow": True})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc, ec=INK2, fs=9, color=INK, weight="normal", ha="center", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03", fc=fc, ec=ec, lw=lw))
    tx = x + w / 2 if ha == "center" else x + 0.03
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=color, weight=weight, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.6, ms=13):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw))


# =============================================================================================== drawings
def a1_dfn_effects():
    """DFN cell with the physical effects it describes, numbered."""
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.add_patch(Rectangle((0.3, 0.9), 0.22, 3.2, fc="#c9c9c9", ec=INK2))
    ax.add_patch(Rectangle((9.48, 0.9), 0.22, 3.2, fc="#c9c9c9", ec=INK2))
    ax.add_patch(Rectangle((0.52, 0.9), 3.4, 3.2, fc="#eef4fc", ec=INK2))
    ax.add_patch(Rectangle((3.92, 0.9), 1.9, 3.2, fc="white", ec=INK2, hatch="////", alpha=0.6))
    ax.add_patch(Rectangle((5.82, 0.9), 3.66, 3.2, fc="#fdf0ea", ec=INK2))
    ax.text(2.2, 4.3, "negative electrode\n(graphite particles)", ha="center", fontsize=9)
    ax.text(4.87, 4.3, "separator", ha="center", fontsize=9)
    ax.text(7.65, 4.3, "positive electrode\n(NMC particles)", ha="center", fontsize=9)
    ax.text(0.41, 0.6, "Cu", ha="center", fontsize=8, color=INK2); ax.text(9.59, 0.6, "Al", ha="center", fontsize=8, color=INK2)
    rng = np.random.default_rng(3)
    for (x0, x1, c) in ((0.8, 3.6, C1), (6.1, 9.2, C2)):
        n = 0
        while n < 10:
            x, y = rng.uniform(x0, x1), rng.uniform(1.2, 3.8)
            if c == C2 and np.hypot(x - 7.6, y - 2.45) < 1.05:
                continue
            ax.add_patch(Circle((x, y), 0.26, fc=c, ec="white", lw=1.2, alpha=0.85))
            n += 1
    # (1) diffusion inside a particle: big particle with gradient
    for r, a in ((0.62, 0.25), (0.46, 0.45), (0.3, 0.7), (0.14, 0.95)):
        ax.add_patch(Circle((7.6, 2.45), r, fc=C2, ec="none", alpha=a))
    ax.add_patch(Circle((7.6, 2.45), 0.62, fc="none", ec=INK, lw=1.4))
    arrow(ax, 7.6, 2.45, 8.15, 2.45, color=INK, lw=1.4, ms=10)
    # (2) reaction at the surface
    ax.plot([8.22], [2.45], "o", ms=7, mfc="#ffd84d", mec=INK)
    # (3) ions in the electrolyte
    arrow(ax, 2.6, 1.15, 7.0, 1.15, color=C3, lw=2.2)
    # (4) electrons in the solid / external circuit
    ax.plot([0.41, 0.41, 9.59, 9.59], [4.1, 5.25, 5.25, 4.1], color=INK2, lw=1.5)
    arrow(ax, 4.2, 5.25, 5.8, 5.25, color=C7, lw=2.0)
    # number badges
    def badge(x, y, n):
        ax.add_patch(Circle((x, y), 0.19, fc=INK, ec="white", lw=1.0, zorder=5))
        ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=9, weight="bold", zorder=6)
    badge(7.0, 3.1, 1); badge(8.5, 2.75, 2); badge(4.85, 1.45, 3); badge(5.0, 5.25 + 0.3, 4); badge(9.2, 5.55, 5)
    ax.text(8.95, 5.55, "V", ha="right", va="center", fontsize=10, weight="bold", color=INK)
    legend = ("1  lithium diffuses inside each particle (solid diffusion)\n"
              "2  lithium enters / leaves a particle at its surface (reaction, Butler-Volmer)\n"
              "3  Li$^+$ ions move through the liquid electrolyte (diffusion + migration)\n"
              "4  electrons move through the solid and the external wire (Ohm's law)\n"
              "5  the cell voltage = what is left after all these losses")
    ax.text(0.3, 0.42, legend, ha="left", va="top", fontsize=8.6, color=INK, linespacing=1.45,
            transform=ax.transData)
    ax.set_ylim(-1.15, 5.7)
    save(fig, "a1_dfn_effects.png")


def a1b_lispan_cell():
    """Li-SPAN cell as the Simanjuntak model sees it, with numbered effects (drawing)."""
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.set_xlim(0, 10); ax.set_ylim(-1.15, 5.7); ax.axis("off")
    ax.add_patch(Rectangle((0.3, 0.9), 0.22, 3.2, fc="#c9c9c9", ec=INK2))
    ax.add_patch(Rectangle((0.52, 0.9), 0.55, 3.2, fc="#d9dde3", ec=INK2))
    ax.add_patch(Rectangle((1.07, 0.9), 3.6, 3.2, fc="white", ec=INK2, hatch="////", alpha=0.6))
    ax.add_patch(Rectangle((4.67, 0.9), 4.81, 3.2, fc="#fdf0ea", ec=INK2))
    ax.add_patch(Rectangle((9.48, 0.9), 0.22, 3.2, fc="#c9c9c9", ec=INK2))
    ax.text(0.8, 4.3, "lithium\nmetal", ha="center", fontsize=9)
    ax.text(2.87, 4.3, "separator (glass fibre)", ha="center", fontsize=9)
    ax.text(7.07, 4.3, "SPAN electrode\n(SPAN + carbon black)", ha="center", fontsize=9)
    rng = np.random.default_rng(5)
    pts = []
    while len(pts) < 9:
        x, y = rng.uniform(5.0, 9.1), rng.uniform(1.25, 3.75)
        if all(np.hypot(x - a, y - b) > 0.75 for a, b in pts) and np.hypot(x - 6.2, y - 2.5) > 1.3 and not (6.9 < x < 8.6 and 1.55 < y < 3.45):
            pts.append((x, y))
    for x, y in pts:
        ax.add_patch(Circle((x, y), 0.3, fc="#f2b48f", ec=INK2, lw=0.8))
    # zoomed SPAN particle: polymer backbone with S4 bridges, Li2S on the surface
    ax.add_patch(Circle((6.2, 2.5), 0.85, fc="#f2b48f", ec=INK, lw=1.4))
    for yy in (2.05, 2.5, 2.95):
        ax.plot([5.55, 6.85], [yy, yy], color=INK2, lw=1.4)
    for xx in (5.8, 6.2, 6.6):
        ax.plot([xx, xx], [2.05, 2.95], color=C4, lw=2.4)
        for yy in (2.16, 2.27, 2.38, 2.61, 2.72, 2.83):
            ax.plot(xx, yy, "o", ms=3.2, color=C4)
    for ang in (0.3, 1.2, 2.2, 3.6, 4.6, 5.5):
        ax.add_patch(Rectangle((6.2 + 0.9 * np.cos(ang) - 0.08, 2.5 + 0.9 * np.sin(ang) - 0.08), 0.16, 0.16,
                               fc="#7d6bd6", ec="white", lw=0.6))
    ax.text(7.25, 2.5, "sulfur fixed\non the polymer\n(no diffusion\ninside)", ha="left", va="center", fontsize=7.5, color=INK2)
    arrow(ax, 1.6, 1.15, 5.2, 1.15, color=C3, lw=2.2)
    ax.plot([0.41, 0.41, 9.59, 9.59], [4.1, 5.25, 5.25, 4.1], color=INK2, lw=1.5)
    arrow(ax, 4.2, 5.25, 5.8, 5.25, color=C7, lw=2.0)
    ax.add_patch(Rectangle((9.44, 1.0), 0.06, 3.0, fc=C8, ec="none"))

    def badge(x, y, n):
        ax.add_patch(Circle((x, y), 0.19, fc=INK, ec="white", lw=1.0, zorder=5))
        ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=9, weight="bold", zorder=6)
    badge(6.2, 3.6, 1); badge(7.3, 3.25, 2); badge(3.0, 1.45, 3); badge(0.8, 3.6, 4); badge(9.2, 3.8, 5); badge(9.2, 5.55, 6)
    ax.text(8.95, 5.55, "V", ha="right", va="center", fontsize=10, weight="bold", color=INK)
    legend = ("1  S$_4$ bridges on the polymer are reduced in three steps, one after the other (no solid diffusion)\n"
              "2  steps 2 and 3 release S$^{2-}$, which meets Li$^+$ and precipitates as solid Li$_2$S (purple)\n"
              "3  Li$^+$ ions move through the electrolyte (diffusion + migration)\n"
              "4  lithium dissolves from the metal foil\n"
              "5  contact resistance between the electrode and the current collector (red)\n"
              "6  the cell voltage = what is left after all these losses")
    ax.text(0.3, 0.42, legend, ha="left", va="top", fontsize=8.6, color=INK, linespacing=1.45)
    save(fig, "a1b_lispan_cell.png")


def a2_lispan_mechanism():
    shutil.copy(ROOT / "report" / "figures" / "fig09_lispan_reactions.png", OUT / "a2_lispan_mechanism.png")
    print("copied a2_lispan_mechanism.png")


def a3_dfn_vs_lispan():
    """What each model includes (checklist drawing)."""
    rows = [
        ("Negative electrode", "graphite particles", "lithium metal foil (surface only)"),
        ("Positive electrode", "NMC particles (Li inserts into a crystal)", "SPAN: sulfur bound to a polymer backbone"),
        ("Lithium diffusion inside the solid", "yes (main limitation at high rate)", "no: the sulfur does not move"),
        ("Reactions", "one insertion reaction per electrode", "three reduction steps, one after the other"),
        ("Solid product that grows", "no", "yes: Li$_2$S precipitates and fills pores"),
        ("Dissolved sulfur species", "no", "only S$^{2-}$ (very low solubility)"),
        ("Ions in the electrolyte", "yes", "yes"),
        ("Contact resistance", "usually lumped", "yes (Z$_{CC}$, important at 1 C)"),
    ]
    missing = ["polysulfide shuttle (assumed absent: carbonate electrolyte)",
               "different chain lengths (only S$_4$ chains are modelled)",
               "first-cycle activation (irreversible Li trapping)",
               "electrolyte properties that change with concentration",
               "temperature changes; charging; 2-D/3-D structure"]
    fig, ax = plt.subplots(figsize=(9.6, 6.1))
    ax.set_xlim(0, 10); ax.set_ylim(-0.7, 10); ax.axis("off")
    ax.text(3.0, 9.6, "classic DFN\n(graphite | NMC, e.g. LG M50)", ha="left", va="center", fontsize=10, weight="bold", color=C1)
    ax.text(6.55, 9.6, "Li-SPAN model\n(Simanjuntak et al. 2024)", ha="left", va="center", fontsize=10, weight="bold", color=C2)
    y = 8.65
    for i, (what, dfn, span) in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.05, y - 0.42), 9.9, 0.84, fc="#f6f5f2", ec="none"))
        ax.text(0.15, y, what, ha="left", va="center", fontsize=8.8, weight="bold", color=INK)
        ax.text(3.0, y, dfn, ha="left", va="center", fontsize=8.6, color=INK)
        ax.text(6.55, y, span, ha="left", va="center", fontsize=8.6, color=INK)
        y -= 0.84
    y -= 0.25
    ax.add_patch(FancyBboxPatch((0.1, -0.55), 9.8, y + 0.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#fdf3e7", ec=C4, lw=1.3))
    ax.text(0.3, y - 0.35, "Not in the Li-SPAN model (its limitations):", ha="left", va="center", fontsize=9.2,
            weight="bold", color=INK)
    for j, m in enumerate(missing):
        col = 0 if j < 3 else 1
        yy = y - 0.85 - 0.45 * (j if j < 3 else j - 3)
        ax.text(0.45 + 4.9 * col, yy, "x  " + m, ha="left", va="center", fontsize=8.6, color=INK)
    save(fig, "a3_dfn_vs_lispan.png")


def a4_workflow():
    steps = [
        (C1, "1  PINN for the classic DFN model (forward)", "LG M50 cell, Chen et al. 2020 parameters, vs PyBaMM", "the method works where the answer is known"),
        (C1, "2  DFN inverse and aging", "synthetic LG M50 data (PyBaMM, new and aged cells)", "it also finds parameters and aging"),
        (C2, "3  Li-SPAN model rebuilt (our reference solver)", "Simanjuntak et al. 2024, no public code", "we have a trusted copy of the model"),
        (C2, "4  Li-SPAN forward PINN", "checked against the model and the paper's curves", "about 0.2 mV error at 0.1 C and 1 C"),
        (C2, "5  Li-SPAN inverse PINN", "synthetic data, then Simanjuntak's measured points", "8 parameters from two C-rates"),
        (C3, "6  Independent real data", "SPAN300/320/500, Wang et al., Nat. Mater. 2026", "works on independent cells, other electrolyte"),
        (C3, "7  Chemical information", "rates, first cycle, synthesis temperature", "use the tool to learn about the chemistry"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.3); ax.axis("off")
    ax.text(0.2, 7.05, "what we did", fontsize=9, color=INK2, weight="bold")
    ax.text(6.55, 7.05, "what it shows", fontsize=9, color=INK2, weight="bold")
    for i, (c, title, sub, shows) in enumerate(steps):
        y = 6.15 - i * 0.98
        ax.add_patch(FancyBboxPatch((0.15, y), 6.0, 0.78, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc="white", ec=c, lw=1.8))
        ax.add_patch(Rectangle((0.15, y), 0.12, 0.78, fc=c, ec="none"))
        ax.text(0.45, y + 0.53, title, fontsize=9.6, weight="bold", va="center", color=INK)
        ax.text(0.45, y + 0.22, sub, fontsize=8.6, va="center", color=INK2)
        ax.text(6.55, y + 0.39, shows, fontsize=8.8, va="center", color=INK)
        arrow(ax, 6.2, y + 0.39, 6.45, y + 0.39, color=INK2, lw=1.0, ms=8)
        if i < len(steps) - 1:
            arrow(ax, 1.0, y, 1.0, y - 0.2, color=INK2, lw=1.4, ms=9)
    for c, lab, x in ((C1, "classic DFN (graphite | NMC)", 0.2), (C2, "Li-SPAN, model and its own data", 3.4),
                      (C3, "Li-SPAN, independent data", 6.7)):
        ax.add_patch(Rectangle((x, -0.12), 0.25, 0.18, fc=c, ec="none"))
        ax.text(x + 0.35, -0.03, lab, fontsize=8.6, va="center", color=INK)
    ax.set_ylim(-0.3, 7.3)
    save(fig, "a4_workflow.png")


def a5_why_dfn_first():
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0.4, 3.6); ax.axis("off")
    box(ax, 0.1, 0.5, 2.9, 2.6, "Known system\n(graphite | NMC, DFN)\n\na trusted solver exists\n(PyBaMM): if the PINN\nmisses, the PINN is wrong",
        "#eef4fc", ec=C1, fs=9)
    box(ax, 3.55, 0.5, 2.9, 2.6, "Lessons that transfer\n\nbuild exact physics into\nthe network; balance the\nequations; add global\nconservation; use 2 C-rates",
        "#f6f5f2", ec=INK2, fs=9)
    box(ax, 7.0, 0.5, 2.9, 2.6, "New system\n(Li-SPAN)\n\nno public solver, new\nchemistry: the tool is\nalready validated, so a\nmisfit points to the model",
        "#fdf0ea", ec=C2, fs=9)
    arrow(ax, 3.02, 1.8, 3.52, 1.8); arrow(ax, 6.47, 1.8, 6.97, 1.8)
    ax.text(5.0, 3.35, "Validate the tool where the answer is known, then use it where it is not", ha="center", fontsize=10,
            weight="bold", color=INK)
    save(fig, "a5_why_dfn_first.png")


# =============================================================================================== DFN results
def _dfn_model(run, ckpt="final.pt"):
    import torch
    from dfn_pinn.v2.params import CellParams, Protocol, Scales
    from dfn_pinn.v2.model import DFNPINN
    ck = torch.load(run / ckpt, weights_only=False)
    cfg = ck["train"]; cell = CellParams(**ck["cell"]); prot = Protocol(**ck["protocol"])
    m = DFNPINN(Scales(cell, prot), cfg["width"], cfg["depth"], cfg["act"], cfg["projection"], ic_tau_s=cfg["ic_tau_s"],
                inventory=cfg.get("inventory", "soft"), fourier_t=cfg.get("fourier_t", 0),
                width_scalar=cfg.get("width_scalar", 0) or None, short_t=tuple(cfg.get("short_t", [])),
                collector_bc=cfg.get("collector_bc", "soft"), fourier_period=cfg.get("fourier_period", 1.0))
    m.load_state_dict(ck["model"], strict=False)
    return m.to(torch.float64), cfg


def r1_dfn_vs_pybamm():
    from dfn_pinn.v2.reference import load
    from dfn_pinn.v2.evaluate import predict_on_reference
    ref = load(ROOT / "results" / "v2_reference" / "ref_I5A_t3000s_ramp30s_x80_r120.npz")
    t = np.asarray(ref["t"]) / 60
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax[0].plot(t, ref["V"], color=DATA, lw=4, alpha=0.6, label="PyBaMM (reference solver)")
    stats = []
    for run, lab, c in (("f4_hardbc_w96_40k_20261005T194145Z", "PINN, training 1", C1),
                        ("f4_hardbc_w96_40k_seed1_20261006T095948Z", "PINN, training 2", C2)):
        m, cfg = _dfn_model(RES / "v2_runs" / run)
        p = predict_on_reference(m, ref, kinetics=cfg.get("kinetics", "inverse"))
        e = 1e3 * (np.asarray(p["V"]) - np.asarray(ref["V"]))
        stats.append((lab, float(np.sqrt(np.mean(e ** 2)))))  # whole 3000 s window, as in docs/V2_RESULTS.md
        if c == C1:
            ax[0].plot(t, p["V"], "--", color=C1, lw=1.6, label="PINN")
        ax[1].plot(t, e, color=c, lw=1.3, label=f"{lab}: {stats[-1][1]:.2f} mV on average")
    ax[0].set(xlabel="time [min]", ylabel="cell voltage [V]", title="(a) 1C discharge of the LG M50 cell")
    ax[0].legend(fontsize=8.5)
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].set(xlabel="time [min]", ylabel="error [mV]", title="(b) error of the PINN", ylim=(-1.5, 1.5))
    ax[1].legend(fontsize=8.5, loc="lower center")
    save(fig, "r1_dfn_vs_pybamm.png")
    return stats


def r2_dfn_inverse_aging():
    h = json.loads((RES / "v2_runs/B2_hier_w96_cont_20261006T085648Z/history.json").read_text())["history"]
    last = [x for x in h if x.get("parameters")][-1]["parameters"]
    true = {"D_p": 4e-15, "k_n": 6.48e-7, "D_n": 3.3e-14}
    e1 = [100 * (last[k] / v - 1) for k, v in true.items()]
    r = json.loads((RES / "v2_runs/C3b_aging_cellC_c200_Dstress_R0free_20261006T211155Z/aging_result.json").read_text())
    names = ["theta_n0", "theta_p0", "eps_am_n", "eps_am_p"]
    e2 = [100 * (r["estimates"][n] / r["truth"][n] - 1) for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.3), gridspec_kw={"width_ratios": [3, 4]})
    l1 = ["Li diffusion\nin NMC", "reaction rate\nin graphite", "Li diffusion\nin graphite"]
    l2 = ["lithium in\ngraphite", "lithium in\nNMC", "active\ngraphite", "active\nNMC"]
    for a, err, lab, col, title in ((ax[0], e1, l1, C1, "(a) new cell: 3 transport/kinetic parameters"),
                                    (ax[1], e2, l2, C2, "(b) aged cell, 200 cycles: 4 aging parameters")):
        a.bar(range(len(err)), err, color=col, width=0.55)
        a.axhline(0, color=INK, lw=1)
        for i, e in enumerate(err):
            a.text(i, e + (0.12 if e >= 0 else -0.12), f"{e:+.1f} %", ha="center", va="bottom" if e >= 0 else "top", fontsize=9)
        a.set_xticks(range(len(err))); a.set_xticklabels(lab, fontsize=8.5)
        a.set_ylim(-2.4, 2.4); a.set_title(title)
    ax[0].set_ylabel("error of the PINN estimate [%]")
    save(fig, "r2_dfn_inverse_aging.png")
    return {"fresh": dict(zip(true, e1)), "aged": dict(zip(names, e2)), "aged_misfit_mV": r["misfit_mV"]}


# =============================================================================================== Li-SPAN results
def _dig(name):
    d = np.loadtxt(RES / "lispan" / "paper_digitized" / name, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1]


def r3_model_vs_paper():
    from dfn_pinn.lispan.model import load
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for cr, col in ((0.1, C1), (1.0, C2)):
        r = load(RES / "lispan" / "ref_Zcc0" / f"discharge_{cr:g}C.npz")
        qd, vd = _dig(f"fig6a_k0_1e-2_{cr:g}C.csv")
        axs[0].plot(qd, vd, "o", ms=4, mfc="none", mec=col, mew=1.0)
        axs[0].plot(r["Q_mAh_gS"], r["V"], color=col, lw=1.8, label=f"{cr:g} C")
    axs[0].plot([], [], "o", ms=4, mfc="none", mec=INK2, label="their model (Simanjuntak et al. 2024, Fig. 6a)")
    axs[0].plot([], [], color=INK2, label="our model")
    axs[0].set_title("(a) our model vs their published model")
    for cr, col in zip((0.05, 0.1, 0.2, 1.0), (C3, C1, C4, C2)):
        r = load(RES / "lispan" / "ref_Zcc0.025" / f"discharge_{cr:g}C.npz")
        qe, ve = _dig(f"fig4b_exp_{cr:g}C.csv")
        axs[1].plot(qe, ve, "o", ms=4.5, mfc="none", mec=col, mew=1.0)
        axs[1].plot(r["Q_mAh_gS"], r["V"], color=col, lw=1.8, label=f"{cr:g} C")
    axs[1].plot([], [], "o", ms=4.5, mfc="none", mec=INK2, label="measured cell (Simanjuntak et al. 2024, Fig. 4b)")
    axs[1].plot([], [], color=INK2, label="our model")
    axs[1].set_title("(b) our model vs their measured cell")
    for ax in axs:
        ax.set(xlabel="capacity [mAh per g sulfur]", ylabel="cell voltage [V]", ylim=(0.95, 2.45))
        ax.legend(fontsize=7.8, loc="lower left")
    save(fig, "r3_model_vs_paper.png")


def _final_runs(cr):
    from dfn_pinn.lispan import runs
    return [d for d in sorted(RUNS.glob(f"lispan_final_{cr}C_seed*_*")) if (d / "final.pt").exists() and not runs.is_stopped(d)]


def r4_r5_forward():
    import torch
    from dfn_pinn.lispan import runs
    from dfn_pinn.lispan.model import load
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 5.4), sharex="col", gridspec_kw={"height_ratios": [1.25, 1]})
    figs, axs2 = plt.subplots(1, 2, figsize=(9.6, 3.6))
    out = {}
    names = {"c_S4": "PAN-S$_4$ (start)", "c_S3": "PAN-S$_3$Li", "c_S2": "PAN-S$_2$Li", "c_S1": "PAN-SLi (end)"}
    for j, cr in enumerate(("0.1", "1")):
        out[cr] = []
        for k, d in enumerate(_final_runs(cr)):
            cfg = json.loads((d / "config.json").read_text())
            p = runs._params(cfg["params"]); prot = runs._protocol(cfg["protocol"])
            rev = "".join("r" if r else "i" for r in p.reversible)
            ref = load(RES / "lispan" / f"pinn_reference_{prot.crate:g}C_{rev}_Zcc{p.Z_CC:g}.npz")
            model = runs.load_model(d / "final.pt")
            t = np.asarray(ref["t"]); keep = t <= model.t_end
            T = torch.as_tensor(t[keep] / model.t_end, dtype=torch.float32).view(-1, 1)
            with torch.no_grad():
                V = model.voltage(T).view(-1).numpy()
            Q = np.asarray(ref["Q_mAh_gS"])[keep]
            e = 1e3 * (V - np.asarray(ref["V"])[keep])
            m = t[keep] > 100
            out[cr].append(float(np.sqrt(np.mean(e[m] ** 2))))
            col = (C1, C2, C3)[k]
            if k == 0:
                axs[0, j].plot(ref["Q_mAh_gS"], ref["V"], color=DATA, lw=4, alpha=0.6, label="model (our solver)")
                axs[0, j].plot(Q, V, "--", color=C8, lw=1.4, label="PINN")
                yc = np.asarray(ref["y_c"]); nt, ny = len(T), len(yc)
                Yq = torch.as_tensor(yc / p.L_cat, dtype=torch.float32).view(1, ny, 1).expand(nt, ny, 1).reshape(-1, 1)
                Tq = T.view(nt, 1, 1).expand(nt, ny, 1).reshape(-1, 1)
                with torch.no_grad():
                    sp = model.species(Yq, Tq)
                for i, key in enumerate(("c_S4", "c_S3", "c_S2", "c_S1")):
                    cc = (C1, C2, C3, C7)[i]
                    axs2[j].plot(ref["Q_mAh_gS"], np.asarray(ref[key]).mean(axis=0), color=cc, lw=4, alpha=0.35)
                    axs2[j].plot(Q, sp[key].view(nt, ny).numpy().mean(axis=1), "--", color=cc, lw=1.3, label=names[key])
            axs[1, j].plot(Q[m], e[m], color=col, lw=1.0, label=f"training {k + 1}: {out[cr][-1]:.2f} mV on average")
        axs[0, j].set_title(f"({'ab'[j]}) {cr} C discharge"); axs[0, j].set_ylabel("cell voltage [V]")
        axs[0, j].legend(fontsize=8); axs[0, j].set_ylim(0.95, 2.6)
        axs[1, j].axhline(0, color=INK2, lw=0.8)
        axs[1, j].set_ylabel("error [mV]"); axs[1, j].set_xlabel("capacity [mAh per g sulfur]")
        axs[1, j].legend(fontsize=7.5, loc="lower left"); axs[1, j].set_ylim(-3, 3)
        axs2[j].set_title(f"({'ab'[j]}) {cr} C: sulfur species"); axs2[j].set_xlabel("capacity [mAh per g sulfur]")
        axs2[j].set_ylabel("concentration [mol/m$^3$]")
    axs2[0].plot([], [], color=INK2, lw=4, alpha=0.35, label="model (thick)")
    axs2[0].plot([], [], "--", color=INK2, lw=1.3, label="PINN (dashed)")
    axs2[0].legend(fontsize=7.5, loc="upper left")
    save(fig, "r4_pinn_forward.png")
    save(figs, "r5_species.png")
    return out


# ------------------------------------------------------------------------------------------------- SPAN500 (cycle 3)
def r6_span500():
    import lispan_span500 as sp5
    fits = json.loads((RES / "lispan" / "span500" / "fit_cycle3.json").read_text())["fits"]
    q3, v3 = sp5.load_discharge(3)
    q2, v2 = sp5.load_discharge(2)
    pinn = json.loads((sorted(RUNS.glob("lispan_inv_span500_c3_paper_wS_b5_*"))[-1] / "inverse_result.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax[0].plot(q3, v3, color=DATA, lw=4.5, alpha=0.55, label="measured, cycle 3 (Wang et al., Nat. Mater. 2026)")
    ax[0].plot(q2, v2, ":", color=INK2, lw=1.4, label="measured, cycle 2 (same cell)")
    f = fits["paper"]
    Q, V, _, _ = sp5.simulate(f["estimates"], sp5.base_params(1.0))
    ax[0].plot(Q, V, color=C1, lw=1.5, label=f"model fitted to cycle 3 ({f['rms_mV']:.0f} mV)")
    est = dict(f["estimates"]); est.update(pinn["estimates"])
    est["w_S"] = f["estimates"]["w_S"]
    Qp, Vp, _, _ = sp5.simulate(est, sp5.base_params(1.0))
    ax[0].plot(Qp, Vp, "--", color=C2, lw=1.4, label="model at the PINN's parameters")
    ax[0].set(xlabel="capacity [mAh per g SPAN]", ylabel="cell voltage [V]", ylim=(0.95, 2.5),
              title="(a) C/10: data and calibrated model")
    ax[0].legend(fontsize=7.6, loc="lower left")
    for hyp, col, lab in (("paper", C1, "slow reactions (Simanjuntak rates)"), ("fast", C2, "1000x faster reactions")):
        f = fits[hyp]
        b = sp5.base_params(f["k0_factor"])
        Q, V, _, _ = sp5.simulate(f["estimates"], b)
        Q1, V1, _, _ = sp5.simulate(f["estimates"], b, crate_span=1.0)
        ax[1].plot(Q, V, color=col, lw=1.0, alpha=0.6)
        ax[1].plot(Q1, V1, color=col, lw=2.0, label=f"{lab}: C/10 fit {f['rms_mV']:.0f} mV")
    ax[1].plot([], [], color=INK2, lw=1.0, alpha=0.6, label="thin: C/10 (both fit the data)")
    ax[1].plot([], [], color=INK2, lw=2.0, label="thick: predicted 1 C")
    ax[1].set(xlabel="capacity [mAh per g SPAN]", ylabel="cell voltage [V]", ylim=(0.95, 2.5),
              title="(b) same C/10 fit, two different 1 C predictions")
    ax[1].legend(fontsize=7.6, loc="lower left")
    save(fig, "r6_span500.png")


# =============================================================================================== chemistry
def c1_rates():
    d = json.loads((RES / "lispan" / "span_chem" / "rate_analysis.json").read_text())
    z = np.load(RES / "lispan" / "span_chem" / "rate_analysis.npz")
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.5), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    for cr, col in zip((0.05, 0.1, 0.2, 1.0), (C3, C1, C4, C2)):
        ax[0].plot(z[f"rate_{cr:g}_Q"], z[f"rate_{cr:g}_V"], color=col, lw=1.8, label=f"{cr:g} C")
        qe, ve = _dig(f"fig4b_exp_{cr:g}C.csv")
        ax[0].plot(qe, ve, "o", ms=4, mfc="none", mec=col, mew=1.0)
    ax[0].plot([], [], "o", ms=4, mfc="none", mec=INK2, label="measured (Simanjuntak Fig. 4b)")
    ax[0].set(xlabel="capacity [mAh per g sulfur]", ylabel="cell voltage [V]", ylim=(0.95, 2.45),
              title="(a) model calibrated by the PINN")
    ax[0].legend(fontsize=7.5, loc="lower left")
    v = d["variants_1C"]
    total = v["calibrated"]["loss_vs_0.05C_mV"]
    parts = [("contact\nresistance", total - v["no contact resistance"]["loss_vs_0.05C_mV"], C2),
             ("reaction\nkinetics", total - v["fast kinetics (k0 x100)"]["loss_vs_0.05C_mV"], C1),
             ("electrolyte\ntransport", total - v["ideal electrolyte (x100 transport)"]["loss_vs_0.05C_mV"], C3)]
    ax[1].bar(range(3), [p[1] for p in parts], color=[p[2] for p in parts], width=0.55)
    for i, p in enumerate(parts):
        ax[1].text(i, p[1] + 6, f"{p[1]:.0f} mV", ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([p[0] for p in parts], fontsize=8.5)
    ax[1].set(ylabel="voltage lost at 1 C [mV]", title="(b) where the 1 C voltage goes")
    ax[1].set_ylim(0, max(p[1] for p in parts) * 1.2)
    rates = (0.05, 0.1, 0.2, 1.0)
    left = [100 * d["rates"][f"{r:g}"]["end_fraction_S4_S3_S2"][2] for r in rates]
    ax[2].bar(range(4), left, color=[C3, C1, C4, C2], width=0.55)
    for i, l in enumerate(left):
        ax[2].text(i, l + 1.2, f"{l:.0f} %", ha="center", va="bottom", fontsize=9)
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels([f"{r:g} C" for r in rates])
    ax[2].set(ylabel="chains stuck at PAN-S$_2$Li [%]", title="(c) unfinished step 3 at 1.0 V", ylim=(0, 75))
    fig.subplots_adjust(wspace=0.42)
    save(fig, "c1_rates.png")
    return {p[0].replace("\n", " "): p[1] for p in parts} | {"total_1C_mV": total, "left_S2_pct": dict(zip(map(str, rates), left))}


def _chem(label):
    if label == "s500_c3":       # the cycle-3 calibration of the paper case (scripts/lispan_span500.py)
        f = json.loads((RES / "lispan" / "span500" / "fit_cycle3.json").read_text())["fits"]["paper"]
        w = json.loads((RES / "lispan" / "span500" / "fit_cycle3_pinnwindow.json").read_text())
        return {"estimates": f["estimates"], "Z_CC_Ohm_m2": 1.93 * 1.131e-4, "rms_mV": f["rms_mV"],
                "window": {"estimates": w["estimates"], "rms_mV": w["rms_mV"]}}
    f = RES / "lispan" / "span_chem" / f"fit_{label}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _curve(label):
    if label == "s500_c3":
        import lispan_span500 as sp5
        q, v = sp5.load_discharge(3)
        return q, v
    z = np.load(RES / "lispan" / "span_chem" / f"curve_{label}.npz")
    return z["q_data"], z["v_data"]


def _pinn_chem(label):
    if label == "s500_c3":
        runs = sorted(RUNS.glob("lispan_inv_span500_c3_paper_wS_b5_*"))
        return json.loads((runs[-1] / "inverse_result.json").read_text())
    runs = sorted(RUNS.glob(f"lispan_chem_{label}_inv_*"))
    runs = [r for r in runs if (r / "inverse_result.json").exists()]
    return json.loads((runs[-1] / "inverse_result.json").read_text()) if runs else None


def _model_curve(fit, pinn=None):
    import lispan_span500 as sp5
    base = replace(sp5.base_params(1.0), Z_CC=fit["Z_CC_Ohm_m2"])
    est = dict(fit["estimates"])
    if pinn:
        est.update(pinn["estimates"])
    Q, V, _, _ = sp5.simulate(est, base)
    return Q, V


def _span_sets(sets, title_a, name, labels, xlabel_b):
    """Data + PINN-calibrated model for several data sets, and the fitted equilibrium potentials."""
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    cols = (C1, C2, C3)
    import lispan_span500 as sp5
    U0n = np.array(sp5.base_params(1.0).U0)
    rows = []
    for i, label in enumerate(sets):
        fit, pinn = _chem(label), _pinn_chem(label)
        if fit is None:
            continue
        qd, vd = _curve(label)
        ax[0].plot(qd, vd, color=cols[i], lw=4, alpha=0.35)
        Q, V = _model_curve(fit, pinn)
        ax[0].plot(Q, V, "--", color=cols[i], lw=1.4, label=labels[i])
        src = pinn["estimates"] if pinn else fit["window"]["estimates"]
        U0 = U0n + np.array([src["U0_1"], src["U0_2"], src["U0_3"]])
        rows.append((labels[i], U0, fit["estimates"]["w_S"], "PINN" if pinn else "FV"))
    ax[0].plot([], [], color=INK2, lw=4, alpha=0.35, label="measured (thick)")
    ax[0].plot([], [], "--", color=INK2, lw=1.4, label="model at the PINN's parameters")
    ax[0].set(xlabel="capacity [mAh per g SPAN]", ylabel="cell voltage [V]", ylim=(0.95, 2.4), title=title_a)
    ax[0].legend(fontsize=7.3, loc="lower left")
    w = 0.25
    for i, (lab, U0, wS, how) in enumerate(rows):
        ax[1].bar(np.arange(3) + (i - 1) * w, U0, width=w * 0.92, color=cols[i], label=lab)
    ax[1].scatter(np.arange(3), U0n, marker="_", s=500, color=INK, zorder=5, label="Simanjuntak (carbonate)")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["step 1", "step 2", "step 3"])
    ax[1].set(ylabel="equilibrium potential U$_0$ [V]", ylim=(1.0, 2.75), title="(b) where each step happens")
    ax[1].legend(fontsize=7, loc="upper right")
    ax[2].bar(range(len(rows)), [100 * r[2] for r in rows], color=cols[:len(rows)], width=0.55)
    for i, r in enumerate(rows):
        ax[2].text(i, 100 * r[2] + 1, f"{100 * r[2]:.0f} %", ha="center", va="bottom", fontsize=9)
    ax[2].set_xticks(range(len(rows))); ax[2].set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax[2].set(ylabel="effective sulfur [wt% of SPAN]", title=xlabel_b, ylim=(0, 70))
    save(fig, name)
    return [(r[0], r[1].tolist(), r[2], r[3]) for r in rows]


def c2_first_cycle():
    return _span_sets(["s500_c1", "s500_c2", "s500_c3"], "(a) SPAN500, cycles 1-3 (C/10)", "c2_first_cycle.png",
                           ["cycle 1", "cycle 2", "cycle 3"], "(c) sulfur the model 'sees'")


def c3_synthesis():
    return _span_sets(["si300_c2", "si320_c2", "si500_c2"], "(a) cycle 2 at C/10, three SPANs", "c3_synthesis.png",
                           ["SPAN300", "SPAN320", "SPAN500"], "(c) sulfur the model 'sees'")


def main(which):
    funcs = {k: v for k, v in globals().items() if k[:2] in ("a1", "a2", "a3", "a4", "a5", "r1", "r2", "r3", "r4", "r6", "c1", "c2", "c3")
             and callable(v)}
    summary = {}
    for name, f in funcs.items():
        if not which or any(name.startswith(w) for w in which):
            res = f()
            if res is not None:
                summary[name] = res
    if summary:
        p = OUT / "figure_numbers.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(summary)
        p.write_text(json.dumps(old, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
