"""Figures for Phase 1 of the identifiability proposal (results/lispan/phase1).

    python report/phase1/make_figures.py

p1_measured.png   calibrate on the measured 0.1 C curve (Simanjuntak et al. 2024, Fig. 4b) under different
                  contact-resistance assumptions, predict 1 C; compare with the 0.1 + 1 C calibration
p2_ranking.png    linearized analysis: what each measurement set identifies, and the prediction uncertainty
p3_synthetic.png  the same calibrate-then-predict test on synthetic data with a known answer
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dfn_pinn.lispan.params import LiSPANParams  # noqa: E402

RES = ROOT / "results" / "lispan" / "phase1"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

C1, C2, C3, C4, C5, C6, C7, C8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DATA = "#7a7974"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.axisbelow": True})

BASE = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)
Q_PER_T = {cr: 10.0 * cr / 3.6 / (1e3 * BASE.m_S_model) for cr in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)}   # mAh/gS per s


def load(name):
    j, z = RES / f"{name}.json", RES / f"{name}.npz"
    if not j.exists():
        return None, None
    return json.loads(j.read_text()), np.load(z)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------------ p1
EXP_VARIANTS = [   # name, label, colour  (Z_CC in Ohm m2 = 0.025 x multiplier)
    ("fit_experiment_0.1C_Z0", "assume Z$_{CC}$ = 0", C2),
    ("fit_experiment_0.1C_Z0.5", "assume Z$_{CC}$ = 0.0125", C4),
    ("fit_experiment_0.1C", "Z$_{CC}$ fitted (0.1 C only)", C5),
    ("fit_experiment_0.1C_Z1.4", "assume Z$_{CC}$ = 0.035", C7),
    ("fit_experiment_0.1C_Z2", "assume Z$_{CC}$ = 0.05", C3),
    ("fit_experiment_0.1+0.2C", "calibrated on 0.1 C + 0.2 C", C1),
]


def p1_measured():
    ref, zref = load("fit_experiment_0.1+1C")
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    d01 = np.load(ROOT / "results/lispan/inverse_data/V_0.1C_experiment.npz")
    d1 = np.load(ROOT / "results/lispan/inverse_data/V_1C_experiment.npz")
    rows = []
    for name, lab, col in EXP_VARIANTS:
        r, z = load(name)
        if r is None:
            continue
        axs[0].plot(z["Q_0.1"], z["V_0.1"], color=col, lw=1.6, label=lab)
        axs[1].plot(z["Q_1"], z["V_1"], color=col, lw=1.6)
        rows.append((lab, col, r["per_rate"]["0.1C"]["rms_vs_data_mV"], r["per_rate"]["1C"]["rms_vs_data_mV"],
                     r["per_rate"]["1C"]["role"] == "fit"))
    if ref is not None:
        axs[0].plot(zref["Q_0.1"], zref["V_0.1"], color=INK, lw=1.4, ls="--", label="calibrated on 0.1 C + 1 C")
        axs[1].plot(zref["Q_1"], zref["V_1"], color=INK, lw=1.4, ls="--")
        rows.append(("calibrated on 0.1 C + 1 C", INK, ref["per_rate"]["0.1C"]["rms_vs_data_mV"],
                     ref["per_rate"]["1C"]["rms_vs_data_mV"], True))
    for ax, d, title in ((axs[0], d01, "(a) Calibration: 0.1 C, measured"),
                         (axs[1], d1, "(b) Prediction: 1 C, measured")):
        ax.plot(d["Q_mAh_gS"], d["V"], "o", ms=5.5, mfc="white", mec=DATA, mew=1.4, zorder=5, label="measured (Fig. 4b)")
        ax.set_xlabel("capacity [mAh g$^{-1}_S$]"); ax.set_ylabel("voltage [V]"); ax.set_title(title)
        ax.set_xlim(0, 1300); ax.set_ylim(0.95, 2.4)
    axs[0].legend(fontsize=7.6, loc="lower left")
    # (c) dot plot: calibration misfit vs 1 C prediction error
    ax = axs[2]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, col, e01, e1, fitted) in zip(ys, rows):
        ax.plot([e01, e1], [y, y], color=GRID, lw=2.2, zorder=1)
        ax.plot(e01, y, "o", ms=7, mfc="white", mec=col, mew=1.8, zorder=3)
        ax.plot(e1, y, "s" if fitted else "o", ms=7, color=col, zorder=3)
        ax.text(max(e1, e01) * 1.25, y, f"{e1:.0f} mV" + (" (1 C fitted)" if fitted else ""), va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    ax.set_xscale("log"); ax.set_xlim(8, 900)
    ax.set_xlabel("rms error vs measured points [mV]  (log scale)")
    ax.set_title("(c) Same fit at 0.1 C, very different 1 C")
    ax.plot([], [], "o", mfc="white", mec=INK2, mew=1.6, label="0.1 C (calibration)")
    ax.plot([], [], "o", color=INK2, label="1 C (prediction)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "p1_measured.png")


# ------------------------------------------------------------------------------------------------ p2
LABELS = {"U0_1": "U$_{0,1}$ step 1 potential", "b_1": "b$_1$ step 1 slope", "k0_1": "k$_{0,1}$ step 1 rate (reversible)",
          "b_2": "b$_2$ step 2 slope", "b_3": "b$_3$ step 3 slope",
          "k0_2*exp(F U0_2/2RT)": "k$_{0,2}$e$^{FU_{0,2}/2RT}$ step 2 (combined)",
          "k0_3*exp(F U0_3/2RT)": "k$_{0,3}$e$^{FU_{0,3}/2RT}$ step 3 (combined)",
          "k0_2": "k$_{0,2}$ alone", "U0_2": "U$_{0,2}$ alone", "k0_3": "k$_{0,3}$ alone", "U0_3": "U$_{0,3}$ alone",
          "Z_CC": "Z$_{CC}$ contact resistance", "kappa0": "$\\kappa_0$ electrolyte conductivity",
          "D_salt": "D$_{salt}$ salt diffusivity", "t_plus": "t$_+$ transference number",
          "kappa_SPAN": "$\\kappa_{SPAN}$ electrode conductivity"}
ROWS = [("Equilibrium", ["U0_1", "b_1", "b_2", "b_3"]),
        ("Kinetics", ["k0_1", "k0_2*exp(F U0_2/2RT)", "k0_3*exp(F U0_3/2RT)", "k0_2", "U0_2", "k0_3", "U0_3"]),
        ("Resistance and transport", ["Z_CC", "kappa0", "D_salt", "t_plus", "kappa_SPAN"])]


def p2_ranking(sigma="5"):
    rep = json.loads((RES / f"fisher_sigma{sigma}mV.json").read_text())
    sets = list(rep["sets"])
    names, groups = [], []
    for g, ns in ROWS:
        names += ns; groups += [g] * len(ns)
    M = np.zeros((len(names), len(sets)))
    for j, s in enumerate(sets):
        for i, n in enumerate(names):
            S = rep["sets"][s]
            M[i, j] = S["combinations"][n]["ratio"] if n in S["combinations"] else S["ratio_to_prior"][n]
    fig = plt.figure(figsize=(13.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("idf", ["#0d3b73", "#2a78d6", "#a9cdf2", "#f3f2ee"])
    im = ax.imshow(M, cmap=cmap, norm=LogNorm(vmin=0.005, vmax=1.0), aspect="auto")
    for i in range(len(names)):
        for j in range(len(sets)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}" if v >= 0.01 else f"{v:.3f}", ha="center", va="center", fontsize=7.6,
                    color="white" if v < 0.06 else INK)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels(sets, rotation=30, ha="right", fontsize=8.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([LABELS[n] for n in names], fontsize=8.4)
    ax.grid(False)
    for k in np.cumsum([len(ns) for _, ns in ROWS])[:-1]:
        ax.axhline(k - 0.5, color="white", lw=3)
    y0 = 0
    for g, ns in ROWS:
        ax.text(len(sets) - 0.4, y0 + len(ns) / 2 - 0.5, g, rotation=-90, va="center", ha="left", fontsize=8.6,
                color=INK2, weight="bold")
        y0 += len(ns)
    ax.set_title("(a) Uncertainty left after the measurement, relative to the prior\n"
                 "     (1 = nothing learned; darker = better determined)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.09)
    cb.set_label("posterior sd / prior sd", fontsize=8.5)
    # (b) prediction uncertainty at the unmeasured rates
    ax = fig.add_subplot(gs[1])
    rates = ["0.05C", "0.2C", "0.5C", "1C"]
    cols = {"0.05C": C3, "0.2C": C4, "0.5C": C7, "1C": C2}
    ys = np.arange(len(sets))[::-1]
    for k, r in enumerate(rates):
        for y, s in zip(ys, sets):
            pr = rep["sets"][s]["prediction"][r]
            v = pr["rms_mV"]
            off = (k - 1.5) * 0.17
            if pr["measured"]:
                ax.plot(v, y + off, "o", ms=6, mfc="white", mec=cols[r], mew=1.4)
            else:
                ax.plot(v, y + off, "o", ms=6.5, color=cols[r])
    for r in rates:
        ax.plot([], [], "o", color=cols[r], label=r.replace("C", " C"))
    ax.plot([], [], "o", mfc="white", mec=INK2, label="rate was measured")
    ax.set_yticks(ys); ax.set_yticklabels(sets, fontsize=8.6)
    ax.set_xscale("log"); ax.set_xlim(0.4, 400)
    ax.axvline(16, color=INK2, lw=1.0, ls=":")
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.0)
    ax.text(17, ys[0] + 0.55, "model vs real cell (16-18 mV)", fontsize=7.6, color=INK2, va="center")
    ax.set_xlabel("uncertainty of the predicted voltage [mV, rms]  (log scale)")
    ax.set_title("(b) Prediction uncertainty at each rate", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.02, f"Linearized analysis around the calibrated Simanjuntak cell; {rep['sigma_mV']:g} mV voltage noise, "
             "100 points per curve, prior: each parameter known within a factor of 2 (step potentials within 50 mV).",
             fontsize=8, color=INK2)
    save(fig, f"p2_ranking_sigma{sigma}mV.png" if sigma != "5" else "p2_ranking.png")


# ------------------------------------------------------------------------------------------------ p3
SYN = [("fit_synthetic_0.1C_Zhalf", "C/10, assume Z$_{CC}$ = ½ truth", C4),
       ("fit_synthetic_0.1C", "C/10, Z$_{CC}$ fitted", C5),
       ("fit_synthetic_0.1C_Z2x", "C/10, assume Z$_{CC}$ = 2× truth", C7),
       ("fit_synthetic_0.1+0.2C", "C/10 + C/5", C3),
       ("fit_synthetic_0.1+1C", "C/10 + 1 C", INK)]


def p3_synthetic():
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    rows = []
    _, zt = load("fit_synthetic_0.1+1C")
    if zt is not None:
        for ax, cr in ((axs[0], 0.1), (axs[1], 1.0)):
            ax.plot(zt[f"td_{cr:g}"] * Q_PER_T[cr], zt[f"Vtrue_{cr:g}"], color=DATA, lw=5, alpha=0.35, zorder=0,
                    label="truth" if cr == 0.1 else None)
    for name, lab, col in SYN:
        r, z = load(name)
        if r is None:
            continue
        ls = "--" if col == INK else "-"
        for ax, cr in ((axs[0], 0.1), (axs[1], 1.0)):
            key = f"{cr:g}"
            if f"Q_{key}" not in z:
                continue
            ax.plot(z[f"Q_{key}"], z[f"V_{key}"], color=col, lw=1.5, ls=ls, label=lab if cr == 0.1 else None)
        pr = r["per_rate"]
        rows.append((lab, col, pr["0.1C"]["rms_vs_truth_mV"], pr["1C"]["rms_vs_truth_mV"],
                     pr["1C"]["role"] == "fit"))
    axs[0].set_title("(a) Calibration: 0.1 C, synthetic (5 mV noise)")
    axs[1].set_title("(b) 1 C: prediction (fit for C/10 + 1 C)")
    for ax in axs[:2]:
        ax.set_xlabel("capacity [mAh g$^{-1}_S$]"); ax.set_ylabel("voltage [V]")
        ax.set_xlim(0, 1300); ax.set_ylim(0.95, 2.4)
    axs[0].legend(fontsize=7.6, loc="lower left")
    ax = axs[2]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, col, e01, e1, fitted) in zip(ys, rows):
        ax.plot([e01, e1], [y, y], color=GRID, lw=2.2, zorder=1)
        ax.plot(e01, y, "o", ms=7, mfc="white", mec=col, mew=1.8, zorder=3)
        ax.plot(e1, y, "o" if not fitted else "s", ms=7, color=col, zorder=3)
        ax.text(max(e1, e01) * 1.35, y, (f"{e1:.0f} mV" if e1 >= 1 else f"{e1:.1f} mV") + (" (1 C fitted)" if fitted else ""),
                va="center", fontsize=8.5)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    ax.set_xscale("log"); ax.set_xlim(0.1, 3000)
    ax.set_xlabel("rms error vs the true curve [mV]  (log scale)")
    ax.set_title("(c) Known answer: same fit, different 1 C")
    ax.plot([], [], "o", mfc="white", mec=INK2, mew=1.6, label="0.1 C (calibration)")
    ax.plot([], [], "o", color=INK2, label="1 C (prediction)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "p3_synthetic.png")


if __name__ == "__main__":
    p1_measured()
    p2_ranking("5")
    if (RES / "fisher_sigma15mV.json").exists():
        p2_ranking("15")
    p3_synthetic()
