"""Figures for the project overview (doc + deck for the advisor), from the Phase 2 fits.

    python report/overview/make_figures.py

o1_fit_vs_nofit.png  left: Se-SPAN in ether (Chen 2019) calibrated on 0.4 + 10 A/g, the model fits and predicts the
                     rates in between; right: plain SPAN (Hao 2025) calibrated on 0.1 + 2 C, the model cannot follow
                     the capacity drop at 2 C, which is what the tool flags
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results" / "lispan" / "literature_rates"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left", "axes.axisbelow": True})


def panel(ax, ds, fit, unit, title, note):
    rep = json.loads((RES / f"{ds}_fit{fit}.json").read_text())
    z = __import__("numpy").load(RES / f"{ds}_fit{fit}.npz")
    rates = sorted(float(k) for k in rep["per_rate"])
    shades = [0.35, 0.5, 0.65, 0.8, 0.9, 1.0][-len(rates):]
    for a, r in zip(shades, rates):
        k = f"{r:g}"
        role = rep["per_rate"][k]["role"]
        col = BLUE if role == "fit" else ORANGE
        ax.plot(z[f"qd_{k}"], z[f"vd_{k}"], color="#9a9893", lw=4.0, alpha=0.45)
        ax.plot(z[f"Q_{k}"], z[f"V_{k}"], color=col, lw=1.5, ls="-" if role == "fit" else "--", alpha=a if role != "fit" else 1)
    ax.plot([], [], color="#9a9893", lw=4.0, alpha=0.45, label="measured (published)")
    ax.plot([], [], color=BLUE, lw=1.5, label=f"model, calibrated on {fit.replace('+', ' + ')} {unit}")
    ax.plot([], [], color=ORANGE, lw=1.5, ls="--", label="model, predicted rates")
    ax.set(xlabel="capacity [mAh/g]", ylabel="voltage [V]", ylim=(0.95, 2.4), xlim=(0, None))
    ax.set_title(title)
    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
    ax.legend(fontsize=9, loc="lower left")


def o1():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    panel(axs[0], "chen2019_ether", "0.4+10", "A/g", "The model fits this cell (Se-SPAN, ether; Chen 2019)",
          "rates in between predicted\nwithin 33-41 mV")
    panel(axs[1], "hao2025_SPAN", "0.1+2", "C", "The model does not fit this cell (plain SPAN; Hao 2025)",
          "the cell loses 20 % capacity at 2 C,\nthe model only 10 %: flagged")
    fig.tight_layout()
    fig.savefig(OUT / "o1_fit_vs_nofit.png")
    plt.close(fig)
    print("wrote o1_fit_vs_nofit.png")


if __name__ == "__main__":
    o1()
