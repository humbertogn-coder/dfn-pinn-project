"""Figures for the overview slide deck (large type, deck background), from the Phase 2 fits.

    python report/overview/make_deck_figures.py

d1_schematic.png      why two curves: voltage loss vs current, one curve (fan) vs two curves (pinned)
d2_errors.png         prediction error: slowest curve only vs slowest + fastest (Chen 2019 x2, Hao 2025)
d3_fit_vs_nofit.png   the model fits Chen 2019 (ether) / does not fit Hao 2025 (flagged)
d4_resistance.png     fitted resistance (real-cell equivalent) of each Hao calibration vs impedance
d5_speed.png          time per task: classic solver (measured) vs parametric PINN (estimated)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results" / "lispan" / "literature_rates"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
BG, INK, INK2, GRID = "#F7F6F2", "#1B2A3A", "#4A5568", "#DEDBD2"
BLUE, ORANGE, GREY, AMBER = "#2A78D6", "#E0622C", "#A9A69E", "#B86E00"
plt.rcParams.update({
    "font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.6, "legend.frameon": False, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 19, "axes.titlelocation": "left",
    "axes.axisbelow": True, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK})


def d1():
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
    x = np.linspace(0.1, 1.15, 200)
    p0, p1 = (0.1, 0.08), (0.8, 0.55)
    ax = axs[0]
    for amp, tau in ((0.14, 0.3), (0.5, 0.5), (1.3, 1.2)):
        y = p0[1] + amp * (1 - np.exp(-(x - p0[0]) / tau))
        ax.plot(x, y, color=GREY, lw=3)
    ax.axvline(p1[0], color=INK2, ls="--", lw=1.4)
    ax.plot(*p0, "o", ms=16, color=BLUE, zorder=5)
    ax.text(0.14, 0.93, "all fit the slow curve,\nbut disagree at the fast rate", transform=ax.transAxes, va="top",
            fontsize=16)
    ax.set_title("One curve (slowest rate)")
    ax2 = axs[1]
    lin = p0[1] + (x - p0[0]) * (p1[1] - p0[1]) / (p1[0] - p0[0])
    for curv in (0.0, 0.35, 0.7):
        bump = curv * (x - p0[0]) * (p1[0] - x)
        y = lin + bump
        beyond = x > p1[0]
        y[beyond] = p1[1] + (x[beyond] - p1[0]) * ((p1[1] - p0[1]) / (p1[0] - p0[0]) - 0.45 * curv)
        ax2.plot(x, y, color=GREY, lw=3)
    ax2.plot(*p0, "o", ms=16, color=BLUE, zorder=5)
    ax2.plot(*p1, "o", ms=16, color=BLUE, zorder=5)
    ax2.text(0.42, 0.42, "in between:\npinned down", transform=ax2.transAxes, fontsize=16, ha="center")
    ax2.text(0.97, 0.97, "beyond the fastest:\nneeds a 3rd curve", transform=ax2.transAxes, fontsize=16,
             ha="right", va="top")
    ax2.set_title("Two curves (slowest + fastest)")
    for a in axs:
        a.set_xticks([p0[0], p1[0]], ["slow", "fast"])
        a.set_yticks([])
        a.set_xlim(0, 1.2)
        a.set_ylim(0, 1.0)
        a.set_xlabel("current (C-rate)")
        a.set_ylabel("voltage loss")
        a.grid(False)
    fig.tight_layout(w_pad=4)
    fig.savefig(OUT / "d1_schematic.png")
    plt.close(fig)


def d2():
    rows = [("Plain SPAN (Hao 2025)", 470, 70), ("Se-SPAN, carbonate (Chen 2019)", 433, 50),
            ("Se-SPAN, ether (Chen 2019)", 305, 41)]
    fig, ax = plt.subplots(figsize=(14, 5.2))
    for i, (name, one, two) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([two, one], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.plot(one, y, "o", ms=22, color=GREY, zorder=3)
        ax.plot(two, y, "o", ms=22, color=BLUE, zorder=3)
        ax.text(one, y + 0.22, f"{one}", ha="center", fontsize=18, fontweight="bold")
        ax.text(two, y + 0.22, f"{two}", ha="center", fontsize=18, fontweight="bold", color=BLUE)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlim(0, 520)
    ax.set_xlabel("error of predicted curves, mV (rms)")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", ms=14, color=GREY, label="calibrated on the slowest curve only (error at the fastest rate)")
    ax.plot([], [], "o", ms=14, color=BLUE, label="calibrated on slowest + fastest (worst rate in between)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "d2_errors.png")
    plt.close(fig)


def curves(ax, ds, fit, unit, title, note):
    rep = json.loads((RES / f"{ds}_fit{fit}.json").read_text())
    z = np.load(RES / f"{ds}_fit{fit}.npz")
    for r in sorted(float(k) for k in rep["per_rate"]):
        k = f"{r:g}"
        role = rep["per_rate"][k]["role"]
        ax.plot(z[f"qd_{k}"], z[f"vd_{k}"], color=GREY, lw=6, alpha=0.6)
        ax.plot(z[f"Q_{k}"], z[f"V_{k}"], color=BLUE if role == "fit" else ORANGE, lw=2.2,
                ls="-" if role == "fit" else "--")
    ax.plot([], [], color=GREY, lw=6, alpha=0.6, label="measured (published)")
    ax.plot([], [], color=BLUE, lw=2.2, label=f"model, calibrated on {fit.replace('+', ' + ')} {unit}")
    ax.plot([], [], color=ORANGE, lw=2.2, ls="--", label="model, predicted rates")
    ax.set(xlabel="capacity [mAh/g]", ylabel="voltage [V]", ylim=(0.95, 2.4), xlim=(0, None))
    ax.set_title(title)
    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=16,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=GRID))


def d3():
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.4))
    curves(axs[0], "chen2019_ether", "0.4+10", "A/g", "Fits: Se-SPAN, ether (Chen 2019)",
           "rates in between predicted\nwithin 33-41 mV")
    curves(axs[1], "hao2025_SPAN", "0.1+2", "C", "Flagged: plain SPAN (Hao 2025)",
           "cell loses 20 % capacity at 2 C,\nmodel only 10 %")
    h, _ = axs[0].get_legend_handles_labels()
    fig.legend(h, ["measured (published)", "model, at the two calibration rates", "model, predicted rates"],
               loc="lower center", ncol=3, fontsize=16, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=3, rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "d3_fit_vs_nofit.png")
    plt.close(fig)


def d4():
    rows = [("0.1 C only", False, 165, 220), ("0.1 + 0.5 C", False, 0.14, 0.19), ("0.1 + 2 C", True, 14, 19),
            ("0.2 + 2 C", True, 31, 41), ("all five rates", True, 29, 39)]
    fig, ax = plt.subplots(figsize=(14, 5.6))
    for i, (name, fast, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=BLUE if fast else GREY, lw=16, solid_capstyle="round")
        lab = f"{lo:g}-{hi:g}"
        if name == "0.1 C only" or name == "0.1 + 2 C":
            ax.text(lo / 1.18, y, lab, ha="right", va="center", fontsize=17, fontweight="bold")
        else:
            ax.text(hi * 1.18, y, lab, ha="left", va="center", fontsize=17, fontweight="bold")
    for v, lab, yy, ha in ((20, "impedance after 10 cycles: ~20", 4.55, "right"),
                           (310, "impedance, fresh: ~310", 4.95, "left")):
        ax.axvline(v, color=INK2, ls="--", lw=1.6)
        ax.text(v / 1.08 if ha == "right" else v * 1.08, yy, lab, fontsize=15, color=INK2, va="center", ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 2000)
    ax.set_xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
    ax.set_ylim(-0.6, 5.25)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("fitted resistance as a real-cell value, Ohm cm$^2$ (log scale)")
    fig.tight_layout()
    fig.savefig(OUT / "d4_resistance.png")
    plt.close(fig)


def d5():
    rows = [("One discharge curve", 5, 0.06), ("One calibration\n(~140 curves)", 864, 8.4),
            ("Error bars\n(~30,000 curves)", 150000, 1800)]

    def fmt(s):
        return f"{s:g} s" if s < 60 else (f"{round(s / 60)} min" if s < 3600 else f"{round(s / 3600)} h")

    fig, ax = plt.subplots(figsize=(14, 5.4))
    for i, (name, solver, pinn) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([pinn, solver], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.plot(solver, y, "o", ms=22, color=GREY, zorder=3)
        ax.plot(pinn, y, "o", ms=22, color=BLUE, zorder=3)
        ax.text(solver, y + 0.25, fmt(solver), ha="center", fontsize=18, fontweight="bold")
        ax.text(pinn, y + 0.25, fmt(pinn), ha="center", fontsize=18, fontweight="bold", color=BLUE)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 4e5)
    ax.set_xticks([0.1, 1, 60, 3600, 86400], ["0.1 s", "1 s", "1 min", "1 h", "1 day"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", ms=14, color=GREY, label="classic solver (measured)")
    ax.plot([], [], "o", ms=14, color=BLUE, label="parametric PINN (estimated)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=16)
    ax.set_xlabel("time on one CPU core (log scale)")
    fig.tight_layout()
    fig.savefig(OUT / "d5_speed.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (d1, d2, d3, d4, d5):
        f()
        print("wrote", f.__name__)
