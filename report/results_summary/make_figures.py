"""Extra figures for the results summary (the paper figures are copied from paper/figures by the build).

    python report/results_summary/make_figures.py
      -> figures/figA_workflow.png   (drawing: phases of the work)
      -> figures/figB_cost.png       (from results/paper/cost_table.json)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})


def workflow():
    phases = [
        ("1  Reference model", "rebuild the published Li-SPAN model\nas a finite-volume solver and close\nits ambiguities (closures B1-B8)"),
        ("2  Forward PINN", "a network that obeys the model:\nbuilt-in structure, residuals,\nglobal charge inventory"),
        ("3  Identifiability", "which parameters a voltage\ncurve can reveal\n(Fisher information, CRLB)"),
        ("4  Inverse, synthetic data", "8 parameters from 0.1 C + 1 C;\nLevenberg-Marquardt step;\ncompare with the ideal estimator"),
        ("5  Measured data", "Simanjuntak Fig. 4b points;\nSPAN500 C/10 discharges\n(Wang et al., Nat. Mater. 2026)"),
        ("6  Paper steps 1-5", "3 seeds from scratch, cost table,\ntraceable build, assumptions,\nreal-data case"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(0, 3); ax.set_ylim(0, 2); ax.axis("off")
    W, H = 0.84, 0.74
    for i, (title, body) in enumerate(phases):
        r, c = divmod(i, 3)
        x, y = c + 0.08, 1.12 - r * 1.0
        green = i == 5
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc="#e8f6f0" if green else "#eef4fc", ec="#1baf7a" if green else BLUE, lw=1.2))
        ax.text(x + W / 2, y + H - 0.1, title, ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
        ax.text(x + W / 2, y + H - 0.3, body, ha="center", va="top", fontsize=8.6, color=INK2, linespacing=1.4)
        if c < 2:
            ax.annotate("", xy=(c + 1.07, y + H / 2), xytext=(c + 0.93, y + H / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3))
    ax.annotate("", xy=(0.5, 0.88), xytext=(2.5, 1.10),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, connectionstyle="arc3,rad=0.0"))
    ax.text(1.5, 0.02, "Oct 5-6: phases 1-2   |   Oct 7: phases 3-5   |   Oct 8-9: phase 6 (paper steps)",
            ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "figA_workflow.png", dpi=200)
    plt.close(fig)


def cost():
    c = json.loads((ROOT / "results" / "paper" / "cost_table.json").read_text())
    fv = {f"{r['crate']:g}C": r["cpu_s"] for r in c["fv_forward"] if r["grid"] == [40, 20]}
    pt = {f"{r['crate']:g}C": r["projected_cpu_h"] * 3600 for r in c["pinn_step_timing"]}
    inv_fv = c["inverse"]["finite_volume"]["cpu_h"] * 3600
    inv_pinn = c["inverse"]["pinn"]["inverse_only_wall_h"] * 3600
    inv_pinn_total = inv_pinn + c["inverse"]["pinn"]["forward_pretraining_wall_h"] * 3600
    rows = [("one discharge, 0.1 C", fv["0.1C"], pt["0.1C"]),
            ("one discharge, 1 C", fv["1C"], pt["1C"]),
            ("8-parameter inverse", inv_fv, inv_pinn_total)]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    y = range(len(rows))
    h = 0.34
    for k, (lab, a, b) in enumerate(rows):
        ax.barh(k + h / 2 + 0.02, a, h, color=BLUE, label="finite volume" if k == 0 else None)
        ax.barh(k - h / 2 - 0.02, b, h, color=ORANGE, label="PINN" if k == 0 else None)
        ax.text(a * 1.15, k + h / 2 + 0.02, fmt(a), va="center", fontsize=8.5, color=INK)
        ax.text(b * 1.15, k - h / 2 - 0.02, fmt(b) + ("  (incl. forward pre-training)" if k == 2 else "  (training)"),
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(1, 1e6)
    ax.set_xlabel("time on one CPU core [s] (log scale)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figB_cost.png", dpi=200)
    plt.close(fig)


def fmt(s):
    return f"{s:.0f} s" if s < 120 else (f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h")


if __name__ == "__main__":
    workflow()
    cost()
    print("ok")
