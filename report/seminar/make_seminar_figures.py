"""Figures for the group-seminar deck (2026-10-30) that are not in report/spandb/:

    python report/seminar/make_seminar_figures.py

net_accuracy.png   network vs FV solver on the evaluation discharges, by C-rate: v2 (0.1-1 C) and v3 (0.08-3 C)
two_curve_b.png    crop of report/phase3/figures/p3b_two_curve.png, panel b (network fit vs FV optimum)
two_curve_c.png    crop of the same figure, panel c (Hao 0.2 + 1 C, prediction of 0.5 C)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
INK, INK2, GRID, BG = "#1B2A3A", "#52514E", "#E4E2DC", "#FFFFFF"
BLUE, ORANGE, GREY = "#2A78D6", "#EB6834", "#9A988F"
plt.rcParams.update({
    "font.size": 16, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight", "axes.axisbelow": True,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK})
SELECT = {"p0", "p1", "p2", "p3", "p4"}          # chemistries used to pick the checkpoint; the rest are test cases


def net_accuracy():
    runs = [("chem_v2_norar_s0", "network v2 (trained 0.1-1 C)", BLUE, ["0.1C", "0.3C", "1C"]),
            ("chem_v3_s1", "network v3 (trained 0.08-3 C)", ORANGE, ["0.1C", "0.3C", "1C", "3C"])]
    rates = ["0.1C", "0.3C", "1C", "3C"]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    rng = np.random.default_rng(1)
    for j, (run, lab, col, rr) in enumerate(runs):
        m = json.loads((ROOT / "results" / "lispan_chem_runs" / run / "final_metrics.json").read_text())
        for k, r in enumerate(rates):
            if r not in rr:
                continue
            cases = sorted(c.split("@")[1] for c in m if c.startswith("V_rmse_mV@") and c.endswith("_" + r))
            ys = [m[f"V_rmse_mV@{c}"] for c in cases]
            test = [c.split("_")[0] not in SELECT for c in cases]
            xs = k + (j - 0.5) * 0.36 + rng.uniform(-0.09, 0.09, len(ys))
            for x, y, t in zip(xs, ys, test):
                ax.plot(x, y, "o", ms=9, color=col, mfc=col if t else "white", mew=1.8)
            ax.plot([k + (j - 0.5) * 0.36 - 0.14, k + (j - 0.5) * 0.36 + 0.14], [np.median(ys)] * 2, "-", color=INK, lw=2.6)
        ax.plot([], [], "o", ms=9, color=col, label=lab)
    ax.plot([], [], "o", ms=9, color=INK2, mfc="white", mew=1.8, label="chemistries used to pick the checkpoint")
    ax.plot([], [], "-", color=INK, lw=2.6, label="median")
    ax.axhline(5, color=GREY, ls=":", lw=1.6)
    ax.text(-0.45, 5.5, "5 mV: typical measurement noise", ha="left", fontsize=13, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.15, 60)
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20, 50], ["0.2", "0.5", "1", "2", "5", "10", "20", "50"])
    ax.set_xticks(range(4), ["0.1 C", "0.3 C\n(not a training rate)", "1 C", "3 C"])
    ax.set_ylabel("rms error vs FV solver [mV]")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=13, ncol=2)
    fig.savefig(OUT / "net_accuracy.png")
    plt.close(fig)


def crops():
    src = Image.open(ROOT / "report" / "phase3" / "figures" / "p3b_two_curve.png")
    src.crop((0, 594, src.size[0], 1090)).save(OUT / "two_curve_b.png")
    src.crop((0, 1150, src.size[0], src.size[1])).save(OUT / "two_curve_c.png")


if __name__ == "__main__":
    net_accuracy()
    crops()
