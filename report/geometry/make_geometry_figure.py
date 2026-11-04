"""Figure of the geometry known-truth test (docs/GEOMETRY.md, step 0).

    python report/geometry/make_geometry_figure.py

Input : results/lispan/geometry/bias_0.2+2.json (scripts/lispan_geometry_bias.py)
Output: report/geometry/figures/geo_bias.png
  a  fit misfit and blind-prediction error per geometry (mean over 4 chemistries)
  b  bias of the step-2 potential dU0_2 and of the common shift d [mV] (bar = mean, line = range over chemistries)
  c  bias of the step-1 rate constant k0_1 and of w_S [%]
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
INK, INK2, GRID = "#1B2A3A", "#52514E", "#E4E2DC"
BLUE, ORANGE, AQUA, GREY = "#2A78D6", "#EB6834", "#1BAF7A", "#9A988F"
plt.rcParams.update({"font.size": 14, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
                     "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.titlesize": 15, "text.color": INK})
ORDER = ["ref", "celgard", "thick2x", "dense3", "coin_mid", "coin_high", "dense4"]
LABEL = {"ref": "model\ncell", "celgard": "25 um\nseparator", "thick2x": "2x thick\ncathode",
         "dense3": "dense\n1.5 mg S", "coin_mid": "coin cell\n1.5 mg S", "coin_high": "coin cell\n2.3 mg S",
         "dense4": "dense +\nglass fibre"}


def main():
    d = json.loads((ROOT / "results" / "lispan" / "geometry" / "bias_0.2+2.json").read_text())
    by = {}
    for r in d["results"]:
        if "err" in r:
            by.setdefault(r["geometry"], []).append(r)
    gs = [g for g in ORDER if g in by]
    x = np.arange(len(gs))
    stat = lambda g, f: np.array([f(r) for r in by[g]])  # noqa: E731
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.2))
    ax = axs[0]
    mis = [stat(g, lambda r: r["rms_mV"]).mean() for g in gs]
    pre = [np.mean([v["rms_mV"] for r in by[g] for v in r["predictions"].values()]) for g in gs]
    ax.bar(x - 0.2, mis, 0.38, color=BLUE, label="fit misfit")
    ax.bar(x + 0.2, pre, 0.38, color=AQUA, label="blind prediction (0.5, 1 C)")
    ax.axhline(25, color=INK2, ls=":", lw=1.4)
    ax.text(-0.4, 26.5, "25 mV: 'described' limit", fontsize=12, color=INK2)
    ax.set_ylabel("rms error [mV]")
    ax.set_title("a  The geometry error does not show in the fit")
    ax.legend(loc="upper left", fontsize=12)
    ax.set_ylim(0, 42)
    for ax, keys, unit, ttl, lim in ((axs[1], [("dU0_2", "step-2 potential dU0_2", BLUE), ("d", "common shift d", ORANGE)],
                                      "bias [mV]", "b  Potentials shift by 10-20 mV", (-30, 30)),
                                     (axs[2], [("k0_1", "step-1 rate constant k0_1", BLUE), ("w_S", "sulfur utilization w_S", ORANGE)],
                                      "bias [%]", "c  Kinetics shift, utilization does not", (-60, 100))):
        for j, (k, lab, col) in enumerate(keys):
            vals = [stat(g, lambda r: r["err"][k]) for g in gs]
            xs = x + (j - 0.5) * 0.38
            m = np.array([v.mean() for v in vals])
            ax.bar(xs, np.clip(m, *lim), 0.38, color=col, label=lab)
            for xx, v in zip(xs, vals):
                ax.plot([xx, xx], np.clip([v.min(), v.max()], *lim), color=INK, lw=1.6)
            for xx, mm in zip(xs, m):
                if mm > lim[1] or mm < lim[0]:
                    ax.text(xx, lim[1] * 0.92 if mm > 0 else lim[0] * 0.92, f"{mm:+.0f}", ha="center", fontsize=11, color=INK)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_ylim(*lim)
        ax.set_ylabel(unit)
        ax.set_title(ttl)
        ax.legend(loc="lower left", fontsize=12)
    for ax in axs:
        ax.set_xticks(x, [LABEL[g] for g in gs], fontsize=11.5)
        ax.grid(axis="x", visible=False)
        ax.axvspan(len(gs) - 1.5, len(gs) - 0.5, color=GRID, alpha=0.5, lw=0)
    fig.tight_layout(w_pad=3)
    fig.savefig(OUT / "geo_bias.png")
    print("wrote", OUT / "geo_bias.png")


if __name__ == "__main__":
    main()
