"""Figure of geometry step 1: which geometry inputs remove the bias (docs/GEOMETRY.md).

    python report/geometry/make_reduction_figure.py

Input : results/lispan/geometry/reduction_p0.json (scripts/lispan_geometry_reduction.py, chemistry p0)
Output: report/geometry/figures/geo_reduction.png
  a  bias of the step-2 potential dU0_2 [mV]   b  bias of the step-1 rate constant k0_1 [%]
  per data geometry, for three networks: v4 (fixed glass-fibre cell), a network that knows only the loading (f fixed
  at 3, polyolefin separator) and v5 (thickness and f known, polyolefin separator: zero for polyolefin cells).
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
BLUE, ORANGE, AQUA = "#2A78D6", "#EB6834", "#1BAF7A"
plt.rcParams.update({"font.size": 14, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
                     "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.titlesize": 15, "text.color": INK})
ORDER = ["coin_33_f3", "coin_67_f3", "coin_100_f3", "coin_50_f2", "coin_150_f2", "coin_40_f5", "coin_80_f5",
         "pe_67_f3", "gf_67_f3"]
LABEL = {"coin_33_f3": "33 um\nf 3", "coin_67_f3": "67 um\nf 3", "coin_100_f3": "100 um\nf 3", "coin_50_f2": "50 um\nf 2",
         "coin_150_f2": "150 um\nf 2", "coin_40_f5": "40 um\nf 5", "coin_80_f5": "80 um\nf 5",
         "pe_67_f3": "PE 20 um\nsep.", "gf_67_f3": "glass\nfibre sep."}
SERIES = [("ref", "v4 network (fixed test cell)", BLUE), ("load_f3", "loading only (f fixed at 3)", ORANGE),
          ("v5", "v5: thickness and f as inputs", AQUA)]


def main():
    d = json.loads((ROOT / "results" / "lispan" / "geometry" / "reduction_p0.json").read_text())
    data = d["data_geometries"]
    by = {(r["data"], r["reduction"]): r["bias"] for r in d["results"] if "bias" in r}
    zero = {"dU0_2": 0.0, "k0_1": 0.0}
    for g in ORDER:
        # v5 knows (L_cat, f); its separator is the polyolefin one: zero bias for polyolefin data, sep_fixed otherwise
        by[(g, "v5")] = by.get((g, "sep_fixed"), zero)
        if (g, "load_f3") not in by and abs(data[g][1] - 3.0) < 1e-9:
            by[(g, "load_f3")] = by.get((g, "sep_fixed"), zero)
    gs = [g for g in ORDER if (g, "ref") in by]
    x = np.arange(len(gs))
    fig, axs = plt.subplots(1, 2, figsize=(19, 6.2))
    for ax, key, unit, ttl in ((axs[0], "dU0_2", "bias [mV]", "a  Step-2 potential dU0_2"),
                               (axs[1], "k0_1", "bias [%]", "b  Step-1 rate constant k0_1")):
        for j, (red, lab, col) in enumerate(SERIES):
            vals = [by.get((g, red), {}).get(key, np.nan) for g in gs]
            ax.bar(x + (j - 1) * 0.27, vals, 0.25, color=col, label=lab)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(x, [LABEL[g] for g in gs], fontsize=12)
        ax.set_ylabel(unit)
        ax.set_title(ttl)
    axs[0].legend(loc="lower left", fontsize=12)
    fig.text(0.01, -0.04, "Linearized FV fits (chemistry p0, 0.2 + 2 C) of coin-cell-like data; f = SPAN scale factor "
             "(porosity 0.87 / 0.75 / 0.62 / 0.49 / 0.37 for f = 1-5).  v5 is exact for polyolefin cells by construction.",
             fontsize=12, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "geo_reduction.png")
    print("written", OUT / "geo_reduction.png")


if __name__ == "__main__":
    main()
