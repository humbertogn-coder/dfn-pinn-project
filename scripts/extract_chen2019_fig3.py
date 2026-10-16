"""Extract the rate-test discharge curves of Chen et al., Nat. Commun. 10, 1021 (2019), Fig. 3b (ether electrolyte)
and Fig. 3d (carbonate electrolyte), Se0.06SPAN cathodes, from the VECTOR paths of the article PDF (no digitization
error: the curves are stored as polylines; axis calibration from the tick marks, listed below).

Curve colours -> current density (legend "10, 6, 4, 2, 1, 0.4, 0.2 A g-1" from left to right):
  black 0.2 (FIRST cycle of the cell: activation plateau, capacity 1690 / 1500 mAh/g; kept but flagged),
  red 0.4, blue 1, magenta 2, green 4, orange 6, violet 10 A/g (per g of S + Se, the paper's basis).
Each discharge branch may be split into several paths; the pieces of one colour that start above 1.0 V and end at
1.0 V (voltage decreasing with capacity) are joined.

Output: data/literature_span/chen2019/{ether,carbonate}_{rate}Ag.csv (capacity mAh/g_(S+Se), voltage V) and
chen2019_fig3_check.png.  Usage:  python scripts/extract_chen2019_fig3.py <path to s41467-019-08818-6.pdf>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "literature_span" / "chen2019"
COLOURS = {"(0.0, 0.0, 0.0)": 0.2, "(1.0, 0.0, 0.0)": 0.4, "(0.0, 0.0, 1.0)": 1.0, "(1.0, 0.0, 1.0)": 2.0,
           "(0.0, 0.50195997, 0.0)": 4.0, "(1.0, 0.50195997, 0.0)": 6.0, "(0.50195997, 0.0, 1.0)": 10.0}
# tick marks (pdfplumber page coordinates, points, y measured from the top of page 5)
PANELS = {"ether": {"x0": 142.64, "x2000": 266.31, "y32": 188.96, "y08": 274.37, "box": (142.0, 188.9, 266.4, 281.5)},
          "carbonate": {"x0": 338.23, "x2000": 462.52, "y32": 188.99, "y08": 274.40, "box": (338.2, 188.9, 462.6, 281.5)}}


def main():
    pdf = pdfplumber.open(sys.argv[1])
    pg = pdf.pages[4]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, P) in zip(axs, PANELS.items()):
        bx = P["box"]
        pieces = {}
        for o in pg.curves + pg.lines:
            rate = COLOURS.get(str(o.get("stroking_color")))
            pts = np.asarray(o.get("pts", []), float)
            if rate is None or len(pts) < 5:
                continue
            if not (pts[:, 0].min() >= bx[0] - 0.5 and pts[:, 0].max() <= bx[2] + 0.5
                    and pts[:, 1].min() >= bx[1] - 0.5 and pts[:, 1].max() <= bx[3] + 0.5):
                continue
            q = (pts[:, 0] - P["x0"]) / (P["x2000"] - P["x0"]) * 2000.0
            v = 3.2 - (pts[:, 1] - P["y32"]) / (P["y08"] - P["y32"]) * 2.4
            pieces.setdefault(rate, []).append((q, v))
        for rate, segs in sorted(pieces.items()):
            # discharge pieces: voltage falls from start to end; join them in capacity order
            dis = [s for s in segs if s[1][0] > s[1][-1] + 0.05]
            dis.sort(key=lambda s: s[0].min())
            q = np.concatenate([s[0] for s in dis])
            v = np.concatenate([s[1] for s in dis])
            o = np.argsort(q, kind="stable")
            q, v = q[o], v[o]
            keep = np.r_[True, np.diff(q) > 1e-6]
            q, v = q[keep], v[keep]
            tag = f"{name}_{rate:g}Ag"
            np.savetxt(OUT / f"{tag}.csv", np.c_[q, v], delimiter=",", header="capacity_mAh_per_g_S+Se,voltage_V",
                       comments="", fmt="%.4f")
            summary[tag] = {"current_A_per_g": rate, "n": int(q.size), "Q_end": float(q.max()),
                            "V_at_100": float(np.interp(100.0, q, v)), "V_at_500": float(np.interp(500.0, q, v)),
                            "first_cycle": rate == 0.2}
            ax.plot(q, v, lw=1.2, label=f"{rate:g} A/g" + (" (1st cycle)" if rate == 0.2 else ""))
        ax.set(xlabel="capacity [mAh/g (S+Se)]", ylabel="V", title=f"Chen 2019 Fig. 3, {name} (vector extraction)",
               ylim=(0.9, 2.6), xlim=(0, 1800))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "chen2019_fig3_check.png", dpi=130)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
