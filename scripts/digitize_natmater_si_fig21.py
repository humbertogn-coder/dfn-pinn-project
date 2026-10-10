"""Digitize the discharge curves of Supplementary Fig. 21 of Wang et al., Nat. Mater. 25, 791 (2026).

SPAN300 / SPAN320 / SPAN500, cycles 1 (panel a) and 2 (panel b), C/10, Li || SPAN coin cells.
Input: the raster image of the figure extracted from the SI PDF (pdfimages, page 23, image + soft mask
composited on white). The curves are classified by their fill colour and each discharge branch is tracked
from its end point (1.0 V) towards zero capacity by continuity, which separates it from the charge branch of
the same colour. Axis calibration from the tick marks (detected, listed below).

Output: data/span_natmater2026/digitized/si_fig21_<material>_c<cycle>.csv (capacity mAh/g_SPAN, voltage V)
and a check figure. Pixel size: 1.65 mAh/g x 5.7 mV.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "span_natmater2026" / "digitized"

# tick calibration (pixel -> value), measured on the 1341 x 572 raster
PANELS = {
    1: {"x": (152.0, 0.0, 637.0, 800.0), "y": (61.5, 3.2, 413.0, 1.2), "xlim": (122, 666)},
    2: {"x": (822.0, 0.0, 1306.5, 800.0), "y": (61.0, 3.2, 412.5, 1.2), "xlim": (792, 1336)},
}
COLORS = {"SPAN300": (253, 198, 171), "SPAN320": (255, 146, 91), "SPAN500": (235, 84, 88)}
TOL = 38.0


def lin(p, a):
    p0, v0, p1, v1 = a
    return v0 + (np.asarray(p, float) - p0) * (v1 - v0) / (p1 - p0)


def clusters(col_mask):
    ys = np.flatnonzero(col_mask)
    if ys.size == 0:
        return []
    out, start = [], ys[0]
    for i in range(1, ys.size + 1):
        if i == ys.size or ys[i] != ys[i - 1] + 1:
            out.append((start, ys[i - 1]))
            if i < ys.size:
                start = ys[i]
    return [((a + b) / 2.0, b - a + 1) for a, b in out]


def track(mask, xlim, y_end, legend_box):
    x0, x1 = xlim
    # rightmost column with pixels near the end-of-discharge voltage
    xs = [x for x in range(x1, x0, -1) if any(abs(c - y_end) < 6 for c, _ in clusters(mask[:, x]))]
    if not xs:
        raise RuntimeError("end of discharge not found")
    x = xs[0]
    pts = [(x, min(clusters(mask[:, x]), key=lambda c: abs(c[0] - y_end))[0])]
    miss = 0
    for x in range(pts[0][0] - 1, x0, -1):
        if len(pts) >= 4:
            (xa, ya), (xb, yb) = pts[-4], pts[-1]
            pred = yb + (yb - ya) / (xb - xa) * (x - xb)
        else:
            pred = pts[-1][1]
        # a discharge branch tracked towards Q = 0 can only stay level or rise in voltage (pixel y falls);
        # this separates it from the charge branch of the same colour where the two cross
        cands = [c for c, n in clusters(mask[:, x]) if n <= 12 and c <= pts[-1][1] + 1.5
                 and not (legend_box[0] <= x <= legend_box[1] and legend_box[2] <= c <= legend_box[3])]
        if not cands:
            miss += 1
            if miss > 25:
                break
            continue
        best = min(cands, key=lambda c: abs(c - pred))
        if abs(best - pred) > 6 + 2 * miss:
            miss += 1
            if miss > 25:
                break
            continue
        miss = 0
        pts.append((x, best))
    return np.array(pts[::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="composited raster of SI Fig. 21")
    args = ap.parse_args()
    a = np.asarray(Image.open(args.image).convert("RGB")).astype(float)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for cyc, P in PANELS.items():
        ax = axs[cyc - 1]
        y_end = lin(1.0, (P["y"][1], P["y"][0], P["y"][3], P["y"][2]))  # value -> pixel
        legend_box = (P["xlim"][0] + 60, P["xlim"][0] + 260, 70, 175)
        for name, rgb in COLORS.items():
            d = np.sqrt(((a - np.array(rgb)) ** 2).sum(2))
            mask = d < TOL
            pts = track(mask, P["xlim"], y_end, legend_box)
            q = lin(pts[:, 0], P["x"])
            v = lin(pts[:, 1], P["y"])
            keep = q >= 3.0  # drop the vertical initial drop at Q ~ 0
            q, v = q[keep], v[keep]
            # remove the few pixels taken from the crossing charge branch: deviation > 10 mV from a running median
            med = np.array([np.median(v[max(0, i - 10):i + 11]) for i in range(v.size)])
            ok = np.abs(v - med) < 0.010
            q, v = q[ok], v[ok]
            if cyc == 2:   # a second-cycle discharge never rises: drop points > 3 mV above the running minimum
                keep, vmin = np.ones(v.size, bool), v[0]          # (left over from the crossing charge branch)
                for i in range(1, v.size):
                    if v[i] > vmin + 0.003:
                        keep[i] = False
                    else:
                        vmin = min(vmin, v[i])
                q, v = q[keep], v[keep]
            np.savetxt(OUT / f"si_fig21_{name}_c{cyc}.csv", np.c_[q, v], delimiter=",",
                       header="capacity_mAh_per_g_SPAN,voltage_V", comments="", fmt="%.4f")
            summary[f"{name}_c{cyc}"] = {"n": int(q.size), "Q_end": float(q.max()), "V_first": float(v[0]),
                                         "V_at_100": float(np.interp(100, q, v)), "V_at_300": float(np.interp(300, q, v)) if q.max() > 300 else None}
            ax.plot(q, v, ".", ms=1.5, label=name)
        ax.set(xlabel="capacity [mAh/g SPAN]", ylabel="V", title=f"cycle {cyc} (digitized)", ylim=(0.9, 2.3))
        ax.legend()
    fig.savefig(OUT / "si_fig21_check.png", dpi=130)
    (OUT / "si_fig21_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
