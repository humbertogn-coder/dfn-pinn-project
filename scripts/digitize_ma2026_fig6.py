"""Digitize the rate-test discharge curves of Ma et al., Small 22, e74169 (2026), Fig. 6c (SPAN) and 6d (SPAN-A):
Li || SPAN coin cells, 1 M LiPF6 in EC/DMC/EMC (carbonate), ~1.9 mg S/cm2, 1.0-3.0 V, 0.1 / 0.2 / 0.3 / 0.5 / 1 / 2 C,
capacity per g of sulfur (the paper's capacity utilization: 1512.9 mAh/g = 90 %, i.e. 1 C = 1675 mA/g S).

Input: the raster of Fig. 6 extracted from the article PDF (pdfimages -f 8 -l 8 -png, 2050 x 2178 px, 300 dpi).
Each pixel is assigned to the nearest of the six legend colours (or to white / grey / black); each discharge branch is
tracked from its end point (1.0 V) towards zero capacity by continuity, with the constraint that a discharge branch
followed towards Q = 0 never falls in voltage (this separates it from the charge branch of the same colour where the
two cross). Below ~100-150 mAh/g the charge and discharge branches of all rates cross; points there are less reliable
(flagged by the tracker as gaps) and are cut at Q_MIN.

Axis calibration (pixels of the 2050 x 2178 raster, measured from the tick marks):
  panel c: x 188.5 px = 0, 110.25 px per 400 mAh/g;  y 943 px = 1.0 V, 572 px = 3.0 V
  panel d: x 853.0 px = 0, 110.25 px per 400 mAh/g;  y 945 px = 1.0 V, 575 px = 3.0 V
Pixel size: 3.6 mAh/g x 5.4 mV; lines are ~8 px wide (the centre of each run is used).

Output: data/literature_span/ma2026/{SPAN,SPAN-A}_{rate}C.csv (capacity mAh/g_S, voltage V), ma2026_fig6_check.png.
Usage:  python scripts/digitize_ma2026_fig6.py <Fig. 6 png>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "literature_span" / "ma2026"
PANELS = {"SPAN": {"x": (188.5, 0.0, 188.5 + 4.5 * 110.25, 1800.0), "y": (943.0, 1.0, 572.0, 3.0), "xlim": (190, 684),
                   "ylim": (574, 941), "legend": (540, 684, 625, 860)},
          "SPAN-A": {"x": (853.0, 0.0, 853.0 + 4.5 * 110.25, 1800.0), "y": (945.0, 1.0, 575.0, 3.0), "xlim": (855, 1348),
                     "ylim": (577, 943), "legend": (1205, 1348, 625, 860)}}
RATES = {0.1: (248, 183, 108), 0.2: (139, 185, 216), 0.3: (244, 130, 118), 0.5: (191, 189, 223),
         1.0: (249, 249, 174), 2.0: (150, 210, 200)}
OTHER = [(255, 255, 255), (0, 0, 0), (128, 128, 128), (200, 200, 200)]
Q_MIN = 60.0


def lin(p, a):
    p0, v0, p1, v1 = a
    return v0 + (np.asarray(p, float) - p0) * (v1 - v0) / (p1 - p0)


def runs(mask_col):
    ys = np.flatnonzero(mask_col)
    if ys.size == 0:
        return []
    out, s = [], ys[0]
    for i in range(1, ys.size + 1):
        if i == ys.size or ys[i] != ys[i - 1] + 1:
            out.append(((s + ys[i - 1]) / 2.0, ys[i - 1] - s + 1))
            if i < ys.size:
                s = ys[i]
    return out


def classify(a):
    pal = np.array(list(RATES.values()) + OTHER, float)
    d = ((a[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
    lab = np.argmin(d, axis=2)
    dmin = np.sqrt(np.min(d, axis=2))
    return lab, dmin


def track_from(mask, P, start):
    x0 = P["xlim"][0]
    y_lo, y_hi = P["ylim"]
    lg = P["legend"]
    pts, miss = [start], 0
    for x in range(start[0] - 1, x0, -1):
        if len(pts) >= 6:
            (xa, ya), (xb, yb) = pts[-6], pts[-1]
            pred = yb + (yb - ya) / (xb - xa) * (x - xb)
        else:
            pred = pts[-1][1]
        cands = [c + y_lo for c, n in runs(mask[y_lo:y_hi, x]) if 3 <= n <= 30]
        cands = [c for c in cands if c <= pts[-1][1] + 1.5
                 and not (lg[0] <= x <= lg[1] and lg[2] <= c <= lg[3])]
        if not cands:
            miss += 1
            if miss > 40:
                break
            continue
        best = min(cands, key=lambda c: abs(c - pred))
        if abs(best - pred) > 5 + 2 * miss:
            miss += 1
            if miss > 40:
                break
            continue
        miss = 0
        pts.append((x, best))
    return np.array(pts[::-1])


def track(mask, P):
    """Candidate end points: runs of this colour within 12 px of the 1.0 V line (outside the legend); the
    discharge branch is the candidate that can be tracked furthest towards Q = 0 (anti-aliased edges of
    other curves give short spurious tracks)."""
    x0, x1 = P["xlim"]
    y_lo, y_hi = P["ylim"]
    y_end = lin(1.0, (P["y"][1], P["y"][0], P["y"][3], P["y"][2]))
    lg = P["legend"]
    starts = []
    for x in range(x1, x0, -1):
        rr = [c + y_lo for c, n in runs(mask[y_lo:y_hi, x]) if n >= 3]
        near = [c for c in rr if abs(c - y_end) < 12 and not (lg[0] <= x <= lg[1] and lg[2] <= c <= lg[3])]
        if near and (not starts or starts[-1][0] - x > 6):
            starts.append((x, min(near, key=lambda c: abs(c - y_end))))
        if len(starts) >= 6:
            break
    if not starts:
        raise RuntimeError("end of discharge not found")
    tracks = [track_from(mask, P, s0) for s0 in starts]
    return max(tracks, key=lambda t: t[-1, 0] - t[0, 0])


def main():
    a = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(float)
    lab, dmin = classify(a)
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    summary = {}
    for ax, (mat, P) in zip(axs, PANELS.items()):
        for k, (rate, rgb) in enumerate(RATES.items()):
            mask = (lab == k) & (dmin < 40)
            pts = track(mask, P)
            q, v = lin(pts[:, 0], P["x"]), lin(pts[:, 1], P["y"])
            keep = q >= Q_MIN
            q, v = q[keep], v[keep]
            # a discharge branch never rises with capacity: drop points > 4 mV above the running minimum
            ok, vmin = np.ones(v.size, bool), v[0]
            for i in range(1, v.size):
                if v[i] > vmin + 0.004:
                    ok[i] = False
                else:
                    vmin = min(vmin, v[i])
            q, v = q[ok], v[ok]
            tag = f"{mat}_{rate:g}C"
            np.savetxt(OUT / f"{tag}.csv", np.c_[q, v], delimiter=",", header="capacity_mAh_per_g_S,voltage_V",
                       comments="", fmt="%.4f")
            gaps = np.diff(q)
            summary[tag] = {"crate": rate, "current_A_per_gS": 1.675 * rate, "n": int(q.size), "Q_first": float(q[0]),
                            "Q_end": float(q.max()), "largest_gap_mAh_g": float(gaps.max()) if gaps.size else None,
                            "V_at_300": float(np.interp(300.0, q, v)) if q.max() > 300 else None}
            ax.plot(q, v, ".", ms=1.5, color=np.array(rgb) / 255.0 * 0.8, label=f"{rate:g} C")
        ax.set(xlabel="capacity [mAh/g S]", ylabel="V", title=f"Ma 2026 Fig. 6, {mat} (digitized)", xlim=(0, 1800),
               ylim=(0.95, 2.4))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "ma2026_fig6_check.png", dpi=130)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
