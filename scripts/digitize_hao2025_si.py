"""Digitize the plain-SPAN discharge curves in the Supporting Information of Hao et al., ACS Nano 19, 25385 (2025)
(doi 10.1021/acsnano.5c06966): Li || SPAN coin cells, SPAN made at 300 C (53.11 wt% S, SI Table S1), electrode
active:carbon:PAA 90:5:5, 1.5-2 mg/cm2 active material, 1.0 M LiPF6 in EC/DEC/FEC 45:45:10, 1.0-3.0 V.

  S9   rate test, 0.1 / 0.2 / 0.5 / 1 / 2 C (one discharge-charge profile per rate)
  S12b cycles 1 / 2 / 10 / 100 at 0.5 C (cycle 1 is probably the 0.1 C activation cycle; cycle 2 ends at the
       paper's "initial" 0.5 C capacity)

Capacity in mAh per g of active material (SPAN). The paper does not define 1 C (see data/literature_span/README.md).

Input: the two rasters extracted from the SI PDF with `pdfimages -f 6 -l 8 -png <si.pdf> img`
(img-001.png = Fig. S9, 629 x 503 px; img-009.png = Fig. S12b, 419 x 335 px; both 200 ppi). Each pixel is assigned
to the nearest legend colour (or white / grey / black); each discharge branch is tracked from its end point (1.0 V)
towards zero capacity by continuity, with the constraint that a discharge branch followed towards Q = 0 never falls in
voltage (this separates it from the charge branch of the same colour). Points below Q_MIN, where all branches cross,
are dropped.

Hidden ranges: two curves are drawn on top of an earlier curve that coincides with them over part of the discharge
(0.1 C in S9, cycle 2 in S12b), so the earlier colour is not visible there at all,
i.e. the two agree within the line width (~3 px: S9 ~18 mV, S12b ~27 mV). The visible part is tracked from the end of
discharge; below the capacity where it disappears, the covering curve is used ("spliced"); the covering curve is the
later-drawn candidate that meets the visible part with the smallest voltage step. The splice capacity and the voltage
steps at the junction are reported in summary.json.

Axis calibration (pixels, measured from the tick marks):
  S9:   x 109.67 px = 0, 155.17 px per 300 mAh/g;  y 376.5 px = 1.0 V, 43.0 px = 3.0 V   (1.93 mAh/g x 6.0 mV per px)
  S12b: x 72.5 px = 0, 103.0 px per 300 mAh/g;     y 250.5 px = 1.0 V, 28.5 px = 3.0 V   (2.91 mAh/g x 9.0 mV per px)

Output: data/literature_span/hao2025/SPAN_{rate}C.csv (S9), SPAN_0.5C_{n}cyc.csv (S12b, cycle n), hao2025_check.png,
summary.json.
Usage:  python scripts/digitize_hao2025_si.py <img-001.png> <img-009.png>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "literature_span" / "hao2025"
PANELS = {
    "S9": {"x": (109.67, 0.0, 109.67 + 3 * 155.17, 900.0), "y": (376.5, 1.0, 43.0, 3.0), "xlim": (112, 625),
           "ylim": (4, 416), "legend": (470, 625, 145, 320),
           "curves": {"0.1C": (66, 66, 66), "0.2C": (99, 107, 181), "0.5C": (255, 107, 107), "1C": (165, 123, 16),
                      "2C": (107, 165, 16)},
           "text": {"0.1C": 743.89, "0.2C": 692.95, "0.5C": 652.11, "1C": 622.04, "2C": 591.53},
           "splice": {"0.1C": ["0.2C", "0.5C"]}},
    "S12b": {"x": (72.5, 0.0, 72.5 + 3 * 103.0, 900.0), "y": (250.5, 1.0, 28.5, 3.0), "xlim": (75, 416),
             "ylim": (3, 277), "legend": (310, 416, 95, 190),
             "curves": {"cycle1": (66, 66, 66), "cycle2": (99, 107, 181), "cycle10": (247, 99, 49),
                        "cycle100": (165, 123, 16)},
             "text": {"cycle2": 720.83},
             "splice": {"cycle2": ["cycle10", "cycle100"]}},
}
OTHER = [(255, 255, 255), (0, 0, 0), (128, 128, 128), (200, 200, 200)]
Q_MIN = 30.0
MAX_MISS = 25


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


def classify(a, colours):
    pal = np.array(list(colours) + OTHER, float)
    d = ((a[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
    return np.argmin(d, axis=2), np.sqrt(np.min(d, axis=2))


def in_legend(P, x, y):
    lg = P["legend"]
    return lg[0] <= x <= lg[1] and lg[2] <= y <= lg[3]


def track_from(mask, P, start):
    x0 = P["xlim"][0]
    y_lo, y_hi = P["ylim"]
    pts, miss = [start], 0
    for x in range(start[0] - 1, x0, -1):
        if len(pts) >= 6:
            (xa, ya), (xb, yb) = pts[-6], pts[-1]
            pred = yb + (yb - ya) / (xb - xa) * (x - xb)
        else:
            pred = pts[-1][1]
        cands = [c + y_lo for c, n in runs(mask[y_lo:y_hi, x]) if 1 <= n <= 25]
        cands = [c for c in cands if c <= pts[-1][1] + 1.0 and not in_legend(P, x, c)
                 and abs(c - pred) <= 4 + 2 * miss]
        if not cands:
            miss += 1
            if miss > MAX_MISS:
                break
            continue
        miss = 0
        pts.append((x, min(cands, key=lambda c: abs(c - pred))))
    return np.array(pts[::-1])


def track(mask, P):
    x0, x1 = P["xlim"]
    y_lo, y_hi = P["ylim"]
    y_end = P["y"][0]
    starts = []
    for x in range(x1, x0, -1):
        rr = [c + y_lo for c, n in runs(mask[y_lo:y_hi, x]) if n >= 1]
        near = [c for c in rr if abs(c - y_end) < 8 and not in_legend(P, x, c)]
        if near and (not starts or starts[-1][0] - x > 4):
            starts.append((x, min(near, key=lambda c: abs(c - y_end))))
        if len(starts) >= 6:
            break
    if not starts:
        raise RuntimeError("end of discharge not found")
    tracks = [track_from(mask, P, s0) for s0 in starts]
    return max(tracks, key=lambda t: t[-1, 0] - t[0, 0])


def to_curve(pts, P):
    q, v = lin(pts[:, 0], P["x"]), lin(pts[:, 1], P["y"])
    keep = q >= Q_MIN
    q, v = q[keep], v[keep]
    ok, vmin = np.ones(v.size, bool), v[0]          # running-minimum filter (a discharge branch never rises)
    for i in range(1, v.size):
        if v[i] > vmin + 0.004:
            ok[i] = False
        else:
            vmin = min(vmin, v[i])
    return q[ok], v[ok]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    summary = {}
    for ax, (panel, png) in zip(axs, zip(PANELS, sys.argv[1:3])):
        P = PANELS[panel]
        a = np.asarray(Image.open(png).convert("RGB")).astype(float)
        lab, dmin = classify(a, P["curves"].values())
        curves, raw_gap = {}, {}
        for k, name in enumerate(P["curves"]):
            pts = track((lab == k) & (dmin < 45), P)
            if pts[:, 0].size > 1:            # largest jump in x along the visible track (px)
                raw_gap[name] = float(np.max(np.diff(pts[:, 0])))
            curves[name] = to_curve(pts, P)
        spliced = {}
        for name, covers in P.get("splice", {}).items():
            q, v = curves[name]
            # visible part: the contiguous run from the end of discharge back to the first large gap
            gaps = np.flatnonzero(np.diff(q) > 15.0)
            i0 = gaps[-1] + 1 if gaps.size else 0
            qv, vv = q[i0:], v[i0:]
            # the covering curve is the later-drawn candidate that meets the visible part (smallest voltage step)
            steps = {c: float(vv[0] - np.interp(qv[0], *curves[c])) for c in covers}
            cover = min(steps, key=lambda c: abs(steps[c]))
            qc, vc = curves[cover]
            below = qc < qv[0] - 2.0
            curves[name] = (np.r_[qc[below], qv], np.r_[vc[below], vv])
            spliced[name] = {"covered_by": cover, "splice_at_mAh_g": round(float(qv[0]), 1),
                             "voltage_step_at_splice_mV": {c: round(1e3 * st, 1) for c, st in steps.items()},
                             "discarded_visible_points_below": int(i0)}
        for name, rgb in P["curves"].items():
            q, v = curves[name]
            tag = f"SPAN_{name}" if panel == "S9" else f"SPAN_0.5C_{name[5:]}cyc"
            np.savetxt(OUT / f"{tag}.csv", np.c_[q, v], delimiter=",", header="capacity_mAh_per_g_SPAN,voltage_V",
                       comments="", fmt="%.4f")
            gaps = np.diff(q)
            summary[tag] = {"figure": panel, "n": int(q.size), "Q_first": round(float(q[0]), 1),
                            "Q_end": round(float(q.max()), 1), "Q_end_text": P["text"].get(name),
                            "largest_gap_mAh_g": round(float(gaps.max()), 1) if gaps.size else None,
                            "V_at_300": round(float(np.interp(300.0, q, v)), 4) if q[0] < 300 < q.max() else None,
                            "splice": spliced.get(name)}
            ax.plot(q, v, ".", ms=1.5, color=np.array(rgb) / 255.0 * 0.85, label=name)
        ax.set(xlabel="capacity [mAh/g SPAN]", ylabel="V", title=f"Hao 2025 SI Fig. {panel} (digitized)",
               xlim=(0, 950), ylim=(0.95, 2.4))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "hao2025_check.png", dpi=130)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
