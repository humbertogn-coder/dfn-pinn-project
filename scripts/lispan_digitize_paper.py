"""Digitize the simulated discharge curves of Simanjuntak et al. 2024 (Figs 4b, 5a, 6a, 6b) from the
open-access PDF, for the validation of the Li-SPAN reference solver (results/lispan/paper_digitized/).

    python scripts/lispan_digitize_paper.py --pdf references/Simanjuntak2024_LiSPAN_main.pdf

Requires pdftoppm (poppler) and Pillow.  Axis frames are located from the long dark lines, curves by
their MATLAB default colours; accuracy about 0.01 V / 10 mAh/g.  Where several curves overlap only
the top-most drawn one is recovered.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"blue": (0, 114, 189), "red": (217, 83, 25), "yellow": (237, 177, 32),
          "purple": (126, 47, 142), "green": (119, 172, 48), "black": (0, 0, 0)}


def render(pdf, page, dpi=220):
    from PIL import Image
    tmp = tempfile.mkdtemp()
    subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-png", str(pdf),
                    f"{tmp}/p"], check=True)
    f = sorted(Path(tmp).glob("p*.png"))[0]
    return np.asarray(Image.open(f).convert("RGB")).astype(int)


def _longest_run(mask_1d):
    best = cur = 0
    for v in mask_1d:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def frame(sub, min_frac=0.5):
    """Axes box = rows/columns containing one contiguous dark run longer than min_frac of the extent
    (text lines never do)."""
    gray = sub.mean(axis=2); dk = gray < 200
    h, w = dk.shape
    rr = np.array([_longest_run(dk[r]) for r in range(h)])
    cc = np.array([_longest_run(dk[:, c]) for c in range(w)])
    rows = np.where(rr > 0.9 * rr.max())[0]
    cols = np.where(cc > 0.9 * cc.max())[0]
    return rows.min(), rows.max(), cols.min(), cols.max()


def curves(sub, top, bot, left, right, color, xmax=1400.0, ymin=1.0, ymax=3.0, tol=70, exclude=None):
    c0 = np.array(COLORS[color])
    d = np.abs(sub - c0).sum(axis=2) < tol
    if exclude is not None:
        d[exclude] = False
    pts = []
    for c in range(left + 3, right - 2):
        ys = np.where(d[top + 3:bot - 2, c])[0] + top + 3
        if len(ys) == 0:
            continue
        groups = np.split(ys, np.where(np.diff(ys) > 6)[0] + 1)
        q = (c - left) / (right - left) * xmax
        for gr in groups:
            pts.append((q, ymax - (gr.mean() - top) / (bot - top) * (ymax - ymin), len(gr)))
    return np.array(pts)


def follow(pts, v_start=None, max_jump=0.06):
    """Single-curve track following (nearest group to the previous accepted value)."""
    pts = pts[np.argsort(pts[:, 0], kind="stable")]
    out = []
    for q in np.unique(pts[:, 0]):
        grp = pts[pts[:, 0] == q]
        if not out:
            v = grp[:, 1].max() if v_start is None else grp[np.argmin(abs(grp[:, 1] - v_start)), 1]
            out.append((q, v)); continue
        j = np.argmin(abs(grp[:, 1] - out[-1][1]))
        if abs(grp[j, 1] - out[-1][1]) < max_jump + 0.004 * (q - out[-1][0]):
            out.append((q, grp[j, 1]))
    return np.array(out)


def split_solid_dotted(pts, gap=4):
    """Two tracks per colour (solid 0.1 C above dotted 1 C): assign by continuity from the left."""
    pts = pts[np.argsort(pts[:, 0])]
    upper, lower = [], []
    for q, v, n in pts:
        if not upper:
            upper.append((q, v)); continue
        cand_u = abs(v - upper[-1][1]) < 0.06 + 0.002 * (q - upper[-1][0])
        cand_l = lower and abs(v - lower[-1][1]) < 0.06 + 0.002 * (q - lower[-1][0])
        if cand_u and (not cand_l or abs(v - upper[-1][1]) <= abs(v - lower[-1][1])):
            upper.append((q, v))
        elif cand_l or (lower == [] and v < upper[-1][1] - 0.05):
            lower.append((q, v))
    return np.array(upper), np.array(lower) if lower else np.empty((0, 2))


def main():
    ap = argparse.ArgumentParser()
    default_pdf = next((p for p in (ROOT / "references" / "Simanjuntak2024_LiSPAN_main.pdf",
                                    ROOT.parent / "references" / "Simanjuntak2024_LiSPAN_main.pdf") if p.exists()),
                       ROOT / "references" / "Simanjuntak2024_LiSPAN_main.pdf")
    ap.add_argument("--pdf", default=str(default_pdf), help="open-access PDF (owner folder: PINN-DFN-Project/references/)")
    ap.add_argument("--out", default=str(ROOT / "results" / "lispan" / "paper_digitized"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # --- Fig. 5a (page 7): 0.1 C cell voltage (black) and SPAN species (colours, right axis 0-1200)
    im = render(args.pdf, 7); h, w, _ = im.shape
    sub = im[int(0.06 * h):int(0.33 * h), int(0.05 * w):int(0.50 * w)]
    top, bot, left, right = frame(sub)
    excl = np.zeros(sub.shape[:2], bool); excl[top + 3:top + 110, left + 4:left + 160] = True   # legend
    v = curves(sub, top, bot, left, right, "black", exclude=excl)
    v = follow(v[(v[:, 2] <= 14)], v_start=2.3)
    np.savetxt(out / "fig5a_voltage_0.1C.csv", v, delimiter=",", header="Q_mAh_gS,V", comments="")
    for color, name, c0 in (("blue", "c_S4", 598.0), ("red", "c_S3", 0.0), ("yellow", "c_S2", 0.0), ("purple", "c_S1", 0.0)):
        # right axis: 0 at the bottom, 1400 at the top of the box (the "1200" label sits at 6/7 of the height)
        s = curves(sub, top, bot, left, right, color, ymin=0.0, ymax=1400.0, exclude=excl)
        s = follow(s, v_start=c0, max_jump=40.0)
        np.savetxt(out / f"fig5a_{name}_0.1C.csv", s, delimiter=",", header="Q_mAh_gS,c_mol_m3", comments="")
    # --- Fig. 6 (page 8): a) k0 variation (Zcc = 0), b) Zcc variation; solid 0.1 C, dotted 1 C
    im = render(args.pdf, 8); h, w, _ = im.shape
    sub = im[int(0.07 * h):int(0.30 * h), int(0.05 * w):int(0.97 * w)]
    gray = sub.mean(axis=2); dk = gray < 200
    rows = np.where(dk.sum(axis=1) > 900)[0]; cols = np.where(dk.sum(axis=0) > 300)[0]
    top, bot = rows.min(), rows.max()
    cols = np.split(cols, np.where(np.diff(cols) > 5)[0] + 1)
    frames = {"a": (cols[0].min(), cols[1].max()), "b": (cols[2].min(), cols[3].max())}
    labels = {"a": {"blue": "k0_1e-2", "red": "k0_1e-3", "yellow": "k0_1e-4"},
              "b": {"blue": "Zcc_0", "red": "Zcc_0.015", "yellow": "Zcc_0.025", "purple": "Zcc_0.035", "green": "Zcc_0.045"}}
    for panel, (l, r) in frames.items():
        excl = np.zeros(sub.shape[:2], bool); excl[top + 3:top + 140, r - 230:r - 2] = True   # legend (top right)
        for color, lab in labels[panel].items():
            pts = curves(sub, top, bot, l, r, color, exclude=excl)
            up, lo = split_solid_dotted(pts)
            np.savetxt(out / f"fig6{panel}_{lab}_0.1C.csv", up, delimiter=",", header="Q_mAh_gS,V", comments="")
            if len(lo):
                np.savetxt(out / f"fig6{panel}_{lab}_1C.csv", lo, delimiter=",", header="Q_mAh_gS,V", comments="")
    # --- Fig. 4b (page 5): rates 1/20, 1/10, 1/5, 1 C with Zcc = 0.035 (best fit stated in the text)
    im = render(args.pdf, 5); h, w, _ = im.shape
    sub = im[int(0.06 * h):int(0.27 * h), int(0.50 * w):int(0.97 * w)]
    top, bot, left, right = frame(sub)
    excl = np.zeros(sub.shape[:2], bool); excl[top + 3:top + 140, right - 230:right - 2] = True
    for color, lab in (("blue", "0.05C"), ("red", "0.1C"), ("yellow", "0.2C"), ("purple", "1C")):
        pts = curves(sub, top, bot, left, right, color, exclude=excl)
        pts = follow(pts, v_start=None)
        np.savetxt(out / f"fig4b_{lab}.csv", pts, delimiter=",", header="Q_mAh_gS,V", comments="")
    print("written", out)


if __name__ == "__main__":
    sys.exit(main())
