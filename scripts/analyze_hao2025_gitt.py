"""Semi-quantitative reading of the GITT panels of Hao et al., ACS Nano 19, 25385 (2025), SI Fig. S11
(SPAN and FeS2/SPAN, 30 min pulses at 0.05 C, 2 h rests, after two cycles at 0.1 C).

Input: the rasters extracted with `pdfimages -f 6 -l 8 -png <si.pdf> img` (img-006.png = S11a discharge,
img-007.png = S11b charge; 410 x 282 px, ~3 mAh/g and ~10 mV per pixel), so all numbers are approximate.

Reported (data/literature_span/hao2025/gitt_summary.json):
  * capacity per pulse (median spacing of the rest spikes) -> the current of "1 C" used by the authors
    (pulse = 0.05 C x 0.5 h, so 1 C [A/g] = pulse capacity [mAh/g] / 25);
  * voltage recovery during the rests (rested minus loaded voltage) between 150 and 550 mAh/g;
  * rested discharge vs rested charge voltage at equal state of discharge (gap that 2 h of rest does not remove);
  * apparent Li+ diffusion coefficients (right axis) range.
Usage:  python scripts/analyze_hao2025_gitt.py <img-006.png> <img-007.png>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "literature_span" / "hao2025" / "gitt_summary.json"
# axis calibration from the tick marks (px): x = 0 and 200 mAh/g per major tick; y = 1.0 V and 3.0 V; log D axis
CAL = {"discharge": {"x0": 61.5, "px_per_mAh": 64.0 / 200, "y_1V": 233.0, "px_per_V": (233.0 - 34.5) / 2.0,
                     "logD_y_m15": 64.0, "px_per_decade": (169.5 - 64.0) / 5},
       "charge": {"x0": 61.0, "px_per_mAh": 68.0 / 200, "y_1V": 234.0, "px_per_V": (234.0 - 35.0) / 2.0}}
COLOURS = {"SPAN": (66, 66, 66), "FeS2/SPAN": (222, 74, 90)}


def rests(a, rgb, c):
    m = np.abs(a - np.array(rgb)).sum(-1) < 45
    cols = []
    for x in range(64, 350):
        ys = np.flatnonzero(m[95:232, x]) + 95            # below ~2.4 V: excludes the log D markers at the top
        if ys.size < 2:
            continue
        seg = max(np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1), key=len)
        cols.append(((x - c["x0"]) / c["px_per_mAh"], 1 + (c["y_1V"] - seg.min()) / c["px_per_V"],
                     1 + (c["y_1V"] - seg.max()) / c["px_per_V"], len(seg), x))
    cols = np.array(cols)
    ext = cols[:, 3]
    pk = [i for i in range(1, len(ext) - 1) if ext[i] >= 4 and ext[i] >= ext[i - 1] and ext[i] >= ext[i + 1]]
    keep = []
    for i in pk:                                          # one column per rest spike
        if not keep or cols[i, 4] - cols[keep[-1], 4] > 2:
            keep.append(i)
        elif ext[i] > ext[keep[-1]]:
            keep[-1] = i
    return cols[keep]


def main():
    imgs = {"discharge": np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int),
            "charge": np.asarray(Image.open(sys.argv[2]).convert("RGB")).astype(int)}
    out = {"source": "Hao et al. ACS Nano 19, 25385 (2025), SI Fig. S11; approximate (raster ~3 mAh/g, ~10 mV per px)"}
    for name, rgb in COLOURS.items():
        r = {}
        res = {k: rests(imgs[k], rgb, CAL[k]) for k in imgs}
        d = res["discharge"]
        # mean spike spacing over runs of consecutive visible spikes (4-8 px apart; a hidden spike breaks a run):
        # total length of the runs / number of intervals (sub-pixel, unlike the median of integer spacings)
        x = d[:, 4]
        ok = (np.diff(x) > 4) & (np.diff(x) < 8)
        pulse = float(np.sum(np.diff(x)[ok]) / np.sum(ok)) / CAL["discharge"]["px_per_mAh"]
        r["pulse_capacity_mAh_g"] = round(pulse, 1)
        r["implied_1C_A_per_g"] = round(pulse / 25.0, 3)
        for k in ("discharge", "charge"):
            s = res[k]
            mid = s[(s[:, 0] > 150) & (s[:, 0] < 550)]
            rec = 1e3 * (mid[:, 1] - mid[:, 2])
            r[f"{k}_rest_recovery_mV_150_550"] = {"mean": round(float(rec.mean())), "min": round(float(rec.min())),
                                                  "max": round(float(rec.max())), "n_rests": int(len(mid))}
        dq_end, cq_end = d[:, 0].max(), res["charge"][:, 0].max()
        gaps = {}
        for qd in (300, 400, 500):
            vd = np.interp(qd, d[:, 0], d[:, 1])                              # rested discharge (top of spikes)
            qc = cq_end - qd * cq_end / dq_end                                # same state of discharge on charge
            vc = np.interp(qc, res["charge"][:, 0], res["charge"][:, 2])     # rested charge (bottom of spikes)
            gaps[str(qd)] = round(1e3 * float(vc - vd))
        r["rested_charge_minus_rested_discharge_mV"] = gaps
        a = imgs["discharge"]
        m = np.abs(a - np.array(rgb)).sum(-1) < 45
        ys = np.nonzero(m[0:60, 64:350])[0]
        c = CAL["discharge"]
        logd = -15 + (c["logD_y_m15"] - np.array([ys.max(), np.median(ys), ys.min()])) / c["px_per_decade"]
        r["apparent_log10_D_cm2_s_discharge"] = {"min": round(float(logd[0]), 1), "median": round(float(logd[1]), 1),
                                                 "max": round(float(logd[2]), 1)}
        out[name] = r
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
