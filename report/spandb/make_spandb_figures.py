"""Figures and numbers for the SPAN-DB results (two-curve fits of every usable cell, scripts/lispan_spandb_fit.py).

    python report/spandb/make_spandb_figures.py --spandb <span_db folder>

Inputs : results/lispan/spandb/*.json (network chem_v2_norar_s0, 0.1-1 C) and, if present, results/lispan/spandb_v3s1/
         (chem_v3_s1, 0.08-3 C); SPAN-DB tables for the counts.
Outputs: report/spandb/figures/sdb_*.png and report/spandb/numbers.json.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
INK, INK2, GRID, BG = "#1B2A3A", "#52514E", "#E4E2DC", "#FFFFFF"
BLUE, ORANGE, AQUA, GREY, LIGHT = "#2A78D6", "#EB6834", "#1BAF7A", "#9A988F", "#C9C7BF"
plt.rcParams.update({
    "font.size": 15, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.4, "legend.frameon": False, "savefig.dpi": 160,
    "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 16, "axes.titlelocation": "left",
    "axes.axisbelow": True, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK})

GOOD_MV = 25.0          # misfit up to which the continuum model is taken to describe a cell (Phase 2: 11-17 mV typical)
CLASSES = ["carbonate", "ether (dilute)", "LHCE", "HCE", "fluorinated", "other liquid", "polymer / solid"]
SHORT = {"2019_Chen_088186": "Chen 2019", "2023_Hu_3c00450": "Hu 2023", "2023_Huang_202306484": "Huang 2023",
         "2023_Li_d3ta01803b": "Li 2023", "2023_Shi_202301656": "Shi 2023a", "2023_Shi_3c00380": "Shi 2023b",
         "2023_Wang_144142": "Wang 2023", "2023_Wu_cnl261": "Wu 2023", "2024_Chen_202422791": "Chen 2024",
         "2024_Jia_202401055": "Jia 2024", "2024_Jin_4c09751": "Jin 2024", "2024_Qin_4c00859": "Qin 2024",
         "2024_Shi_4c07155": "Shi 2024", "2024_Simanjuntak_144571": "Simanjuntak 2024", "2024_Wang_150300": "Wang 2024",
         "2024_Wu_4c13280": "Wu 2024", "2024_You_4c02576": "You 2024", "2024_Zuo_151977": "Zuo 2024",
         "2026_Ma_74169": "Ma 2026", "2023_Park_3c08876": "Park 2023", "2023_Li_3c08576": "Li 2023b",
         "2023_Wu_3c00815": "Wu 2023b", "2024_Shao_nano14131155": "Shao 2024", "2024_Kakiage_003211": "Kakiage 2024"}


def short(cell):
    p, c = cell.rsplit("_C", 1)
    return f"{SHORT.get(p, p)} C{c}"


def el_class(r):
    fam, txt = (r.get("electrolyte_family") or "").lower(), (r.get("electrolyte") or "").lower() + " " + (r.get("electrolyte_family") or "").lower()
    if any(k in txt for k in ("solid", "llzo", "gel", "polymer", "polymeriz", "hydride", "pise")):
        return "polymer / solid"
    if fam.startswith("carbonate"):
        return "carbonate"
    if fam.startswith("dilute_ether"):
        return "ether (dilute)"
    for k in ("LHCE", "HCE", "fluorinated"):
        if fam.startswith(k.lower()):
            return k
    return "other liquid"


def load(folder):
    folder = ROOT / "results" / "lispan" / folder
    if not folder.exists():
        return []
    return [json.loads(f.read_text()) for f in sorted(folder.glob("*.json")) if not f.name.startswith("summary")]


def fv_err(r):
    fv = (r.get("fv_check") or {}).get("curves") or []
    return max(c["rms_net_vs_fv_mV"] for c in fv) if fv else np.nan


def db_counts(spandb):
    root = Path(spandb) / "db"
    rd = lambda f: list(csv.DictReader(open(root / f, encoding="utf-8")))       # noqa: E731
    papers, cells, curves = rd("papers.csv"), rd("cells.csv"), rd("curves.csv")
    return {"papers": len(papers), "relevance": dict(Counter(p["relevance"] for p in papers)),
            "papers_extracted": len({c["cell_id"].rsplit("_C", 1)[0] for c in cells}), "cells": len(cells),
            "curves": len(curves), "curves_qc": dict(Counter(k["qc_status"] for k in curves)),
            "info_levels": dict(Counter(c["info_level"] for c in cells))}


# ---------------------------------------------------------------------------------------------------------- figures
def fig_misfit(fit):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.axhspan(0, GOOD_MV, color="#EAF2FB", zorder=0)
    ax.text(-0.45, 0.55, f"shaded: model describes the cell (\u2264 {GOOD_MV:.0f} mV)", ha="left", va="bottom",
            fontsize=13, color=BLUE)
    rng = np.random.default_rng(3)
    for k, cl in enumerate(CLASSES):
        rs = [r for r in fit if el_class(r) == cl]
        xs = k + rng.uniform(-0.18, 0.18, len(rs))
        for x, r in zip(xs, rs):
            m, e = r["fit"]["rms_mV"], fv_err(r)
            col = BLUE if m <= GOOD_MV else ORANGE
            ax.plot([x, x], [e, m], color=LIGHT, lw=1.2, zorder=1)
            ax.plot(x, m, "o", ms=11, color=col, mec="white", mew=1.2, zorder=3)
            ax.plot(x, e, "_", ms=12, mew=2.2, color=GREY, zorder=2)
    ax.plot([], [], "o", ms=11, color=BLUE, label="fit misfit (data vs network)")
    ax.plot([], [], "o", ms=11, color=ORANGE, label="fit misfit above 25 mV")
    ax.plot([], [], "_", ms=12, mew=2.2, color=GREY, label="network vs FV solver at the fitted parameters")
    ax.set_xticks(range(len(CLASSES)), [f"{c}\n(n = {sum(el_class(r) == c for r in fit)})" for c in CLASSES], fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 120)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100], ["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_ylabel("rms voltage error [mV]")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=12.5, ncol=3, handletextpad=0.3, columnspacing=1.5)
    fig.savefig(OUT / "sdb_misfit.png")
    plt.close(fig)


def _panel(ax, r, title, pred=True):
    cols = [BLUE, ORANGE]
    for k, c in enumerate(r["curves"]):
        ax.plot(c["q"], c["V"], "o", ms=4, color=cols[k], alpha=0.55, mec="none")
        ax.plot(c["q"], c["V_fit"], "-", color=cols[k], lw=2.4, label=f"{c['label']} (calibration)")
    for p in (r["predictions"] if pred else []):
        if "V" in p:
            ax.plot(p["q"], p["V_data"], "o", ms=3.5, color=AQUA, alpha=0.55, mec="none")
            ax.plot(p["q"], p["V"], "--", color=AQUA, lw=2.2, label=f"{p['label']} (predicted, {p['rms_mV']:.0f} mV)")
            break
    ax.set_title(title, fontsize=14)
    ax.text(0.97, 0.95, f"misfit {r['fit']['rms_mV']:.0f} mV", transform=ax.transAxes, ha="right", va="top", fontsize=13,
            color=BLUE if r["fit"]["rms_mV"] <= GOOD_MV else ORANGE, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.set_xlabel("capacity [mAh/g]", fontsize=13)
    ax.set_ylabel("voltage [V]", fontsize=13)
    ax.tick_params(labelsize=12)


def fig_examples(byc):
    pick = [("2024_Qin_4c00859_C1", "Qin 2024: 1.5 M LiFSI in DEE", True),
            ("2024_Chen_202422791_C2", "Chen 2024: 3 m LiFSI in 2,5-DMTHF (HCE)", True),
            ("2024_Wu_4c13280_C3", "Wu 2024: 1 M LiFSI in DME (dilute ether)", False),
            ("2023_Shi_3c00380_C4", "Shi 2023b: LLZO solid electrolyte", False)]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.6))
    for ax, (c, t, pr) in zip(axs.flat, pick):
        _panel(ax, byc[c], t, pr)
    fig.tight_layout(h_pad=2.0, w_pad=2.5)
    fig.savefig(OUT / "sdb_examples.png")
    plt.close(fig)


def fig_funnel(num):
    d = num["db"]
    rows = [("PDF articles in the SPAN folder", d["papers"], GREY),
            ("relevance A: profiles at \u2265 2 rates", d["relevance"].get("A", 0), GREY),
            ("articles extracted so far", d["papers_extracted"], BLUE),
            ("cells with \u2265 2 discharge currents", num["cells_two_or_more_currents"], BLUE),
            ("fitted (pair inside 0.1-1 C)", num["fitted"], BLUE),
            ("model describes the cell (\u2264 25 mV)", num["good"], AQUA)]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for k, (lab, v, col) in enumerate(rows):
        ax.barh(k, v, color=col, height=0.66)
        ax.text(v + 6, k, f"{v}", va="center", fontsize=15, fontweight="bold", color=INK)
        ax.text(-8, k, lab, va="center", ha="right", fontsize=14, color=INK)
    ax.text(d["papers_extracted"] + 60, 2, f"\u2192 {d['cells']} cells, {d['curves']} curves", va="center", fontsize=13.5, color=INK2)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, d["papers"] * 1.08)
    ax.axis("off")
    fig.savefig(OUT / "sdb_funnel.png")
    plt.close(fig)


CONTROLLED = [("Qin 2024", [("2024_Qin_4c00859_C1", "DEE"), ("2024_Qin_4c00859_C2", "DEE/ETFE"), ("2024_Qin_4c00859_C3", "DEE/ETFE+IL")]),
              ("Chen 2024", [("2024_Chen_202422791_C1", "LHCE"), ("2024_Chen_202422791_C2", "HCE")]),
              ("Wu 2024", [("2024_Wu_4c13280_C1", "DBE/TTE"), ("2024_Wu_4c13280_C2", "DME/TTE")]),
              ("Chen 2019", [("2019_Chen_088186_C1", "ether"), ("2019_Chen_088186_C2", "carbonate")])]


def fig_controlled(byc):
    pars = [("Z_CC", 1e4, "series resistance Z [Ω cm²]"), ("d", 1e3, "common potential shift d [mV]"),
            ("dU0_2", 1e3, "step-2 potential offset [mV]")]
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4))
    xt, xl, x = [], [], 0
    pos = {}
    for gi, (paper, cells) in enumerate(CONTROLLED):
        for c, lab in cells:
            pos[c] = (x, gi)
            xt.append(x); xl.append(f"{lab}")
            x += 1
        x += 0.8
    gcol = [BLUE, ORANGE, AQUA, "#6250D6"]
    for ax, (p, sc, lab) in zip(axs, pars):
        for gi, (paper, cells) in enumerate(CONTROLLED):
            xs = [pos[c][0] for c, _ in cells]
            ys = [byc[c]["fit"]["params"][p] * sc for c, _ in cells]
            es = [byc[c]["fit"]["sd_total"].get(p, 0.0) for c, _ in cells]
            edge = [any(f["flag"] == "box_edge" and p in f["msg"] for f in byc[c]["fit"]["flags"]) for c, _ in cells]
            ax.plot(xs, ys, "-", color=gcol[gi], lw=1.2, alpha=0.6)
            for xx, yy, ee, ed in zip(xs, ys, es, edge):
                ax.errorbar(xx, yy, yerr=ee, fmt="o", ms=10, color=gcol[gi], mfc="white" if ed else gcol[gi], capsize=4, lw=1.6)
            ax.text(np.mean(xs), 1.02, paper, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=12,
                    color=gcol[gi], fontweight="bold")
        ax.set_xticks(xt, xl, rotation=40, ha="right", fontsize=11.5)
        ax.set_title(lab, fontsize=14, pad=26)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "sdb_controlled.png")
    plt.close(fig)


def fig_predictions(fit):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    pts = []
    for r in fit:
        for p in r["predictions"]:
            if "rms_mV" in p:
                pts.append((r["fit"]["rms_mV"], p["rms_mV"], p["between"], r["cell"]))
    lim = 160
    ax.plot([1, lim], [1, lim], "-", color=GREY, lw=1.2)
    ax.plot([1, lim], [2, 2 * lim], ":", color=GREY, lw=1.2)
    for m, e, b, c in pts:
        col = BLUE if m <= GOOD_MV else ORANGE
        ax.plot(m, e, "o", ms=10, color=col, mfc=col if b else "white", mew=2)
    ax.plot([], [], "o", ms=10, color=INK2, label="between the two calibration currents")
    ax.plot([], [], "o", ms=10, color=INK2, mfc="white", mew=2, label="outside (extrapolated)")
    ax.text(110, 95, "same as fit", color=GREY, fontsize=12, ha="right")
    ax.text(55, 125, "2x fit", color=GREY, fontsize=12, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(8, lim); ax.set_ylim(8, 400)
    ax.set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"]); ax.set_yticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    ax.set_xlabel("misfit of the two calibration curves [mV]")
    ax.set_ylabel("error of the predicted third current [mV]")
    ax.legend(loc="upper left", fontsize=12)
    fig.savefig(OUT / "sdb_predictions.png")
    plt.close(fig)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spandb", required=True)
    a = ap.parse_args()
    res = load("spandb")
    fit = [r for r in res if r["status"] == "fitted"]
    byc = {r["cell"]: r for r in res}
    fig_misfit(fit)
    fig_examples(byc)
    fig_controlled(byc)
    pts = fig_predictions(fit)
    good = [r for r in fit if r["fit"]["rms_mV"] <= GOOD_MV]
    num = {"db": db_counts(a.spandb), "cells_two_or_more_currents": len(res), "fitted": len(fit),
           "skipped": dict(Counter(r["reason"] for r in res if r["status"] != "fitted")),
           "good": len(good), "misfit_median_good_mV": float(np.median([r["fit"]["rms_mV"] for r in good])),
           "misfit_all_mV": sorted(round(r["fit"]["rms_mV"], 1) for r in fit),
           "net_vs_fv_good_max_mV": float(max(fv_err(r) for r in good)),
           "net_vs_fv_good_median_mV": float(np.median([fv_err(r) for r in good])),
           "by_class": {c: {"n": sum(el_class(r) == c for r in fit),
                            "good": sum(el_class(r) == c and r["fit"]["rms_mV"] <= GOOD_MV for r in fit),
                            "cells": {short(r["cell"]): round(r["fit"]["rms_mV"], 1) for r in fit if el_class(r) == c}}
                        for c in CLASSES},
           "predictions": [{"cell": short(c), "fit_mV": round(m, 1), "pred_mV": round(e, 1), "between": b} for m, e, b, c in pts],
           "params_good": {short(r["cell"]): {k: round(v, 4) for k, v in r["fit"]["params"].items()} | {"rms_mV": round(r["fit"]["rms_mV"], 1),
                           "class": el_class(r)} for r in good}}
    v3 = load("spandb_v3s1")
    if v3:
        f3 = [r for r in v3 if r["status"] == "fitted"]
        num["v3s1"] = {"fitted": len(f3), "good": sum(r["fit"]["rms_mV"] <= GOOD_MV for r in f3),
                       "predictions": [{"cell": short(r["cell"]), "fit_mV": round(r["fit"]["rms_mV"], 1), "i": p["i_sp"],
                                        "pred_mV": round(p["rms_mV"], 1), "between": p["between"]}
                                       for r in f3 for p in r["predictions"] if "rms_mV" in p]}
    fig_funnel(num)
    (HERE / "numbers.json").write_text(json.dumps(num, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in num.items() if k not in ("params_good", "predictions", "misfit_all_mV")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
