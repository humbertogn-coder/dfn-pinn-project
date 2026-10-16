"""Figures for the Phase 2 pilot on published multi-rate SPAN data (results/lispan/literature_rates).

    python report/phase2/make_figures.py

l1_curves.png   four calibrations shown against the published curves: Se-SPAN in ether (Chen 2019) calibrated on
                slow + medium and on slowest + fastest rate; SPAN aerogel (Ma 2026) calibrated on 0.2 C only and on
                0.2 + 1 C
l2_errors.png   rms error at every rate for every calibration and dataset
l3_split.png    why two rates cannot be extrapolated: Chen 2019 voltage at 400 mAh/g vs current, with the
                log (kinetic) + linear (ohmic) decomposition and the two one-mechanism extrapolations
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results" / "lispan" / "literature_rates"
DATA = ROOT / "data" / "literature_span"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

C1, C2, C3, C4, C5, C6, C7, C8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.axisbelow": True})
RATE_COL = [C1, C3, C4, C7, C2, C8]
DATASETS = [("chen2019_carbonate", "Se-SPAN, carbonate\n(Chen 2019)", "A/g"),
            ("chen2019_ether", "Se-SPAN, ether\n(Chen 2019)", "A/g"),
            ("ma2026_SPAN-A", "SPAN aerogel, carbonate\n(Ma 2026)", "C"),
            ("ma2026_SPAN", "SPAN, carbonate\n(Ma 2026)", "C")]
CASES = {  # label -> fit string per dataset
    "slow rate only": {"chen2019_carbonate": "0.4", "chen2019_ether": "0.4", "ma2026_SPAN-A": "0.2", "ma2026_SPAN": "0.2"},
    "slow + medium": {"chen2019_carbonate": "0.4+2", "chen2019_ether": "0.4+2", "ma2026_SPAN-A": "0.2+1",
                      "ma2026_SPAN": "0.2+1"},
    "slowest + fastest": {"chen2019_carbonate": "0.4+10", "chen2019_ether": "0.4+10"},
    "slow + medium, salt diffusivity free": {"ma2026_SPAN-A": "0.2+1_Dfree", "ma2026_SPAN": "0.2+1_Dfree"},
    "all rates": {"chen2019_carbonate": "0.4+1+2+4+6+10", "chen2019_ether": "0.4+1+2+4+6+10"},
}
CASE_STYLE = {"slow rate only": (C2, "o"), "slow + medium": (C1, "s"), "slowest + fastest": (C3, "^"),
              "slow + medium, salt diffusivity free": (C7, "v"), "all rates": (INK2, "D")}


def load(ds, fit):
    j = RES / f"{ds}_fit{fit}.json"
    if fit is None or not j.exists():
        return None, None
    return json.loads(j.read_text()), np.load(RES / f"{ds}_fit{fit}.npz")


def curves_panel(ax, ds, fit, unit, title):
    rep, z = load(ds, fit)
    if rep is None:
        ax.set_title(title + " (pending)")
        return
    rates = sorted(float(k) for k in rep["per_rate"])
    for col, r in zip(RATE_COL, rates):
        k = f"{r:g}"
        ax.plot(z[f"qd_{k}"], z[f"vd_{k}"], color=col, lw=3.4, alpha=0.33)
        pr = rep["per_rate"][k]
        if f"Q_{k}" in z:
            ax.plot(z[f"Q_{k}"], z[f"V_{k}"], color=col, lw=1.4, ls="-" if pr["role"] == "fit" else "--",
                    label=f"{k} {unit}: {'calibration' if pr['role'] == 'fit' else 'predicted'}, {pr['rms_mV']:.0f} mV")
    ax.set(xlabel="capacity [mAh/g, paper's basis]", ylabel="voltage [V]", ylim=(0.95, 2.45), xlim=(0, None))
    ax.set_title(title)
    ax.legend(fontsize=7.3, loc="lower left")


def l1_curves():
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.6))
    curves_panel(axs[0, 0], "chen2019_ether", "0.4+2", "A/g", "(a) Se-SPAN, ether: calibrated on 0.4 + 2 A/g")
    curves_panel(axs[0, 1], "chen2019_ether", "0.4+10", "A/g", "(b) Same data: calibrated on 0.4 + 10 A/g")
    curves_panel(axs[1, 0], "ma2026_SPAN-A", "0.2", "C", "(c) SPAN aerogel: calibrated on 0.2 C only")
    curves_panel(axs[1, 1], "ma2026_SPAN-A", "0.2+1", "C", "(d) Same data: calibrated on 0.2 + 1 C")
    fig.text(0.01, -0.01, "Thick pale lines: published curves (Chen 2019 read exactly from the PDF vectors; Ma 2026 "
             "digitized). Thin lines: calibrated model, solid at the calibration rates, dashed where predicted.",
             fontsize=8.2, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "l1_curves.png")
    plt.close(fig)
    print("wrote l1_curves.png")


def l2_errors():
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.1))
    for ax, (ds, title, unit) in zip(axs, DATASETS):
        rates = None
        cases = [c for c in CASES if ds in CASES[c]]
        for j, case in enumerate(cases):
            rep, _ = load(ds, CASES[case][ds])
            if rep is None:
                continue
            col, mk = CASE_STYLE[case]
            rates = sorted(float(k) for k in rep["per_rate"])
            xs = np.arange(len(rates)) + (j - (len(cases) - 1) / 2) * 0.17
            e = [rep["per_rate"][f"{r:g}"]["rms_mV"] for r in rates]
            role = [rep["per_rate"][f"{r:g}"]["role"] for r in rates]
            ax.scatter(xs, e, s=30, marker=mk, facecolors=[col if ro == "prediction" else "white" for ro in role],
                       edgecolors=col, linewidths=1.4, zorder=3)
            ax.plot([], [], ls="none", marker=mk, ms=6, color=col, label=case)
        if rates is None:
            ax.set_title(title + " (pending)")
            continue
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:g}" for r in rates])
        ax.set_xlabel(f"rate [{unit}]")
        ax.set_yscale("log")
        ax.set_ylim(4, 700)
        ax.axhline(20, color=INK2, lw=0.9, ls=":")
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=7, loc="upper left")
    axs[0].set_ylabel("rms error [mV]  (log scale)")
    fig.text(0.01, -0.04, "Open markers: rates used for the calibration; filled: predicted rates. Dotted line: 20 mV. "
             "Ma 2026: the 0.1 C curve is the first cycle of the cell.", fontsize=8.2, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "l2_errors.png")
    plt.close(fig)
    print("wrote l2_errors.png")


def l3_split():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    rates = np.array([0.4, 1, 2, 4, 6, 10])
    for ax, el, title in ((axs[0], "carbonate", "(a) Se-SPAN, carbonate"), (axs[1], "ether", "(b) Se-SPAN, ether")):
        V = []
        for r in rates:
            d = np.loadtxt(DATA / "chen2019" / f"{el}_{r:g}Ag.csv", delimiter=",", skiprows=1)
            V.append(np.interp(400.0, d[:, 0], d[:, 1]))
        V = np.array(V)
        A = np.c_[np.ones_like(rates), -np.log(rates), -rates]
        coef = np.linalg.lstsq(A, V, rcond=None)[0]
        ii = np.linspace(0.3, 10.5, 200)
        dV = V[0] - V[2]                                   # 0.4 -> 2 A/g
        tafel = V[0] - dV / np.log(5.0) * np.log(ii / 0.4)
        ohm = V[0] - dV / 1.6 * (ii - 0.4)
        ax.plot(ii, coef[0] - coef[1] * np.log(ii) - coef[2] * ii, color=INK2, lw=1.3,
                label=f"log + linear fit to all six: {1e3 * coef[1]:.0f} mV per e-fold + {1e3 * coef[2]:.0f} mV per A/g")
        ax.plot(ii, tafel, color=C1, lw=1.4, ls="--", label="only kinetic (log), from 0.4 and 2 A/g")
        ax.plot(ii, ohm, color=C2, lw=1.4, ls="--", label="only ohmic (linear), from 0.4 and 2 A/g")
        ax.plot(rates, V, "o", ms=7, color=INK, zorder=4, label="measured")
        ax.plot(rates[[0, 2]], V[[0, 2]], "o", ms=12, mfc="none", mec=C3, mew=2, zorder=5, label="the two calibration rates")
        ax.set(xlabel="current [A/g (S+Se)]", ylabel="voltage at 400 mAh/g [V]", ylim=(0.9, 2.1), xlim=(0, 10.6))
        ax.set_title(title)
        ax.legend(fontsize=7.2, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "l3_split.png")
    plt.close(fig)
    print("wrote l3_split.png")


if __name__ == "__main__":
    l1_curves()
    l2_errors()
    l3_split()
