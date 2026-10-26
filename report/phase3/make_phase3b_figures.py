"""Phase 3b v1 figures (docs/PHASE3B_CHEM_PINN.md, "v1 results").

    python report/phase3/make_phase3b_figures.py --runs results/lispan_chem_runs --best chem_v1_s2 \\
        --recovery recovery_chem_v1_s2_final.json

p3b_v1_errors.png
  a  voltage rms error vs FV for the 24 evaluation cases (8 chemistries x 0.1 / 0.3 / 1 C), every run (final.pt); the
     best run also outside the voltage cliffs (where the FV curve falls faster than 5 V per discharge, +-1 % of the
     capacity around them);
  b  the worst case of the best run (p7 at 1 C): FV and network curves and the error, which sits at the two cliffs
     at the end of a plateau.
p3b_v1_recovery.png
  noise-free parameter error of the two-curve fit (0.1 + 1 C; scripts/lispan_chem_recovery.py) for the 8 chemistries
  against the scatter of the same fit with 2 and 5 mV noise (linearized standard deviation).
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dfn_pinn.lispan.chem_pinn import chem_vector, load_chem_model  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
INK, INK2, GRID = "#1B2A3A", "#4A5568", "#E3E8EE"
BLUE, ORANGE, GREY = "#2A78D6", "#E0622C", "#8A94A3"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})
SLOPE, PAD = 5.0, 0.01


def case_curve(model, case):
    z = np.load(ROOT / case["ref"])
    t, Vr = np.asarray(z["t"], float), np.asarray(z["V"], float)
    vals = chem_vector(model.chem_inputs, case["chem"])
    te = model.window_end(case["rate"], vals)
    keep = (t <= te) & (t > 100.0)
    T = torch.as_tensor(t[keep] / te, dtype=next(model.parameters()).dtype).view(-1, 1)
    with torch.no_grad():
        V = model.voltage(T, model.make_z(case["rate"], vals, int(keep.sum()))).view(-1).double().numpy()
    return t[keep] / t[-1], Vr[keep], V


def rms_split(x, Vr, V):
    e = 1e3 * (V - Vr)
    slope = -np.gradient(Vr, x)
    mask = np.zeros(len(x), bool)
    for i in np.nonzero(slope > SLOPE)[0]:
        mask |= np.abs(x - x[i]) < PAD
    return float(np.sqrt(np.mean(e ** 2))), float(np.sqrt(np.mean(e[~mask] ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--best", default="chem_v1_s2")
    ap.add_argument("--recovery", default=None)
    ap.add_argument("--design", default=str(ROOT / "configs" / "lispan_chem_design_v1.json"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    design = json.loads(Path(a.design).read_text())
    cases = design["eval_cases"]
    runs = sorted(p.name for p in Path(a.runs).iterdir() if (p / "final.pt").exists())
    models = {r: load_chem_model(Path(a.runs) / r / "final.pt")[0] for r in runs}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.6), gridspec_kw={"height_ratios": [1.15, 1]})
    xs = np.arange(len(cases))
    for r in runs:
        tot = [rms_split(*case_curve(models[r], c))[0] for c in cases]
        if r == a.best:
            split = [rms_split(*case_curve(models[r], c)) for c in cases]
            ax1.semilogy(xs, [s[0] for s in split], "o", color=BLUE, ms=7, zorder=4, label=f"{r} (best): all points")
            ax1.semilogy(xs, [s[1] for s in split], "o", mfc="white", mec=BLUE, mew=1.6, ms=7, zorder=5,
                         label=f"{r}: outside the voltage cliffs")
        else:
            ax1.semilogy(xs, tot, "_", color=GREY, ms=11, mew=1.6, zorder=3)
    ax1.plot([], [], "_", color=GREY, ms=11, mew=1.6, label="other runs (all points)")
    n_sel = sum(c["select"] for c in cases)
    ax1.axvspan(n_sel - 0.5, len(cases) - 0.5, color=GRID, alpha=0.6, lw=0)
    ax1.text(n_sel, 0.115, "test chemistries (never used in training or selection)", color=INK2, fontsize=9.5,
             va="bottom")
    ax1.axhline(2.0, color=INK2, lw=0.8, ls="--"); ax1.axhline(5.0, color=INK2, lw=0.8, ls=":")
    ax1.set_xticks(xs); ax1.set_xticklabels([c["id"].replace("_", "\n") for c in cases], fontsize=8)
    ax1.set_ylim(0.1, 15); ax1.set_ylabel("V rms error vs FV [mV]")
    ax1.set_title("a  Voltage error at the 24 evaluation cases (final.pt); dashed 2 mV, dotted 5 mV", loc="left",
                  fontweight="bold", color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    worst = max(cases, key=lambda c: rms_split(*case_curve(models[a.best], c))[0])
    x, Vr, V = case_curve(models[a.best], worst)
    ax2.plot(x, Vr, color=INK, lw=2.2, label="FV reference")
    ax2.plot(x, V, color=BLUE, lw=1.3, ls="--", label=f"{a.best}")
    ax2.set_ylabel("voltage [V]"); ax2.set_xlabel("time / FV cut-off time")
    ax2b = ax2.inset_axes([0.55, 0.52, 0.42, 0.42])
    ax2b.plot(x, 1e3 * (V - Vr), color=ORANGE, lw=1.2)
    ax2b.set_title("error [mV]", fontsize=9, color=INK2); ax2b.tick_params(labelsize=8); ax2b.grid(True, color=GRID)
    ch = ", ".join(f"{k} {v:+.3f}" for k, v in worst["chem"].items())
    ax2.set_title(f"b  Worst case {worst['id']} ({ch}): the error sits at the cliffs", loc="left", fontweight="bold",
                  color=INK, fontsize=10.5)
    ax2.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "p3b_v1_errors.png"); plt.close(fig)
    print("wrote", OUT / "p3b_v1_errors.png")

    if a.recovery:
        rec = json.loads(Path(a.recovery).read_text())
        units = rec["units"]
        res = rec["results"]
        params = list(res[0]["err_mean"])
        chems = sorted({r["chem"] for r in res})
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8))
        for ax, p in zip(axes.flat, params):
            for i, ch in enumerate(chems):
                r0 = [r for r in res if r["chem"] == ch and r["noise_mV"] == 0]
                for nz, col, al in ((5.0, GREY, 0.25), (2.0, BLUE, 0.25)):
                    rn = [r for r in res if r["chem"] == ch and r["noise_mV"] == nz]
                    if rn:
                        s = rn[0]["sd_linear"][p]
                        ax.add_patch(plt.Rectangle((i - 0.3, -s), 0.6, 2 * s, color=col, alpha=al, lw=0))
                if r0:
                    ax.plot(i, r0[0]["err_mean"][p], "o", color=ORANGE, ms=7, zorder=5)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(len(chems))); ax.set_xticklabels(chems, fontsize=9)
            u = units.get(p, "")
            ax.set_title(f"{p} [{'%' if '%' in u else u}]", loc="left", fontweight="bold", color=INK, fontsize=10.5)
        axes.flat[0].plot([], [], "o", color=ORANGE, label="noise-free fit: error from the network")
        axes.flat[0].add_patch(plt.Rectangle((0, 0), 0, 0, color=BLUE, alpha=0.25, label="+-1 sd, 2 mV noise"))
        axes.flat[0].add_patch(plt.Rectangle((0, 0), 0, 0, color=GREY, alpha=0.25, label="+-1 sd, 5 mV noise"))
        fig.legend(*axes.flat[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(f"Two-curve fit (0.1 + 1 C) of FV curves with {a.best}: parameter errors for the 8 chemistries",
                     x=0.01, ha="left", fontweight="bold", color=INK, fontsize=11.5)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        fig.savefig(OUT / "p3b_v1_recovery.png"); plt.close(fig)
        print("wrote", OUT / "p3b_v1_recovery.png")


if __name__ == "__main__":
    main()
