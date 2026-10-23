"""Wave-3 figure (docs/PHASE3_PARAMETRIC_PINN.md, "Wave 3 results").

    python report/phase3/make_wave3_figure.py --runs results/lispan_rate_runs

p3a_wave3.png, three panels over the training steps:
  a  mean voltage error vs FV at the evaluations of the wave-3 runs (solid; star = best.pt) and of their wave-2 twins
     (same seed and range, dashed);
  b  charge_s, the residual of solid-phase charge conservation (unweighted, 500-step running median), with the
     wave-3 rollbacks: every excursion starts there;
  c  adaptive weight of the charge group: it collapses (to ~0.2) whenever charge_s grows - the positive feedback of
     the plain gradient-norm rule found in wave 1.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
INK, INK2, GRID = "#1B2A3A", "#4A5568", "#E3E8EE"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})
RUNS = {"s1": "#2A78D6", "wide_s0": "#E0622C", "s2": "#1B2A3A", "s0": "#B91C1C"}


def running_median(y, n=50):
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        out[i] = np.nanmedian(y[max(0, i - n + 1):i + 1])
    return out


def load(run_dir: Path):
    h = json.loads((run_dir / "history.json").read_text())
    log = (run_dir / "train.log").read_text(errors="replace") if (run_dir / "train.log").exists() else ""
    best = json.loads((run_dir / "best_metrics.json").read_text()) if (run_dir / "best_metrics.json").exists() else None
    fm = json.loads((run_dir / "final_metrics.json").read_text()) if (run_dir / "final_metrics.json").exists() else {}
    return h["history"], h["evals"], [int(s) for s in re.findall(r"step (\d+): ROLLBACK", log)], best, fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder with rate_v3_* (and rate_v2_* for the twins)")
    a = ap.parse_args()
    runs = Path(a.runs)
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 0.9]})
    for key, col in RUNS.items():
        d3, d2 = runs / f"rate_v3_{key}", runs / f"rate_v2_{key}"
        if (d2 / "history.json").exists():
            _, ev2, _, _, _ = load(d2)
            ax1.semilogy([e["step"] for e in ev2], [e["V_rmse_mean_mV"] for e in ev2], "--", color=col, lw=1.0, alpha=0.45)
        hist, ev, rb, best, fm = load(d3)
        ax1.semilogy([e["step"] for e in ev], [e["V_rmse_mean_mV"] for e in ev], "o-", color=col, lw=1.8, ms=3,
                     label=f"v3_{key}: final {fm.get('V_rmse_mean_mV', np.nan):.2f} mV"
                           + (f", best.pt {best['V_rmse_mean_mV']:.2f} mV" if best else ""))
        if best:
            ax1.plot([best["step"]], [best["V_rmse_mean_mV"]], "*", color=col, ms=15, mec="white", mew=1.0, zorder=5)
        st = np.array([r["step"] for r in hist])
        cs = running_median(np.array([r.get("charge_s", np.nan) for r in hist]))
        ax2.semilogy(st, cs, color=col, lw=1.3)
        if rb:
            ax2.plot(rb, np.interp(rb, st, cs), "v", color=col, ms=6, mec="white", mew=0.6)
        w = [(r["step"], r["w"]["charge"]) for r in hist if "w" in r and "charge" in r["w"]]
        ax3.semilogy([s for s, _ in w], [v for _, v in w], color=col, lw=1.1)
    ax1.plot([], [], "--", color=INK2, lw=1.0, alpha=0.6, label="wave-2 twin (same seed / range)")
    ax1.axhline(2.0, color=INK2, lw=1, ls=":")
    ax1.text(1500, 2.0, "gate 2 mV", va="bottom", ha="left", color=INK2, fontsize=10)
    ax1.set_ylabel("mean V rmse vs FV [mV]")
    ax1.set_title("a  Voltage error at the evaluations;  ★ = best.pt", loc="left", fontweight="bold", color=INK)
    ax1.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=1)
    ax2.set_ylabel("charge_s residual (unweighted)")
    ax2.set_title("b  Solid-phase charge residual, 500-step running median;  ▼ = rollback", loc="left",
                  fontweight="bold", color=INK)
    ax3.set_ylabel("charge-group weight")
    ax3.set_xlabel("training step")
    ax3.set_title("c  Adaptive weight of the charge group (plain rule)", loc="left", fontweight="bold", color=INK)
    ax3.set_xlim(0, 122000)
    fig.tight_layout()
    fig.savefig(OUT / "p3a_wave3.png")
    plt.close(fig)
    print("wrote", OUT / "p3a_wave3.png")


if __name__ == "__main__":
    main()
