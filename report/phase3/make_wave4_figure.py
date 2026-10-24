"""Wave-4 figure (docs/PHASE3_PARAMETRIC_PINN.md, "Wave 4 results").

    python report/phase3/make_wave4_figure.py --runs results/lispan_rate_runs

p3a_wave4.png, one panel per seed / range (s0, s1, s2, wide_s0): mean voltage error vs FV at the evaluations of the
wave-4 run (solid; star = best.pt) and of its wave-3 twin (dashed).  The two runs are identical step by step until the
first wave-3 rollback (circle on the dashed line); ticks at the bottom mark the rollbacks of each run.  Shaded: before
step 30 000 wave 4 had no rollbacks; dotted line: adaptive weights frozen at 50 000 (wave 4).
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
C4, C3 = "#2A78D6", "#8A94A3"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})
KEYS = [("s0", "0.1-1 C, seed 0"), ("s1", "0.1-1 C, seed 1"), ("s2", "0.1-1 C, seed 2"), ("wide_s0", "0.08-1.25 C, seed 0")]


def find(dirs, name):
    for d in dirs:
        p = Path(d) / name
        if (p / "history.json").exists():
            return p
    return None


def load(run_dir: Path):
    h = json.loads((run_dir / "history.json").read_text())
    log = (run_dir / "train.log").read_text(errors="replace")
    best = json.loads((run_dir / "best_metrics.json").read_text()) if (run_dir / "best_metrics.json").exists() else None
    fm = json.loads((run_dir / "final_metrics.json").read_text()) if (run_dir / "final_metrics.json").exists() else {}
    return h["history"], h["evals"], [int(s) for s in re.findall(r"step (\d+): ROLLBACK", log)], best, fm


def first_difference(h_a, h_b):
    a = {r["step"]: r["loss"] for r in h_a}
    return next((s for s in sorted(a) if s in {r["step"] for r in h_b} and a[s] != {r["step"]: r["loss"] for r in h_b}[s]),
                None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="folder(s) holding rate_v4_* and rate_v3_*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    for ax, (key, title) in zip(axes.flat, KEYS):
        d4, d3 = find(a.runs, f"rate_v4_{key}"), find(a.runs, f"rate_v3_{key}")
        ax.axvspan(0, 30000, color=GRID, alpha=0.6, lw=0)
        ax.axvline(50000, color=INK2, lw=1, ls=":")
        ax.axhline(2.0, color=INK2, lw=0.8, ls="--", alpha=0.6)
        lo = 0.05
        if d3:
            h3, ev3, rb3, best3, fm3 = load(d3)
            ax.semilogy([e["step"] for e in ev3], [e["V_rmse_mean_mV"] for e in ev3], "--", color=C3, lw=1.5,
                        label=f"wave 3: final {fm3.get('V_rmse_mean_mV', np.nan):.2f}"
                              + (f", best.pt {best3['V_rmse_mean_mV']:.2f}" if best3 else "") + " mV")
            if best3:
                ax.plot([best3["step"]], [best3["V_rmse_mean_mV"]], "*", color=C3, ms=13, mec="white", mew=0.8, zorder=5)
            ax.plot(rb3, np.full(len(rb3), lo * 1.25), "|", color=C3, ms=7, mew=1.0)
        h4, ev4, rb4, best4, fm4 = load(d4)
        ax.semilogy([e["step"] for e in ev4], [e["V_rmse_mean_mV"] for e in ev4], "o-", color=C4, lw=2.0, ms=3,
                    label=f"wave 4: final {fm4.get('V_rmse_mean_mV', np.nan):.2f}"
                          + (f", best.pt {best4['V_rmse_mean_mV']:.2f}" if best4 else "") + " mV")
        if best4:
            ax.plot([best4["step"]], [best4["V_rmse_mean_mV"]], "*", color=C4, ms=15, mec="white", mew=0.8, zorder=6)
        ax.plot(rb4, np.full(len(rb4), lo * 1.8), "|", color=C4, ms=7, mew=1.0)
        if d3:
            s = first_difference(h3, h4)
            if s is not None:
                ax.axvline(s, color=C3, lw=0.8, alpha=0.7)
                ax.text(s + 1500, 600, f"paths split\nat step {s}", color=INK2, fontsize=8.5, va="top")
        ax.set_ylim(lo, 2000)
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean V rmse vs FV [mV]")
    for ax in axes[1, :]:
        ax.set_xlabel("training step")
        ax.set_xlim(0, 122000)
    fig.text(0.01, -0.01, "Shaded: no rollbacks in wave 4 (rollback_start 30 000).  Dotted: wave-4 weights frozen (50 000).  "
             "Dashed grey line: 2 mV gate.  Ticks at the bottom: rollbacks (grey wave 3, blue wave 4).  ★ best.pt.",
             fontsize=9, color=INK2, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "p3a_wave4.png")
    plt.close(fig)
    print("wrote", OUT / "p3a_wave4.png")


if __name__ == "__main__":
    main()
