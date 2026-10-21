"""Wave-2 diagnosis figure (docs/PHASE3_PARAMETRIC_PINN.md, "Wave 2 results").

    python report/phase3/make_wave2_figures.py --wave2 results/lispan_rate_runs --wave1 results/lispan_rate_runs

p3a_wave2.png, three panels over the training steps:
  a  mean voltage error vs FV at the evaluations, wave-2 runs and the wave-1 run with the same seed and settings
     but without the weight guard / rollback (rate_w96_120k_s0);
  b  summed kinetics losses (ode1-3, charge_s, charge_e; unweighted; 500-step running median) with the rollbacks;
  c  adaptive weight of the salt group: the weight guard pins it at the cap (1000) in the two failed runs, while the
     plain rule of wave 1 keeps it at 6-21 (checkpoints at 10 000 steps and the final ones).
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
RUNS = {"rate_v2_s1": ("#2A78D6", "v2_s1 (0.24 mV)"), "rate_v2_s2": ("#1B2A3A", "v2_s2 (0.44 mV)"),
        "rate_v2_s0": ("#B91C1C", "v2_s0 (262 mV)"), "rate_v2_wide_s0": ("#E0622C", "v2_wide_s0 (1040 mV)")}
KIN = ("ode1", "ode2", "ode3", "charge_s", "charge_e")
WAVE1_SALT = (5.7, 21.2)        # salt weight in the wave-1 checkpoints (step 10 000 and final): plain rule


def running_median(y, n=50):
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        out[i] = np.nanmedian(y[max(0, i - n + 1):i + 1])
    return out


def load(run_dir: Path):
    h = json.loads((run_dir / "history.json").read_text())
    log = (run_dir / "train.log").read_text(errors="replace") if (run_dir / "train.log").exists() else ""
    return h["history"], h["evals"], [int(s) for s in re.findall(r"step (\d+): ROLLBACK", log)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave2", required=True)
    ap.add_argument("--wave1", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                                        gridspec_kw={"height_ratios": [1.1, 1, 0.9]})
    hist1, ev1, _ = load(Path(a.wave1) / "rate_w96_120k_s0")
    ax1.semilogy([e["step"] for e in ev1], [e["V_rmse_mean_mV"] for e in ev1], "--", color="#8E959E", lw=1.8,
                 label="wave 1, w96_120k_s0: same seed,\nno guard / rollback (0.15 mV)")
    for name, (col, lab) in RUNS.items():
        hist, ev, rb = load(Path(a.wave2) / name)
        ax1.semilogy([e["step"] for e in ev], [e["V_rmse_mean_mV"] for e in ev], "o-", color=col, lw=1.8, ms=3, label=lab)
        st = np.array([r["step"] for r in hist])
        kin = running_median(np.array([sum(r.get(k, np.nan) for k in KIN) for r in hist]))
        ax2.semilogy(st, kin, color=col, lw=1.4)
        if rb:
            ax2.plot(rb, np.interp(rb, st, kin), "v", color=col, ms=7, mec="white", mew=0.8)
        w = [(r["step"], r["w"]["salt"]) for r in hist if "w" in r and "salt" in r["w"]]
        ax3.semilogy([s for s, _ in w], [v for _, v in w], color=col, lw=1.4)
    st1 = np.array([r["step"] for r in hist1])
    ax2.semilogy(st1, running_median(np.array([sum(r.get(k, np.nan) for k in KIN) for r in hist1])), "--",
                 color="#8E959E", lw=1.4)
    ax1.axhline(2.0, color=INK2, lw=1, ls=":")
    ax1.text(47000, 2.0, "gate 2 mV", va="bottom", ha="left", color=INK2, fontsize=10)
    ax1.set_ylabel("mean V rmse vs FV [mV]")
    ax1.set_title("a  Voltage error at the evaluations (7 rates; wide run: 9 rates)", loc="left", fontweight="bold",
                  color=INK)
    ax1.legend(frameon=False, fontsize=10, loc="center right", bbox_to_anchor=(1.0, 0.45))
    ax2.set_ylabel("kinetics losses (unweighted)")
    ax2.set_title("b  Kinetics residuals, 500-step running median;  ▼ = rollback", loc="left", fontweight="bold",
                  color=INK)
    ax3.axhspan(*WAVE1_SALT, color="#8E959E", alpha=0.25, lw=0)
    ax3.text(121000, WAVE1_SALT[0] * 0.85, "wave 1 (plain rule): 6-21", color=INK2, fontsize=10, va="top", ha="right")
    ax3.axhline(1000, color=INK2, lw=1, ls=":")
    ax3.text(121000, 1000, "cap 1000", va="bottom", ha="right", color=INK2, fontsize=10)
    ax3.set_ylabel("salt-group weight")
    ax3.set_xlabel("training step")
    ax3.set_title("c  Adaptive weight of the salt residuals (weight guard on)", loc="left", fontweight="bold", color=INK)
    ax3.set_xlim(0, 122000)
    fig.tight_layout()
    fig.savefig(OUT / "p3a_wave2.png")
    plt.close(fig)
    print("wrote", OUT / "p3a_wave2.png")


if __name__ == "__main__":
    main()
