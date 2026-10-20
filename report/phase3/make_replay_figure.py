"""Replay test of the wave-2 stabilization (docs/PHASE3_PARAMETRIC_PINN.md, "Stabilization and its test").

    python report/phase3/make_replay_figure.py --pair "rate_w96_2xpts_s0:<nofix dir>:<fix dir>" --pair "rate_w128_s0:..."

For each pair: kinetics residuals (ode1 + charge_s, unweighted) along the replay without and with the stabilization,
with the rollbacks marked (from train.log).  Output report/phase3/figures/p3a_replay.png.
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


def series(run_dir: Path, start: int):
    h = json.loads((run_dir / "history.json").read_text())["history"]
    h = [r for r in h if r["step"] > start]
    st = np.array([r["step"] for r in h])
    y = np.array([r.get("ode1", np.nan) + r.get("charge_s", np.nan) for r in h])
    ys = np.array([np.nanmedian(y[max(0, i - 2):i + 3]) for i in range(len(y))])
    rb = [int(m.group(1)) for m in re.finditer(r"step (\d+): ROLLBACK", (run_dir / "train.log").read_text())]
    return st, ys, rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="label:nofix_dir:fix_dir")
    ap.add_argument("--start", action="append", type=int, help="replay start step per pair")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, len(a.pair), figsize=(6.5 * len(a.pair), 4.6), squeeze=False)
    for k, (ax, pair) in enumerate(zip(axs[0], a.pair)):
        label, nofix, fix = pair.split(":")
        start = a.start[k] if a.start else 0
        for d, col, name in [(Path(nofix), "#B91C1C", "wave-1 settings"), (Path(fix), "#2A78D6", "weight guard + rollback")]:
            st, ys, rb = series(d, start)
            ax.semilogy(st, ys, color=col, lw=1.8, label=name)
            for s in rb:
                ax.axvline(s, color=col, lw=1, ls=":")
        ax.set_title(f"Replay of {label} from step {start}", loc="left", fontweight="bold", color=INK, fontsize=13)
        ax.set_xlabel("training step"); ax.set_ylabel("ode1 + charge_s (unweighted)")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=11, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(OUT / "p3a_replay.png")
    print("wrote", OUT / "p3a_replay.png")


if __name__ == "__main__":
    main()
