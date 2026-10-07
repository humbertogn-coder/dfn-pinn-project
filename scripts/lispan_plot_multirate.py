"""Parameter trajectories of one or more Li-SPAN multi-rate inverse runs (one panel per run).

    python scripts/lispan_plot_multirate.py results/lispan_runs/<run A> [<run B> ...] --out fig.png
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.runs), figsize=(6.5 * len(args.runs), 4.4), squeeze=False)
    for ax, run in zip(axes[0], args.runs):
        run = Path(run)
        spec = json.loads((run / "spec.json").read_text())
        names = spec["train"]["inverse_params"]
        hist = json.loads((run / "history.json").read_text())["history"]
        steps = np.array([h["step"] for h in hist])
        for n in names:
            v = np.array([h["params"][n] for h in hist])
            ax.plot(steps, 1e3 * v if n.startswith("U0") else 100 * (v - 1), label=n + (" [mV]" if n.startswith("U0") else " [%]"), lw=1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("step"); ax.set_ylabel("error vs truth")
        ax.set_title(run.name, fontsize=8)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("written", args.out)


if __name__ == "__main__":
    sys.exit(main())
