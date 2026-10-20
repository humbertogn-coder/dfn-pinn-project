"""Figures for Phase 3a (rate-parametric Li-SPAN PINN), docs/PHASE3_PARAMETRIC_PINN.md.

    python report/phase3/make_figures.py [--runs results/lispan_rate_runs] [--best rate_w96_120k_s0]

p3a_curves.png   : discharge curves of ONE trained network at the seven evaluation rates vs the finite-volume solver,
                   and the voltage error along each curve.
p3a_training.png : mean voltage error over the seven rates vs training step for every run of wave 1 (shows the
                   excursions discussed in the doc).
FV references: results/lispan/pinn_reference_<rate>C_rii_Zcc0.025.npz (created by scripts/lispan_rate_train.py).
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dfn_pinn.lispan import load  # noqa: E402
from dfn_pinn.lispan.rate_pinn import load_rate_model, window_t_end  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
INK, INK2, GRID = "#1B2A3A", "#4A5568", "#E3E8EE"
RATE_COLORS = ["#17467F", "#1F5FAE", "#2A78D6", "#5596E6", "#E0622C", "#C2410C", "#7C2D12"]
plt.rcParams.update({"font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})


def curves(run_dir: Path):
    model, cfg, params = load_rate_model(run_dir / "final.pt")
    rev = "".join("r" if r else "i" for r in params.reversible)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    for rate, col in zip(cfg.eval_rates, RATE_COLORS):
        ref = load(ROOT / "results" / "lispan" / f"pinn_reference_{rate:g}C_{rev}_Zcc{params.Z_CC:g}.npz")
        te = window_t_end(model.window_coef.tolist(), rate, model.window_fraction)
        t = np.asarray(ref["t"]); keep = t <= te
        t = t[keep]
        T = torch.as_tensor(t / te, dtype=torch.float32).view(-1, 1)
        r = torch.full_like(T, float(rate))
        with torch.no_grad():
            V = model.voltage(T, r).view(-1).numpy()
            cap = (model.charge_passed(T, r).view(-1).numpy() / 3.6) / (params.m_S_model * 1e3)
        Vfv = np.asarray(ref["V"])[keep]
        m = t > 100.0               # as in the evaluation: the first 100 s (current ramp, t = 0 singularity) left out
        a1.plot(cap[m], Vfv[m], color="#C9C5BB", lw=5, solid_capstyle="round")
        a1.plot(cap[m], V[m], color=col, lw=1.6, ls=(0, (4, 2)), label=f"{rate:g} C")
        a2.plot(cap[m], 1e3 * (V[m] - Vfv[m]), color=col, lw=1.3, label=f"{rate:g} C")
    a1.plot([], [], color="#C9C5BB", lw=5, label="FV solver")
    a1.set_xlabel("capacity [mAh/g$_S$]"); a1.set_ylabel("voltage [V]  (t > 100 s)")
    a1.set_title("One parametric network, seven C-rates", loc="left", fontweight="bold", color=INK)
    a1.legend(frameon=False, fontsize=10, ncol=2, loc="lower left")
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xlabel("capacity [mAh/g$_S$]"); a2.set_ylabel("PINN - FV [mV]")
    a2.set_title("Voltage error along each curve", loc="left", fontweight="bold", color=INK)
    lim = max(1.0, float(np.nanmax(np.abs(a2.get_ylim()))))
    a2.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(OUT / "p3a_curves.png")
    plt.close(fig)


def training(runs_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    styles = {"rate_w96_120k_s0": ("#2A78D6", 2.4), "rate_w96_s1": ("#1B2A3A", 1.6), "rate_w96_s0": ("#8E959E", 1.6),
              "rate_w96_2xpts_s0": ("#E0622C", 1.6), "rate_w128_s0": ("#B91C1C", 1.6)}
    for d in sorted(runs_dir.iterdir()):
        hist = d / "history.json"
        if not hist.exists():
            continue
        ev = json.loads(hist.read_text()).get("evals", [])
        if not ev:
            continue
        col, lw = styles.get(d.name, ("#6B7280", 1.2))
        ax.semilogy([e["step"] for e in ev], [e["V_rmse_mean_mV"] for e in ev], "o-", color=col, lw=lw, ms=3.5,
                    label=d.name.replace("rate_", ""))
    ax.axhline(2.0, color=INK2, lw=1, ls="--")
    ax.text(ax.get_xlim()[1], 2.0, " gate 2 mV", va="bottom", ha="right", color=INK2, fontsize=10)
    ax.set_xlabel("training step"); ax.set_ylabel("mean V rmse over 7 rates [mV]")
    ax.set_title("Wave 1 training histories", loc="left", fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=10, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "p3a_training.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "lispan_rate_runs"))
    ap.add_argument("--best", default="rate_w96_120k_s0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = Path(a.runs)
    curves(runs / a.best)
    training(runs)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
