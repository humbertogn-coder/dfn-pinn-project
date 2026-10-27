"""Two-curve tool validation figure (docs/TWO_CURVE_TOOL.md).

    python report/phase3/make_two_curve_figure.py

Reads results/lispan/two_curve/<name>.json (network fit, scripts/lispan_two_curve_fit.py) and <name>_fv.json (FV fit
of the same points from the network's solution, scripts/lispan_lit_v2form.py) and writes p3b_two_curve.png:
  a  network fit - FV fit for every parameter, in units of the network fit's linearized sd;
  b  rms misfit to the data: network fit, FV solver at the network's parameters, FV optimum; predictions of a third
     rate where the data have one inside the network's range;
  c  one fit (Hao 2025, 0.2 + 1 C, prediction 0.5 C): data, network, FV at the network's parameters.
Also prints the table of docs/TWO_CURVE_TOOL.md.
"""

import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dfn_pinn.lispan.chem_pinn import load_chem_model  # noqa: E402
from dfn_pinn.lispan.two_curve import Curve, TwoCurveFit  # noqa: E402

RES = ROOT / "results" / "lispan" / "two_curve"
OUT = Path(__file__).resolve().parent / "figures"
INK, INK2, GRID = "#1B2A3A", "#4A5568", "#E3E8EE"
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2A78D6", "#E0622C", "#2E8B57", "#7A4FB5", "#8A94A3"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})
CASES = [("simanjuntak_0.1+1", "Simanjuntak\n0.1+1 C"), ("chen2019_ether_0.4+1", "Chen ether\n0.4+1 A/g"),
         ("chen2019_carbonate_0.4+1", "Chen carb.\n0.4+1 A/g"), ("hao2025_SPAN_0.2+1", "Hao\n0.2+1 C"),
         ("ma2026_SPAN_0.2+0.5", "Ma SPAN\n0.2+0.5 C"), ("ma2026_SPAN-A_0.2+0.5", "Ma SPAN-A\n0.2+0.5 C")]
PARAMS = ["w_S", "dU0_2", "dU0_3rel2", "b_12", "b_3", "k0_1", "d", "Z_CC"]
MARK = dict(zip(PARAMS, ["o", "s", "D", "^", "v", "P", "X", "*"]))
PCOL = dict(zip(PARAMS, [INK, BLUE, "#1C5FA8", ORANGE, "#B84A1C", GREEN, PURPLE, GREY]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, lab in CASES:
        f_net, f_fv = RES / f"{name}.json", RES / f"{name}_fv.json"
        if f_net.exists() and f_fv.exists():
            rows.append((name, lab, json.loads(f_net.read_text()), json.loads(f_fv.read_text())))
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], width_ratios=[1, 1])
    ax1, ax2, ax3 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
    xs = np.arange(len(rows))
    off = np.linspace(-0.3, 0.3, len(PARAMS))
    ax1.axhspan(-1, 1, color=GRID, alpha=0.7, lw=0)
    for k, p in enumerate(PARAMS):
        for i, (_, _, net, fv) in enumerate(rows):
            d = fv["difference"].get(p)
            if d is None or d["in_sd"] is None:
                continue
            edge = any(fl.get("param") == p and fl["flag"] in ("box_edge", "rate_edge") for fl in net["flags"])
            ax1.plot(xs[i] + off[k], d["in_sd"], MARK[p], color=PCOL[p], ms=7 if p != "Z_CC" else 9,
                     mfc="white" if edge else PCOL[p], mew=1.4, zorder=3)
        ax1.plot([], [], MARK[p], color=PCOL[p], ms=7 if p != "Z_CC" else 9, mew=1.4, label=p)
    ax1.axhline(0, color=INK2, lw=0.8)
    ax1.set_xticks(xs); ax1.set_xticklabels([r[1] for r in rows])
    ax1.set_ylabel("(network - FV) / sd")
    ax1.set_title("a  Network fit vs FV fit of the same points: parameter difference in units of the linearized sd "
                  "(open: parameter at the box edge)", loc="left", fontweight="bold", color=INK, fontsize=10.5)
    ax1.legend(frameon=False, ncol=8, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.2))

    w = 0.22
    lab_done = set()
    for i, (_, _, net, fv) in enumerate(rows):
        n_fit = net["rms_mV"]
        fv_at_net = np.sqrt(np.mean([c["rms_fv_at_net_mV"] ** 2 for c in fv["per_curve"]]))
        fv_opt = np.sqrt(np.mean([c["rms_fv_at_fv_mV"] ** 2 for c in fv["per_curve"]]))
        for j, (v, col, lab) in enumerate(((n_fit, BLUE, "network fit"), (fv_at_net, GREY, "FV at the network's parameters"),
                                           (fv_opt, INK, "FV optimum (same points, same box)"))):
            ax2.bar(xs[i] + (j - 1) * w, v, w * 0.92, color=col, label=lab if lab not in lab_done else None)
            lab_done.add(lab)
        for pr, pf in zip(net.get("predictions", []), fv.get("predictions", [])):
            if "rms_mV" in pr:
                ax2.plot(xs[i] + 1.6 * w, pr["rms_mV"], "D", color=ORANGE, ms=7,
                         label="prediction of a third rate (network)" if "pred" not in lab_done else None)
                lab_done.add("pred")
                ax2.text(xs[i] + 1.6 * w + 0.05, pr["rms_mV"], pr["label"], fontsize=8, color=INK2, va="center")
    ax2.set_xticks(xs); ax2.set_xticklabels([r[1] for r in rows])
    ax2.set_ylabel("rms misfit to the data [mV]")
    ax2.set_title("b  Misfit: the network reaches the FV optimum; the misfit itself is the continuum model's",
                  loc="left", fontweight="bold", color=INK, fontsize=10.5)
    ax2.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")

    ex = next((r for r in rows if r[0].startswith("hao2025")), rows[0])
    net = ex[2]
    model = load_chem_model(net["network"])[0]
    curves = [Curve(c["i_sp"], c["q"], c["V"], c["label"]) for c in net["curves"]]
    fit = TwoCurveFit(model, curves, fixed={k: float(v) for k, v in (it.split("=") for it in net["args"]["fix"])} or None)
    Vm = fit.model_curves(np.asarray(net["x"]))
    cols = [BLUE, ORANGE, GREEN]
    for k, (c, vm) in enumerate(zip(curves, Vm)):
        ax3.plot(c.q, c.V, "o", color=cols[k], ms=3.2, alpha=0.55)
        ax3.plot(c.q, vm, "-", color=cols[k], lw=1.8, label=f"{c.label}: network fit ({net['rms_per_curve_mV'][k]:.1f} mV)")
        fvc = net["fv_check"]["curves"][k] if net.get("fv_check") else None
        if fvc and fvc.get("V"):
            ax3.plot(c.q, [np.nan if v is None else v for v in fvc["V"]], ":", color=INK, lw=1.2)
    for pr in net.get("predictions", []):
        if "V" in pr:
            ax3.plot(pr["q"], pr["V_data"], "o", color=cols[2], ms=3.2, alpha=0.55)
            ax3.plot(pr["q"], pr["V"], "--", color=cols[2], lw=1.8,
                     label=f"{pr['label']}: prediction ({pr['rms_mV']:.1f} mV)")
    ax3.plot([], [], ":", color=INK, lw=1.2, label="FV solver at the network's parameters")
    ax3.plot([], [], "o", color=GREY, ms=3.2, label="data (Hao et al. 2025, SI Fig. S9)")
    ax3.set_xlabel("capacity [mAh/g SPAN]"); ax3.set_ylabel("voltage [V]")
    ax3.set_title(f"c  Example: {ex[1].replace(chr(10), ' ')}, prediction of 0.5 C", loc="left", fontweight="bold",
                  color=INK, fontsize=10.5)
    ax3.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "p3b_two_curve.png"); plt.close(fig)
    print("wrote", OUT / "p3b_two_curve.png")

    print("| data | free | network rms [mV] | FV at network [mV] | FV optimum [mV] | net - FV, max over parameters [sd] "
          "| prediction (network / FV optimum) [mV] | flags |")
    print("|---|---|---|---|---|---|---|---|")
    for name, lab, net, fv in rows:
        fv_at_net = np.sqrt(np.mean([c["rms_fv_at_net_mV"] ** 2 for c in fv["per_curve"]]))
        fv_opt = np.sqrt(np.mean([c["rms_fv_at_fv_mV"] ** 2 for c in fv["per_curve"]]))
        big = max(((abs(d["in_sd"]), p) for p, d in fv["difference"].items() if d["in_sd"] is not None))
        preds = "; ".join(f"{p['label']}: {p.get('rms_net_at_net_mV', float('nan')):.1f} / {p.get('rms_fv_at_fv_mV', float('nan')):.1f}"
                          for p in fv.get("predictions", [])) or "-"
        flags = ", ".join(sorted({f["flag"] + (f" {f['param']}" if "param" in f else "") for f in net["flags"]
                                  if f["level"] != "info"}))
        print(f"| {lab.replace(chr(10), ' ')} | {len(net['sd'])} | {net['rms_mV']:.1f} | {fv_at_net:.1f} | {fv_opt:.1f} | "
              f"{big[0]:.2f} ({big[1]}) | {preds} | {flags} |")


if __name__ == "__main__":
    main()
