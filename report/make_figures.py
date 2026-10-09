"""Figures for the student guide (report/PINN_DFN_LiSPAN_student_guide.docx).

    python report/make_figures.py            # writes report/figures/*.png

Every data figure is regenerated from the saved runs / references of the repository, so each number in the guide
can be traced back to a file.  Schematic figures (cell, PINN, valley) are drawings, labelled as such.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical palette (reference instance of the dataviz method), light mode, fixed order
C1, C2, C3, C4, C5, C6, C7, C8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 11})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc, ec=INK2, fs=9, color=INK, round_=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={round_}", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, wrap=True)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.5, style="-|>", ms=12, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls))


# ---------------------------------------------------------------------------------------------- schematics
def fig_cell():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis("off")
    # collectors
    ax.add_patch(Rectangle((0.3, 0.6), 0.25, 3.4, fc="#c9c9c9", ec=INK2))
    ax.add_patch(Rectangle((9.45, 0.6), 0.25, 3.4, fc="#c9c9c9", ec=INK2))
    ax.text(0.42, 0.35, "Cu", ha="center", fontsize=8, color=INK2); ax.text(9.57, 0.35, "Al", ha="center", fontsize=8, color=INK2)
    # regions
    ax.add_patch(Rectangle((0.55, 0.6), 3.3, 3.4, fc="#eef4fc", ec=INK2))
    ax.add_patch(Rectangle((3.85, 0.6), 1.9, 3.4, fc="#f6f6f4", ec=INK2, hatch="////", alpha=0.6))
    ax.add_patch(Rectangle((5.75, 0.6), 3.7, 3.4, fc="#fdf0ea", ec=INK2))
    rng = np.random.default_rng(3)
    for _ in range(14):
        ax.add_patch(Circle((rng.uniform(0.9, 3.5), rng.uniform(0.95, 3.65)), 0.28, fc=C1, ec="white", lw=1.2, alpha=0.85))
    for _ in range(16):
        ax.add_patch(Circle((rng.uniform(6.1, 9.1), rng.uniform(0.95, 3.65)), 0.24, fc=C2, ec="white", lw=1.2, alpha=0.85))
    ax.text(2.2, 4.25, "negative electrode\n(graphite particles)", ha="center", fontsize=9, color=INK)
    ax.text(4.8, 4.25, "separator\n(electrolyte only)", ha="center", fontsize=9, color=INK)
    ax.text(7.6, 4.25, "positive electrode\n(NMC particles  -  or SPAN)", ha="center", fontsize=9, color=INK)
    # ion arrow
    arrow(ax, 3.0, 2.3, 7.0, 2.3, color=C3, lw=2.5, ms=16)
    ax.text(4.8, 2.6, r"Li$^+$ ions through the electrolyte", ha="center", fontsize=9, color=INK,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
    # external circuit
    ax.plot([0.42, 0.42, 9.57, 9.57], [4.0, 4.85, 4.85, 4.0], color=INK2, lw=1.5)
    arrow(ax, 3.0, 4.85, 7.0, 4.85, color=C7, lw=2.0)
    ax.text(5.0, 5.05, r"electrons through the external wire during discharge (this current does useful work)", ha="center", fontsize=8.5, color=INK)
    ax.text(5.0, 0.15, r"x (through the cell, about 0.2 mm in total)  $\rightarrow$", ha="center", fontsize=8.5, color=INK2)
    save(fig, "fig01_cell_schematic.png")


def fig_pinn_concept():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.2, 1.7, 1.3, 1.2, "inputs\nposition x\ntime t", "#eef4fc")
    # network
    xs = [2.2, 3.0, 3.8]
    for i, xx in enumerate(xs):
        for yy in np.linspace(1.3, 3.3, 5):
            ax.add_patch(Circle((xx, yy), 0.11, fc="white", ec=C1, lw=1.5))
            if i < 2:
                for y2 in np.linspace(1.3, 3.3, 5):
                    ax.plot([xx, xs[i + 1]], [yy, y2], color="#bcd3f0", lw=0.5, zorder=0)
    ax.text(3.0, 0.85, "neural network\n(weights = knobs)", ha="center", fontsize=9)
    arrow(ax, 1.5, 2.3, 2.05, 2.3)
    arrow(ax, 3.95, 2.3, 4.5, 2.3)
    box(ax, 4.5, 1.6, 1.6, 1.4, "outputs\nconcentrations\npotentials\ncurrents", "#eef4fc")
    box(ax, 6.6, 2.75, 1.7, 1.2, "physics residuals\n(how badly the\nequations fail)", "#fdf0ea")
    box(ax, 6.6, 0.75, 1.7, 1.2, "data misfit\n(only for the\ninverse problem)", "#f6f6f4")
    arrow(ax, 6.1, 2.6, 6.6, 3.3); arrow(ax, 6.1, 2.0, 6.6, 1.4)
    ax.text(6.0, 4.25, "derivatives by automatic differentiation", fontsize=8, ha="center", color=INK2)
    box(ax, 8.8, 1.7, 1.0, 1.2, "loss\n= sum of\nsquares", "#e8f6f0")
    arrow(ax, 8.3, 3.3, 8.8, 2.6); arrow(ax, 8.3, 1.4, 8.8, 2.0)
    arrow(ax, 9.3, 1.65, 9.3, 0.35, color=C2)
    ax.plot([9.3, 3.0], [0.35, 0.35], color=C2, lw=1.5)
    arrow(ax, 3.0, 0.35, 3.0, 0.62, color=C2)
    ax.text(6.2, 0.1, "optimizer turns the knobs to make the loss small (thousands of times)", ha="center", fontsize=8.5, color=C2)
    save(fig, "fig03_pinn_concept.png")


def fig_forward_inverse():
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    box(ax, 0.3, 1.0, 2.4, 1.1, "parameters\n(diffusivities, rate\nconstants, resistances)", "#eef4fc")
    box(ax, 3.8, 1.0, 2.4, 1.1, "model\n(equations: DFN or\nLi-SPAN)", "#f6f6f4")
    box(ax, 7.3, 1.0, 2.4, 1.1, "predicted voltage\nV(t)", "#fdf0ea")
    arrow(ax, 2.75, 1.8, 3.75, 1.8, color=C1, lw=2); arrow(ax, 6.25, 1.8, 7.25, 1.8, color=C1, lw=2)
    ax.text(5.0, 2.55, "FORWARD problem: parameters known  ->  predict the voltage", ha="center", color=C1, fontsize=9.5, weight="bold")
    arrow(ax, 7.25, 1.3, 6.25, 1.3, color=C2, lw=2); arrow(ax, 3.75, 1.3, 2.75, 1.3, color=C2, lw=2)
    ax.text(5.0, 0.35, "INVERSE problem: voltage measured  ->  infer the parameters", ha="center", color=C2, fontsize=9.5, weight="bold")
    save(fig, "fig04_forward_inverse.png")


def fig_valley():
    """Illustration (not data): a long, narrow valley and two optimizers."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    a, b = np.meshgrid(np.linspace(-2.2, 2.2, 300), np.linspace(-2.2, 2.2, 300))
    rho = 0.95
    f = (a ** 2 - 2 * rho * a * b + b ** 2) / (1 - rho ** 2)
    ax.contour(a, b, np.log1p(f), levels=14, colors=GRID, linewidths=1.0)
    ax.contourf(a, b, np.log1p(f), levels=14, cmap="Blues_r", alpha=0.35)
    # gradient-descent-like path: zig-zag, slow along the valley
    p = np.array([1.8, -0.6]); path = [p.copy()]
    H = np.array([[1, -rho], [-rho, 1]]) * 2 / (1 - rho ** 2)
    for _ in range(60):
        p = p - 0.045 * H @ p; path.append(p.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "-o", ms=2.5, color=C2, lw=1.4, label="first-order steps (Adam-like): zig-zag, slow")
    ax.plot([1.8, 0.0], [-0.6, 0.0], "-s", ms=5, color=C1, lw=2.2, label="Newton/LM step: uses the valley shape")
    ax.plot(0, 0, "*", ms=14, color=C3, label="true parameters")
    ax.set_xlabel("parameter A (e.g. log k0_1)"); ax.set_ylabel("parameter B (e.g. U0_1)")
    ax.set_title("Illustration: a correlated (narrow-valley) misfit")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    save(fig, "fig14_valley_illustration.png")


def fig_lispan_network():
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")
    spec = [(0.2, "PAN-S$_4$-PAN\n(start: 4 S per bridge)", "#eef4fc"), (2.75, "PAN-S$_3$Li\n+ PAN-SLi", "#e8f6f0"),
            (5.3, "PAN-S$_2$Li\n+ S$^{2-}$ (+ Li$_2$S)", "#fdf0ea"), (7.85, "PAN-SLi\n+ S$^{2-}$ (+ Li$_2$S)", "#f6f6f4")]
    for x, t, c in spec:
        box(ax, x, 1.6, 2.0, 1.1, t, c, fs=9)
    labels = ["reaction 1\n+2 e$^-$\nnear 2.2 V", "reaction 2\n+2 e$^-$\nnear 1.9 V", "reaction 3\n+2 e$^-$\nnear 1.66 V"]
    for k, (x0, x1) in enumerate(((2.2, 2.75), (4.75, 5.3), (7.3, 7.85))):
        arrow(ax, x0, 2.15, x1, 2.15, color=[C1, C3, C2][k], lw=2.2)
        ax.text((x0 + x1) / 2, 0.75, labels[k], ha="center", fontsize=8, color=INK)
    ax.text(5.0, 3.15, "each step adds 2 electrons per sulfur bridge: 3 steps x 2 e$^-$ = 6 e$^-$ per chain (theoretical 1254 mAh/g of S)",
            ha="center", fontsize=8.5, color=INK2)
    ax.text(5.0, 0.15, "dissolved S$^{2-}$ meets Li$^+$ and precipitates as solid Li$_2$S when the solubility product is exceeded",
            ha="center", fontsize=8.5, color=INK2)
    save(fig, "fig09_lispan_reactions.png")


# ---------------------------------------------------------------------------------------------- DFN data figures
def fig_dfn_rates():
    from dfn_pinn.v2.reference import load
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for f, lab, c in (("ref_I2.5A_t6500s_ramp30s_x80_r120.npz", "C/2 (2.5 A)", C1), ("ref_I5A_t3000s_ramp30s_x80_r120.npz", "1C (5 A)", C2),
                      ("ref_I10A_t1400s_ramp30s_x80_r120.npz", "2C (10 A)", C3)):
        r = load(ROOT / "results" / "v2_reference" / f)
        t, V, I = np.asarray(r["t"]), np.asarray(r["V"]), np.asarray(r["I"])
        Q = np.concatenate([[0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))]) / 3600.0
        ax.plot(Q, V, color=c, label=lab)
    ax.set_xlabel("charge delivered [Ah]"); ax.set_ylabel("cell voltage [V]")
    ax.set_title("DFN (PyBaMM) discharge of a 5 Ah LG M50 cell")
    ax.legend()
    save(fig, "fig02_dfn_rates.png")


def _dfn_model(run, ckpt="final.pt"):
    import torch
    from dfn_pinn.v2.params import CellParams, Protocol, Scales
    from dfn_pinn.v2.model import DFNPINN
    ck = torch.load(run / ckpt, weights_only=False)
    cfg = ck["train"]; cell = CellParams(**ck["cell"]); prot = Protocol(**ck["protocol"])
    m = DFNPINN(Scales(cell, prot), cfg["width"], cfg["depth"], cfg["act"], cfg["projection"], ic_tau_s=cfg["ic_tau_s"],
                inventory=cfg.get("inventory", "soft"), fourier_t=cfg.get("fourier_t", 0),
                width_scalar=cfg.get("width_scalar", 0) or None, short_t=tuple(cfg.get("short_t", [])),
                collector_bc=cfg.get("collector_bc", "soft"), fourier_period=cfg.get("fourier_period", 1.0))
    m.load_state_dict(ck["model"], strict=False)
    return m.to(torch.float64), cfg


def fig_dfn_pinn():
    from dfn_pinn.v2.reference import load
    from dfn_pinn.v2.evaluate import predict_on_reference
    ref = load(ROOT / "results" / "v2_reference" / "ref_I5A_t3000s_ramp30s_x80_r120.npz")
    t = np.asarray(ref["t"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    ax[0].plot(t / 60, ref["V"], color=INK, label="PyBaMM (reference)")
    for run, lab, c in (("f4_hardbc_w96_40k_20261005T194145Z", "PINN seed 0", C1), ("f4_hardbc_w96_40k_seed1_20261006T095948Z", "PINN seed 1", C2)):
        m, cfg = _dfn_model(ROOT / "results" / "v2_runs" / run)
        p = predict_on_reference(m, ref, kinetics=cfg.get("kinetics", "inverse"))
        if c == C1:
            ax[0].plot(t / 60, p["V"], "--", color=c, label=lab)
            x = np.asarray(ref["x"]) * 1e6
            for tt, cc in ((300, C1), (1500, C2), (2900, C3)):
                i = int(np.argmin(abs(t - tt)))
                ax[2].plot(x, ref["c_e"][:, i], color=cc, label=f"t = {t[i]:.0f} s")
                ax[2].plot(x, p["c_e"][:, i], "--", color=INK, lw=1.2)
        ax[1].plot(t / 60, 1e3 * (p["V"] - ref["V"]), color=c, lw=1.4, label=lab)
    ax[0].set_xlabel("time [min]"); ax[0].set_ylabel("voltage [V]"); ax[0].legend(); ax[0].set_title("1C discharge")
    ax[1].set_xlabel("time [min]"); ax[1].set_ylabel("PINN - PyBaMM [mV]"); ax[1].legend(); ax[1].set_title("voltage error")
    ax[2].set_xlabel("x [um]"); ax[2].set_ylabel("salt concentration [mol/m$^3$]"); ax[2].set_title("electrolyte (dashed: PINN)")
    ax[2].legend(fontsize=8)
    save(fig, "fig05_dfn_pinn_vs_pybamm.png")


def fig_dfn_inverse():
    hB = json.loads((ROOT / "results/v2_runs/B_hier_w96_20261006T043317Z/history.json").read_text())["history"]
    hB2 = json.loads((ROOT / "results/v2_runs/B2_hier_w96_cont_20261006T085648Z/history.json").read_text())["history"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    off = hB[-1]["step"]
    for name, c in (("D_p", C1), ("k_n", C2), ("D_n", C3)):
        s = [h["step"] for h in hB if h.get("parameters")] + [off + h["step"] for h in hB2 if h.get("parameters")]
        true = {"D_p": 4e-15, "k_n": 6.48e-7, "D_n": 3.3e-14}[name]     # Chen2020 values used to generate the data
        v = [h["parameters"][name] / true for h in hB if h.get("parameters")] + [h["parameters"][name] / true for h in hB2 if h.get("parameters")]
        ax.plot(s, v, color=c, label=name)
    ax.axhline(1.0, color=INK, lw=1, ls=":")
    ax.axvline(off, color=INK2, lw=1, ls="--"); ax.text(off + 150, 2.2, "continuation\n(B2)", fontsize=8, color=INK2)
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("estimate / true value")
    ax.set_title("DFN inverse: three parameters from noisy 1C voltage")
    ax.legend()
    save(fig, "fig06_dfn_inverse.png")


def fig_aging():
    r = json.loads((ROOT / "results/v2_runs/C3b_aging_cellC_c200_Dstress_R0free_20261006T211155Z/aging_result.json").read_text())
    names = ["theta_n0", "theta_p0", "eps_am_n", "eps_am_p"]
    labels = ["initial lithium\nin negative\n(theta_n0)", "initial lithium\nin positive\n(theta_p0)", "active material\nnegative\n(eps_am_n)",
              "active material\npositive\n(eps_am_p)"]
    err = [100 * (r["estimates"][n] / r["truth"][n] - 1) for n in names]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(range(4), err, color=[C1, C1, C2, C2], width=0.55)
    ax.axhline(0, color=INK, lw=1)
    for i, e in enumerate(err):
        ax.text(i, e + (0.08 if e >= 0 else -0.08), f"{e:+.1f} %", ha="center", va="bottom" if e >= 0 else "top", fontsize=9, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("error of the estimate [%]")
    ax.set_title("Aged cell (200 cycles): what the PINN recovers from one discharge")
    ax.set_ylim(-2.4, 2.4)
    save(fig, "fig08_aging_errors.png")


# ---------------------------------------------------------------------------------------------- Li-SPAN data figures
def _lispan_pred(run, ref_path, ckpt="final.pt"):
    import torch
    from lispan_plot_run import load_run
    from dfn_pinn.lispan.model import load
    model, cfg, params, protocol, step, dtype = load_run(ROOT / "results" / "lispan_runs" / run, ckpt)
    ref = load(ROOT / "results" / "lispan" / ref_path)
    t = np.asarray(ref["t"]); keep = (t <= model.t_end) & (t > 60)
    T = torch.as_tensor(t[keep] / model.t_end, dtype=dtype).view(-1, 1)
    with torch.no_grad():
        V = model.voltage(T).view(-1).numpy()
    return ref, keep, V, model, params


def fig_lispan_forward():
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for run, refp, lab, c in (("lispan_f01C_w96_inv_20261007T032802Z", "pinn_reference_0.1C_rii_Zcc0.025.npz", "0.1 C", C1),
                              ("lispan_f1C_w96_sepscale_20261007T061233Z", "pinn_reference_1C_rii_Zcc0.025.npz", "1 C", C2)):
        ref, keep, V, model, params = _lispan_pred(run, refp)
        Q = np.asarray(ref["Q_mAh_gS"])[keep]
        ax[0].plot(Q, np.asarray(ref["V"])[keep], color=c, label=f"finite volume, {lab}")
        ax[0].plot(Q, V, "--", color=INK, lw=1.2)
        ax[1].plot(Q, 1e3 * (V - np.asarray(ref["V"])[keep]), color=c, lw=1.4, label=lab)
    ax[0].plot([], [], "--", color=INK, lw=1.2, label="PINN")
    ax[0].set_xlabel("specific capacity [mAh/g S]"); ax[0].set_ylabel("cell voltage [V]"); ax[0].legend(fontsize=8)
    ax[0].set_title("Li-SPAN discharge: PINN vs reference")
    ax[1].set_xlabel("specific capacity [mAh/g S]"); ax[1].set_ylabel("PINN - reference [mV]"); ax[1].legend()
    ax[1].set_title("voltage error"); ax[1].set_ylim(-2, 2)
    # species at 0.1 C
    import torch
    ref, keep, V, model, params = _lispan_pred("lispan_f01C_w96_inv_20261007T032802Z", "pinn_reference_0.1C_rii_Zcc0.025.npz")
    t = np.asarray(ref["t"])[keep]; Q = np.asarray(ref["Q_mAh_gS"])[keep]
    yc = np.asarray(ref["y_c"]); nq = len(yc)
    T = torch.as_tensor(t / model.t_end, dtype=torch.float32).view(-1, 1)
    Yq = torch.as_tensor(yc / params.L_cat, dtype=torch.float32).view(1, -1, 1).expand(len(t), nq, 1).reshape(-1, 1)
    Tq = T.view(-1, 1, 1).expand(len(t), nq, 1).reshape(-1, 1)
    with torch.no_grad():
        sp = model.species(Yq, Tq)
    for key, lab, c in (("c_S4", "PAN-S$_4$", C1), ("c_S3", "PAN-S$_3$Li", C2), ("c_S2", "PAN-S$_2$Li", C3), ("c_S1", "PAN-SLi", C7)):
        ax[2].plot(Q, np.asarray(ref[key])[:, keep].mean(0), color=c, label=lab)
        ax[2].plot(Q, sp[key].view(len(t), nq).numpy().mean(1), "--", color=INK, lw=1.0)
    ax[2].set_xlabel("specific capacity [mAh/g S]"); ax[2].set_ylabel("mean concentration [mol/m$^3$]")
    ax[2].set_title("sulfur species at 0.1 C (dashed: PINN)"); ax[2].legend(fontsize=8)
    save(fig, "fig11_lispan_pinn_forward.png")


def fig_inventory():
    import torch
    from lispan_plot_run import load_run
    from dfn_pinn.lispan.model import load
    from dfn_pinn.lispan.params import F
    ref = load(ROOT / "results/lispan/pinn_reference_0.1C_rii_Zcc0.025.npz")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for run, lab, c in (("lispan_f01C_w96_20261006T194300Z", "run B (no inventory term)", C2),
                        ("lispan_f01C_w96_inv_20261007T032802Z", "run C (with inventory term)", C1)):
        model, cfg, p, prot, step, dt = load_run(ROOT / "results/lispan_runs" / run, "final.pt")
        t = np.asarray(ref["t"]); keep = (t <= model.t_end) & (t > 100); t = t[keep]
        T = torch.as_tensor(t / model.t_end, dtype=dt).view(-1, 1)
        q, w = model.q_nodes.to(dt), model.q_weights.to(dt); nt, nq = len(t), len(q)
        Yq = q.view(1, nq).expand(nt, nq).reshape(-1, 1); Tq = T.view(nt, 1).expand(nt, nq).reshape(-1, 1)
        with torch.no_grad():
            xi = model.extents(Yq, Tq)
            inv = (((xi[0] + xi[1] + xi[2]).view(nt, nq)) * w.view(1, nq)).sum(1).numpy() * p.c_init[0]
            V = model.voltage(T).view(-1).numpy()
        Qc = prot.current * (t + prot.ramp_s * (np.log1p(np.exp(-2 * t / prot.ramp_s)) - np.log(2)))
        ax[0].plot(t / 3600, inv - Qc / (2 * F * p.L_cat), color=c, label=lab)
        ax[1].plot(t / 3600, 1e3 * (V - np.asarray(ref["V"])[keep]), color=c, lw=1.4, label=lab)
    ax[0].axhline(0, color=INK, lw=1)
    ax[0].set_xlabel("time [h]"); ax[0].set_ylabel("electron bookkeeping error [mol/m$^3$]")
    ax[0].set_title("do the reactions add up to the charge passed?"); ax[0].legend(fontsize=8)
    ax[1].axhline(0, color=INK, lw=1)
    ax[1].set_xlabel("time [h]"); ax[1].set_ylabel("voltage error [mV]"); ax[1].set_title("effect on the voltage")
    ax[1].legend(fontsize=8)
    save(fig, "fig12_inventory_fix.png")


def fig_identifiability():
    d1 = np.load(ROOT / "results/lispan/identifiability/sensitivities_0.1C.npz")
    d2 = np.load(ROOT / "results/lispan/identifiability/sensitivities_1C.npz")
    names = list(d1["names"])
    s1 = np.sqrt((d1["S"] ** 2).mean(1)) * 1e3; s2 = np.sqrt((d2["S"] ** 2).mean(1)) * 1e3
    # U0 per 10 mV instead of per volt, so all bars mean "mV of voltage change for a realistic change of the parameter"
    scale = np.array([0.01 if n.startswith("U0") else 0.1 for n in names])   # 10 mV, or 10 % (log 0.1)
    s1, s2 = s1 * scale, s2 * scale
    order = np.argsort(-np.maximum(s1, s2))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(names))
    ax.bar(x - 0.18, s1[order], width=0.34, color=C1, label="0.1 C")
    ax.bar(x + 0.18, s2[order], width=0.34, color=C2, label="1 C")
    ax.axhline(1.0, color=INK, lw=1, ls="--"); ax.text(len(names) - 0.6, 1.15, "1 mV (measurement noise)", ha="right", fontsize=8, color=INK2)
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("rms voltage change [mV]")
    ax.set_title("How much does the voltage move when a parameter changes by 10 % (U0: by 10 mV)?")
    ax.legend()
    save(fig, "fig13_sensitivities.png")


def fig_experiment():
    from dfn_pinn.lispan.params import LiSPANParams, LiSPANProtocol
    from dfn_pinn.lispan.model import solve
    from lispan_make_inverse_data import apply_truth
    base = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)
    fit = json.loads((ROOT / "results/lispan/fit_experiment/fit_0.1+1C_from_pinn.json").read_text())["estimates"]
    tr = json.loads(next((ROOT / "results/lispan_runs").glob("lispan_inv8p_experiment_tr_*/inverse_result.json")).read_text())["estimates"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, (cr, c) in enumerate(((0.05, C1), (0.1, C2), (0.2, C3), (1.0, C7))):
        e = np.loadtxt(ROOT / f"results/lispan/paper_digitized/fig4b_exp_{cr:g}C.csv", delimiter=",", skiprows=1)
        a = ax[0] if cr in (0.1, 1.0) else ax[1]
        a.plot(e[:, 0], e[:, 1], "o", mfc="white", mec=c, ms=6, mew=1.6, label=f"measured {cr:g} C")
        for prm, st, lab in ((base, ":", "nominal model"), (apply_truth(base, fit), "-", "least-squares fit"),
                             (apply_truth(base, tr), "--", "PINN (trust region)")):
            _, res = solve(prm, LiSPANProtocol.from_crate(cr, ramp_s=30.0), N_c=20, N_s=10, n_out=600)
            a.plot(res["Q_mAh_gS"], res["V"], st, color=c, lw=1.6 if st != "-" else 2.0, label=lab if cr in (0.1, 0.05) else None)
    ax[0].set_title("used for fitting: 0.1 C and 1 C"); ax[1].set_title("not used (prediction): 0.05 C and 0.2 C")
    for a in ax:
        a.set_xlabel("specific capacity [mAh/g S]"); a.set_ylabel("cell voltage [V]"); a.legend(fontsize=7.5); a.set_ylim(0.95, 2.4)
    save(fig, "fig17_experiment_fit.png")


def fig_copy_existing():
    import shutil
    for src, dst in (("results/lispan/ref_Zcc0.025/discharge_vs_paper.png", "fig10_lispan_reference_vs_paper.png"),
                     ("results/lispan_runs/lispan_inv01C_k0_20261007T044408Z/inverse_summary.png", "fig15_inverse_k0.png"),
                     ("results/lispan/inverse_8p_multirate_adam_vs_lm.png", "fig16_multirate_adam_vs_lm.png"),
                     ("results/lispan/span500_vs_lispan_model.png", "fig18_span500.png")):
        shutil.copy(ROOT / src, OUT / dst); print("copied", dst)


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    funcs = {k[4:]: v for k, v in globals().items() if k.startswith("fig_")}
    for name, f in funcs.items():
        if "all" in which or name in which:
            f()
