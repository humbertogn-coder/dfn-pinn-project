"""Paper build (step 3, traceability): regenerate every figure, table and quoted number of the Li-SPAN paper from
saved artifacts, and record exactly which files they came from.

    python scripts/paper_build.py                 # figures + tables + numbers from existing results (a few minutes)
    python scripts/paper_build.py --recompute     # first rerun the cheap finite-volume scripts (~30 min CPU)

Outputs (paper/):
    figures/fig1_reference_vs_paper.png   FV reference vs digitized Simanjuntak Figs. 6a and 4b
    figures/fig2_forward_voltage.png      PINN vs FV voltage and error, all seeds, 0.1 C and 1 C
    figures/fig3_forward_species.png      cathode-averaged SPAN species and Li2S, PINN vs FV
    figures/fig4_identifiability.png      Cramer-Rao bounds of the eight inverse parameters, 0.1 C / 1 C / both
    figures/fig5_inverse_synthetic.png    multi-rate inverse: Adam vs Levenberg-Marquardt, final errors vs ideal estimator
    figures/fig6_measured.png             Simanjuntak Fig. 4b circles and SPAN500 C/10 (Nat. Mater. 2026) calibrations
    figures/figS1_assumption_checks.png   benchmark choices (copy of results/paper/assumption_checks.png)
    tables/T1_assumption_checks.md, T2_forward_accuracy.md, T3_inverse_synthetic.md, T4_cost.md, T5_measured.md
    numbers.json      every number quoted in the text: value, unit, source file
    MANIFEST.json     sha256 of every input file and of the code that produced them, package versions

Nothing here trains a network: PINN numbers are recomputed from the saved checkpoints (final.pt) against the
finite-volume references, which also checks that the checkpoints reproduce their logged metrics.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dfn_pinn.lispan.params import LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.model import solve, load  # noqa: E402
from dfn_pinn.lispan.pinn import evaluate  # noqa: E402
from dfn_pinn.lispan import runs  # noqa: E402

OUT = ROOT / "paper"
FIG, TAB = OUT / "figures", OUT / "tables"
RES = ROOT / "results"
RUNS = RES / "lispan_runs"
DIG = RES / "lispan" / "paper_digitized"
BENCH = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)
INV8 = ["k0_1", "k0_2", "k0_3", "b_1", "b_2", "b_3", "U0_1", "Z_CC"]
RUN_IDS = {
    "inverse_adam": "lispan_inv8p_multirate_freeze_20261007T073258Z",
    "inverse_lm": "lispan_inv8p_multirate_lm_20261007T083347Z",
    "inverse_stage1": ["lispan_inv8p_stage1_0.1C_20261007T060146Z", "lispan_inv8p_stage1_1C_20261007T070125Z"],
    "experiment_pinn": "lispan_inv8p_experiment_tr_20261007T112339Z",
}
NUMBERS, INPUTS = {}, set()


def use(path):
    path = Path(path)
    INPUTS.add(path.resolve())
    return path


def num(key, value, unit, source):
    NUMBERS[key] = {"value": value if not isinstance(value, (np.floating, np.integer)) else value.item(),
                    "unit": unit, "source": str(source)}


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def dig(name):
    d = np.loadtxt(use(DIG / name), delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1]


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200})
    return plt


# ---------------------------------------------------------------------------------------------------------- figure 1
def fig_reference(plt):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for cr, col in ((0.1, "C0"), (1.0, "C3")):
        f = use(RES / "lispan" / "ref_Zcc0" / f"discharge_{cr:g}C.npz")
        r = load(f)
        qd, vd = dig(f"fig6a_k0_1e-2_{cr:g}C.csv")
        m = (qd >= r["Q_mAh_gS"][0]) & (qd <= r["Q_mAh_gS"][-1])
        e = 1e3 * (np.interp(qd[m], r["Q_mAh_gS"], r["V"]) - vd[m])
        num(f"fig6a_rms_mV_{cr:g}C", float(np.sqrt(np.mean(e ** 2))), "mV", rel(f))
        num(f"fig6a_max_mV_{cr:g}C", float(np.abs(e).max()), "mV", rel(f))
        axs[0].plot(r["Q_mAh_gS"], r["V"], color=col, lw=1.5, label=f"this work, {cr:g} C")
        axs[0].plot(qd, vd, "o", ms=2.5, mfc="none", color=col, label=f"Simanjuntak Fig. 6a, {cr:g} C")
    axs[0].set_title("(a) Z_CC = 0, Table 3 OCV, effective k0")
    for cr, col in zip((0.05, 0.1, 0.2, 1.0), ("C2", "C0", "C1", "C3")):
        f = use(RES / "lispan" / "ref_Zcc0.025" / f"discharge_{cr:g}C.npz")
        r = load(f)
        axs[1].plot(r["Q_mAh_gS"], r["V"], color=col, lw=1.5, label=f"this work, {cr:g} C")
        qe, ve = dig(f"fig4b_exp_{cr:g}C.csv")
        axs[1].plot(qe, ve, "o", ms=3, mfc="none", color=col)
        num(f"fig4b_capacity_model_mAh_gS_{cr:g}C", float(r["Q_mAh_gS"][-1]), "mAh/g_S", rel(f))
    axs[1].plot([], [], "ko", mfc="none", ms=3, label="experiment (Fig. 4b)")
    axs[1].set_title("(b) Z_CC = 0.025 Ohm m2, rates of Fig. 4b")
    for ax in axs:
        ax.set_xlabel("capacity [mAh / g S]"); ax.set_ylabel("cell voltage [V]"); ax.set_ylim(0.95, 2.4)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig1_reference_vs_paper.png"); plt.close(fig)


# ------------------------------------------------------------------------------------------------- figures 2-3, T2
def forward_runs(override=None):
    found = {}
    for cr in ("0.1", "1"):
        if override and override.get(cr):
            found[cr] = [RUNS / r for r in override[cr]]
            continue
        found[cr] = [d for d in sorted(RUNS.glob(f"lispan_final_{cr}C_seed*_*"))
                     if (d / "final.pt").exists() and not runs.is_stopped(d)]
    return found


def fig_forward(plt, override=None):
    """With an override (stand-in runs, for testing) nothing goes to numbers.json and the table gets a _TEST suffix."""
    found = forward_runs(override)
    record = num if not override else (lambda *a, **k: None)
    rows = []
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.2), sharex="col", gridspec_kw={"height_ratios": [1.3, 1]})
    figs, axs2 = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, cr in enumerate(("0.1", "1")):
        if not found[cr]:
            continue
        for k, d in enumerate(found[cr]):
            cfg = json.loads(use(d / "config.json").read_text())
            p = runs._params(cfg["params"]); prot = runs._protocol(cfg["protocol"])
            rev = "".join("r" if r else "i" for r in p.reversible)
            ref_path = use(RES / "lispan" / f"pinn_reference_{prot.crate:g}C_{rev}_Zcc{p.Z_CC:g}.npz")
            ref = load(ref_path)
            model = runs.load_model(use(d / "final.pt"))
            ev = evaluate(model, ref, torch.float32)
            logged = json.loads(use(d / "final_metrics.json").read_text())
            use(d / "history.json"); h = runs.history(d)
            rows.append({"crate": float(cr), "seed": cfg["train"]["seed"], "run": d.name, **ev,
                         "logged_V_rmse_mV": logged.get("V_rmse_mV"), "train_wall_h": h["history"][-1]["time_s"] / 3600})
            t = np.asarray(ref["t"]); keep = t <= model.t_end
            T = torch.as_tensor(t[keep] / model.t_end, dtype=torch.float32).view(-1, 1)
            with torch.no_grad():
                V = model.voltage(T).view(-1).numpy()
            Q = np.asarray(ref["Q_mAh_gS"])[keep]
            if k == 0:
                axs[0, j].plot(ref["Q_mAh_gS"], ref["V"], "k", lw=2.2, alpha=0.35, label="finite volume")
                axs[0, j].plot(Q, V, "C3--", lw=1.0, label=f"PINN, seed {cfg['train']['seed']}")
                # species (cathode averages) for seed 0
                yc = np.asarray(ref["y_c"]); nt, ny = len(T), len(yc)
                Yq = torch.as_tensor(yc / p.L_cat, dtype=torch.float32).view(1, ny, 1).expand(nt, ny, 1).reshape(-1, 1)
                Tq = T.view(nt, 1, 1).expand(nt, ny, 1).reshape(-1, 1)
                with torch.no_grad():
                    sp = model.species(Yq, Tq)
                for i, key in enumerate(("c_S4", "c_S3", "c_S2", "c_S1")):
                    axs2[j].plot(ref["Q_mAh_gS"], np.asarray(ref[key]).mean(axis=0), color=f"C{i}", lw=2.2, alpha=0.35)
                    axs2[j].plot(Q, sp[key].view(nt, ny).numpy().mean(axis=1), "--", color=f"C{i}", lw=1.0,
                                 label=key.replace("c_", ""))
                ax2b = axs2[j].twinx()
                ax2b.plot(ref["Q_mAh_gS"], np.asarray(ref["eps_L"]).mean(axis=0), color="0.4", lw=2.2, alpha=0.35)
                ax2b.plot(Q, sp["eps_L"].view(nt, ny).numpy().mean(axis=1), "--", color="0.4", lw=1.0)
                ax2b.set_ylabel("Li2S volume fraction (grey)"); ax2b.grid(False)
            axs[1, j].plot(Q, 1e3 * (V - np.asarray(ref["V"])[keep]), lw=0.9, label=f"seed {cfg['train']['seed']}")
            if k == 0:   # metrics use t >= 100 s; before that the ramp starts and, at t -> 0 (no current, intermediate
                q100 = float(np.interp(100.0, t, ref["Q_mAh_gS"]))     # species -> 0), the Delta phi root sits at its bracket
                axs[1, j].axvspan(Q[0], q100, color="0.85", lw=0, zorder=0)
        axs[0, j].set_title(f"{cr} C"); axs[0, j].set_ylabel("cell voltage [V]"); axs[0, j].legend(fontsize=7)
        axs[0, j].set_ylim(0.95, 2.8)
        axs[1, j].set_ylabel("PINN - FV [mV]"); axs[1, j].set_xlabel("capacity [mAh / g S]"); axs[1, j].legend(fontsize=7)
        axs[1, j].set_ylim(-3, 3)
        axs2[j].set_title(f"{cr} C: cathode averages (solid FV, dashed PINN seed 0)")
        axs2[j].set_xlabel("capacity [mAh / g S]"); axs2[j].set_ylabel("concentration [mol / m3]")
        axs2[j].legend(fontsize=7, loc="center left")
    tag = "_TEST" if override else ""
    fig.tight_layout(); fig.savefig(FIG / f"fig2_forward_voltage{tag}.png"); plt.close(fig)
    figs.tight_layout(); figs.savefig(FIG / f"fig3_forward_species{tag}.png"); plt.close(figs)
    # table + numbers
    L = ["| rate | seed | V rms [mV] | V max [mV] | c_S rms (S4/S3/S2/S1) [mol/m3] | c_e rms / max [mol/m3] | "
         "training [h] | run |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['crate']:g} C | {r['seed']} | {r['V_rmse_mV']:.3f} | {r['V_max_mV']:.3f} | "
                 f"{r['c_S4_rmse']:.2f} / {r['c_S3_rmse']:.2f} / {r['c_S2_rmse']:.2f} / {r['c_S1_rmse']:.2f} | "
                 f"{r['c_e_rmse']:.2f} / {r['c_e_max']:.2f} | {r['train_wall_h']:.2f} | {r['run']} |")
        record(f"forward_V_rmse_mV_{r['crate']:g}C_seed{r['seed']}", r["V_rmse_mV"], "mV", r["run"] + "/final.pt")
        record(f"forward_V_max_mV_{r['crate']:g}C_seed{r['seed']}", r["V_max_mV"], "mV", r["run"] + "/final.pt")
        if r["logged_V_rmse_mV"] is not None and abs(r["logged_V_rmse_mV"] - r["V_rmse_mV"]) > 1e-3:
            print(f"WARNING {r['run']}: recomputed V rms {r['V_rmse_mV']:.4f} vs logged {r['logged_V_rmse_mV']:.4f} mV")
    for cr in (0.1, 1.0):
        v = [r["V_rmse_mV"] for r in rows if r["crate"] == cr]
        if v:
            record(f"forward_V_rmse_mV_{cr:g}C_mean", float(np.mean(v)), "mV", f"{len(v)} seeds")
            record(f"forward_V_rmse_mV_{cr:g}C_spread", float(np.max(v) - np.min(v)), "mV", f"{len(v)} seeds")
            L.append(f"| {cr:g} C | mean of {len(v)} | {np.mean(v):.3f} (range {np.min(v):.3f}-{np.max(v):.3f}) | | | | | |")
    (TAB / ("T2_forward_accuracy_TEST.md" if override else "T2_forward_accuracy.md")).write_text("\n".join(L) + "\n")
    return rows


# ---------------------------------------------------------------------------------------------------------- figure 4
def fig_identifiability(plt):
    from lispan_identifiability import fisher_report
    sens = {}
    for cr in (0.1, 1.0):
        d = np.load(use(RES / "lispan" / "identifiability" / f"sensitivities_{cr:g}C.npz"))
        names = [str(n) for n in d["names"]]
        idx = [names.index(n) for n in INV8]
        sens[cr] = d["S"][idx]
    units = {n: ("volt" if n.startswith("U0") else "rel") for n in INV8}
    sets = {"0.1 C": [0.1], "1 C": [1.0], "0.1 C + 1 C": [0.1, 1.0]}
    rep = {k: fisher_report(np.concatenate([sens[r] for r in v], axis=1), INV8, 1e-3, units) for k, v in sets.items()}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(INV8)); w = 0.27
    for i, (k, r) in enumerate(rep.items()):
        vals = [min(r["crlb"][n], 1e4) for n in INV8]
        ax.bar(x + (i - 1) * w, vals, w, label=k)
        for n in INV8:
            num(f"crlb_{k.replace(' ', '')}_{n}", r["crlb"][n], "mV" if n.startswith("U0") else "%",
                "results/lispan/identifiability/sensitivities_*.npz (100 points per curve, 1 mV noise)")
    # transport / Li2S parameters, from the full study (results/lispan/identifiability/identifiability.json, reduced set
    # without the confounded U0_2, U0_3)
    ij = RES / "lispan" / "identifiability" / "identifiability.json"
    idf = json.loads(use(ij).read_text())
    red, names = idf["reduced"]["0.05C+0.1C+0.2C+1C"], idf["reduced_params"]
    for n in ("kappa0", "t_plus", "kappa_SPAN", "D_salt"):
        num(f"crlb_4rates_{n}", red["crlb"][n], "%", rel(ij) + " (reduced, 0.05+0.1+0.2+1 C)")
    cmat = np.array(red["corr"])
    num("corr_4rates_kappa0_Z_CC", float(cmat[names.index("kappa0"), names.index("Z_CC")]), "-", rel(ij))
    num("sens_max_Li2S_mV_per_efold", float(max(v[n] for v in idf["rms_sensitivity_mV"].values() for n in ("K_sp", "k0_L", "D_S"))),
        "mV per e-fold", rel(ij))
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([n + (" [mV]" if n.startswith("U0") else " [%]") for n in INV8])
    ax.set_ylabel("Cramer-Rao bound (1 mV noise)"); ax.axhline(1.0, color="k", lw=0.6); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig4_identifiability.png"); plt.close(fig)


# ------------------------------------------------------------------------------------------------------ figure 5, T3
def fig_inverse(plt):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, key, title in ((axs[0], "inverse_adam", "(a) Adam on all unknowns"),
                           (axs[1], "inverse_lm", "(b) Adam + Levenberg-Marquardt on parameters")):
        d = RUNS / RUN_IDS[key]
        use(d / "history.json"); hist = runs.history(d)["history"]
        res = json.loads(use(d / "inverse_result.json").read_text())
        steps = np.array([0] + [h["step"] for h in hist])
        for i, n in enumerate(INV8):
            v = np.array([res["initial"][n]] + [h["params"][n] for h in hist])
            err = 1e3 * v if n.startswith("U0") else 100.0 * (v - 1.0)
            ax.plot(steps, err, lw=1, color=f"C{i}", label=n + (" [mV]" if n.startswith("U0") else " [%]"))
        ax.set_yscale("symlog", linthresh=1.0); ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("training step"); ax.set_ylabel("error vs truth"); ax.set_title(title, fontsize=9)
    axs[0].legend(fontsize=6, ncol=2)
    lm = json.loads(use(RUNS / RUN_IDS["inverse_lm"] / "inverse_result.json").read_text())
    adam = json.loads(use(RUNS / RUN_IDS["inverse_adam"] / "inverse_result.json").read_text())
    fv_path = RES / "lispan" / "fit_experiment" / "fit_0.1+1C_nominal.json"
    fv = json.loads(use(fv_path).read_text()) if fv_path.exists() else None
    x = np.arange(len(INV8)); w = 0.38
    axs[2].bar(x - w / 2, [lm["errors_pct_or_mV"][n] for n in INV8], w, label="PINN (LM)")
    if fv:
        axs[2].bar(x + w / 2, [fv["errors_pct_or_mV"][n] for n in INV8], w, label="ideal estimator (FV least squares)")
        crlb = [fv["crlb_pct_or_mV"][n] for n in INV8]
        axs[2].errorbar(x + w / 2, [fv["errors_pct_or_mV"][n] for n in INV8], yerr=crlb, fmt="none", ecolor="k",
                        lw=0.8, capsize=2, label="+- Cramer-Rao bound")
    emp0 = RES / "paper" / "inverse_error_model.json"
    if fv and emp0.exists():                   # ideal estimator + bias predicted from the PINN's forward error
        em0 = json.loads(use(emp0).read_text())
        axs[2].plot(x - w / 2, [fv["errors_pct_or_mV"][n] + em0["predicted_bias_pct_or_mV"][n] for n in INV8], "kx",
                    ms=6, label="ideal + bias predicted from the PINN forward error")
    axs[2].axhline(0, color="k", lw=0.5); axs[2].set_xticks(x)
    axs[2].set_xticklabels([n + ("\n[mV]" if n.startswith("U0") else "\n[%]") for n in INV8], fontsize=7)
    axs[2].set_title("(c) final errors, same data and noise", fontsize=9); axs[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig5_inverse_synthetic.png"); plt.close(fig)
    emp = RES / "paper" / "inverse_error_model.json"
    em = json.loads(use(emp).read_text()) if emp.exists() else None
    L = ["| parameter | initial guess | Adam only (7000 steps) | PINN + LM (6000 steps) | ideal estimator | CRLB | "
         "(PINN - ideal) / CRLB | PINN - ideal: actual / predicted from the PINN forward error |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    dev = []
    for n in INV8:
        u = "mV" if n.startswith("U0") else "%"
        ini = 1e3 * lm["initial"][n] if n.startswith("U0") else 100 * (lm["initial"][n] - 1)
        L.append(f"| {n} [{u}] | {ini:+.1f} | {adam['errors_pct_or_mV'][n]:+.2f} | {lm['errors_pct_or_mV'][n]:+.2f} | "
                 + (f"{fv['errors_pct_or_mV'][n]:+.2f} | {fv['crlb_pct_or_mV'][n]:.2f} | "
                    f"{(lm['errors_pct_or_mV'][n] - fv['errors_pct_or_mV'][n]) / fv['crlb_pct_or_mV'][n]:+.1f} |"
                    if fv else "- | - | - |")
                 + (f" {em['actual_pinn_minus_ideal_pct_or_mV'][n]:+.2f} / {em['predicted_bias_pct_or_mV'][n]:+.2f} |"
                    if em else " - |"))
        if em:
            num(f"inverse_error_model_predicted_{n}", em["predicted_bias_pct_or_mV"][n], u, rel(emp))
            num(f"inverse_error_model_actual_{n}", em["actual_pinn_minus_ideal_pct_or_mV"][n], u, rel(emp))
        if fv:
            dev.append(abs(lm["errors_pct_or_mV"][n] - fv["errors_pct_or_mV"][n]) / fv["crlb_pct_or_mV"][n])
        num(f"inverse_lm_error_{n}", lm["errors_pct_or_mV"][n], u, RUN_IDS["inverse_lm"] + "/inverse_result.json")
        num(f"inverse_adam_error_{n}", adam["errors_pct_or_mV"][n], u, RUN_IDS["inverse_adam"] + "/inverse_result.json")
        if fv:
            num(f"inverse_ideal_error_{n}", fv["errors_pct_or_mV"][n], u, rel(fv_path))
            num(f"inverse_crlb_{n}", fv["crlb_pct_or_mV"][n], u, rel(fv_path))
    if dev:
        num("inverse_lm_vs_ideal_max_crlb", float(max(dev)), "CRLB units", RUN_IDS["inverse_lm"] + " vs " + rel(fv_path))
        num("inverse_lm_vs_ideal_median_crlb", float(np.median(dev)), "CRLB units", RUN_IDS["inverse_lm"] + " vs " + rel(fv_path))
        num("inverse_fv_solves_per_rate", fv["n_solves_per_rate"], "-", rel(fv_path))
    if em:
        for r, v in em["forward_error_rms_mV"].items():
            num(f"inverse_pinn_forward_error_rms_mV_{r}", v, "mV", rel(emp))
    (TAB / "T3_inverse_synthetic.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------------ figure 6, T5
def fv_voltage(est, cr, base=BENCH, grid=(20, 10)):
    from lispan_make_inverse_data import apply_truth
    p = apply_truth(base, est)
    _, r = solve(p, LiSPANProtocol.from_crate(cr, ramp_s=30.0), N_c=grid[0], N_s=grid[1], n_out=1500)
    return r


def fig_measured(plt):
    import lispan_span500 as sp5
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    fvj = RES / "lispan" / "fit_experiment" / "fit_0.1+1C_fdabs.json"     # absolute FD steps (fdjac), 2026-10-08
    if not fvj.exists():
        fvj = RES / "lispan" / "fit_experiment" / "fit_0.1+1C.json"
    fv = json.loads(use(fvj).read_text())
    pinn = json.loads(use(RUNS / RUN_IDS["experiment_pinn"] / "inverse_result.json").read_text())
    L = ["| data | model | rms misfit [mV] | notes |", "| --- | --- | --- | --- |"]
    for cr, col in zip((0.05, 0.1, 0.2, 1.0), ("C2", "C0", "C1", "C3")):
        d = np.load(use(RES / "lispan" / "inverse_data" / f"V_{cr:g}C_experiment.npz"))
        role = "fit" if cr in (0.1, 1.0) else "prediction"
        axs[0].plot(d["Q_mAh_gS"], d["V"], "o", ms=3.5, mfc="none", color=col, label=f"exp. {cr:g} C ({role})")
        for lab, est, ls in (("FV least squares", fv["estimates"], "-"), ("PINN inverse", pinn["estimates"], "--")):
            r = fv_voltage(est, cr)
            Vm = np.where(d["t"] <= r["t"][-1], np.interp(d["t"], r["t"], r["V"]), 1.0)
            rms = float(1e3 * np.sqrt(np.mean((Vm - d["V"]) ** 2)))
            num(f"measured_fig4b_{lab.split()[0]}_rms_mV_{cr:g}C", rms, "mV", rel(fvj) if ls == "-" else RUN_IDS["experiment_pinn"])
            L.append(f"| Simanjuntak Fig. 4b, {cr:g} C ({role}) | {lab} | {rms:.1f} | |")
            if cr in (0.1, 1.0) or ls == "-":
                axs[0].plot(r["Q_mAh_gS"], r["V"], ls, color=col, lw=1.0 if ls == "--" else 1.4)
    axs[0].plot([], [], "k-", label="FV least squares"); axs[0].plot([], [], "k--", label="PINN inverse (same 8 parameters)")
    axs[0].set_title("(a) Simanjuntak et al. cell, fit at 0.1 + 1 C", fontsize=9)
    axs[0].set_xlabel("capacity [mAh / g S]")
    num("measured_fig4b_Z_CC_fit_Ohm_m2", 0.025 * fv["estimates"]["Z_CC"], "Ohm m2", rel(fvj))
    num("measured_fig4b_Z_CC_pinn_Ohm_m2", 0.025 * pinn["estimates"]["Z_CC"], "Ohm m2", RUN_IDS["experiment_pinn"])
    # SPAN500
    sj = RES / "lispan" / "span500" / "fit_cycle3.json"
    if sj.exists():
        rep = json.loads(use(sj).read_text())
        use(sp5.XLSX)
        q3, v3 = sp5.load_discharge(rep["data"]["cycle"])
        axs[1].plot(q3, v3, "k", lw=2.5, alpha=0.3, label=f"SPAN500 cycle {rep['data']['cycle']}, C/10")
        q2, v2 = sp5.load_discharge(rep["data"]["check_cycle"])
        axs[1].plot(q2, v2, color="0.45", lw=0.9, ls=(0, (1, 1)),
                    label=f"cycle {rep['data']['check_cycle']} (same cell, {rep['data']['cycle_to_cycle_rms_mV']:.0f} mV rms apart)")
        num("span500_cycle_to_cycle_rms_mV", rep["data"]["cycle_to_cycle_rms_mV"], "mV", rel(sj))
        pred1c = {}
        for hyp, col in (("paper", "C0"), ("fast", "C3")):
            if hyp not in rep["fits"]:
                continue
            f = rep["fits"][hyp]
            base = sp5.base_params(f["k0_factor"])
            Q, V, _, _ = sp5.simulate(f["estimates"], base)
            Q1, V1, _, _ = sp5.simulate(f["estimates"], base, crate_span=1.0)
            axs[1].plot(Q, V, color=col, lw=1.2, label=f"k0 x{f['k0_factor']:g}: C/10 fit {f['rms_mV']:.1f} mV rms")
            axs[1].plot(Q1, V1, ":", color=col, lw=1.4, label=f"k0 x{f['k0_factor']:g}: 1 C prediction")
            pred1c[hyp] = (Q1, V1)
            L.append(f"| SPAN500 cycle {rep['data']['cycle']} C/10 | FV, kinetics k0 x{f['k0_factor']:g} | {f['rms_mV']:.1f} | "
                     f"w_S {f['estimates']['w_S']:.3f}, U0 {', '.join(f'{u:.3f}' for u in f['U0_fitted_V'])} V, "
                     f"b {', '.join(f'{b:.3f}' for b in f['b_fitted_V'])} V; cycle {rep['data']['check_cycle']} "
                     f"{f['check_cycle_rms_mV']:.1f} mV |")
            num(f"span500_{hyp}_rms_mV", f["rms_mV"], "mV", rel(sj))
            num(f"span500_{hyp}_w_S", f["estimates"]["w_S"], "-", rel(sj))
            num(f"span500_{hyp}_check_cycle_rms_mV", f["check_cycle_rms_mV"], "mV", rel(sj))
            num(f"span500_{hyp}_Q_1C_mAh_gSPAN", f["prediction_1C"]["Q_end_mAh_gSPAN"], "mAh/g_SPAN", rel(sj))
        # PINN inverse on the same data at the FV-fitted w_S (newest lispan_inv_span500_c3_paper_wS_* run)
        pruns = sorted(d for d in RUNS.glob("lispan_inv_span500_c3_paper_wS_b5_*") if (d / "inverse_result.json").exists() and not runs.is_stopped(d))
        if pruns and "paper" in rep["fits"]:
            pr = json.loads(use(pruns[-1] / "inverse_result.json").read_text())
            vals = {"w_S": rep["fits"]["paper"]["estimates"]["w_S"], **pr["estimates"]}
            Qp, Vp, _, _ = sp5.simulate(vals, sp5.base_params(1.0))
            spec = json.loads(use(pruns[-1] / "spec.json").read_text())
            dwin = np.load(use(ROOT / spec["rates"][0]["data_path"]))     # the PINN's data window
            qd, vd = np.asarray(dwin["Q_mAh_gSPAN"]), np.asarray(dwin["V"])
            rp = 1e3 * (np.where(qd <= Qp[-1], np.interp(qd, Qp, Vp), 1.0) - vd)
            num("span500_pinn_fv_rms_mV", float(np.sqrt(np.mean(rp ** 2))), "mV", pruns[-1].name)
            Qf, Vf, _, _ = sp5.simulate(rep["fits"]["paper"]["estimates"], sp5.base_params(1.0))
            rf = 1e3 * (np.where(qd <= Qf[-1], np.interp(qd, Qf, Vf), 1.0) - vd)
            num("span500_fv_rms_mV_pinn_window", float(np.sqrt(np.mean(rf ** 2))), "mV", rel(sj) + " on the PINN window")
            num("span500_pinn_window_q_max", float(qd[-1]), "mAh/g_SPAN", spec["rates"][0]["data_path"])
            wfit = RES / "lispan" / "span500" / "fit_cycle3_pinnwindow.json"      # FV least squares on the same window
            if wfit.exists():
                wf = json.loads(use(wfit).read_text())
                num("span500_fv_window_fit_rms_mV", wf["rms_mV"], "mV", rel(wfit))
                rf = np.array([wf["rms_mV"]])
            for k, v in pr["estimates"].items():
                num(f"span500_pinn_{k}", v, "V" if k.startswith("U0") else "-", pruns[-1].name)
            axs[1].plot(Qp, Vp, "--", color="C0", lw=1.0, label=f"PINN inverse (k0 x1), FV at its parameters: "
                                                                 f"{np.sqrt(np.mean(rp ** 2)):.1f} mV rms")
            u0s = ", ".join("%+.3f" % pr["estimates"]["U0_%d" % m] for m in (1, 2, 3))
            bs = ", ".join("%.3f" % pr["estimates"]["b_%d" % m] for m in (1, 2, 3))
            L.append(f"| SPAN500 cycle {rep['data']['cycle']} C/10, Q <= {qd[-1]:.0f} mAh/g | PINN inverse (k0 x1, w_S from FV), "
                     f"FV at its parameters | {np.sqrt(np.mean(rp ** 2)):.1f} (FV least squares on the same window, same w_S: "
                     f"{np.sqrt(np.mean(rf ** 2)):.1f}) | U0 offsets {u0s} V, b x {bs} ({pruns[-1].name}) |")
        if len(pred1c) == 2:
            (Qa, Va), (Qb, Vb) = pred1c["paper"], pred1c["fast"]
            qq = np.linspace(5.0, 0.98 * min(Qa[-1], Qb[-1]), 400)
            num("span500_1C_prediction_max_diff_mV", float(1e3 * np.abs(np.interp(qq, Qa, Va) - np.interp(qq, Qb, Vb)).max()),
                "mV", rel(sj))
        axs[1].set_title("(b) SPAN500 (Wang et al. 2026): one rate, two kinetic hypotheses", fontsize=9)
        axs[1].set_xlabel("capacity [mAh / g SPAN]")
    for ax in axs:
        ax.set_ylabel("cell voltage [V]"); ax.set_ylim(0.95, 2.45); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG / "fig6_measured.png"); plt.close(fig)
    (TAB / "T5_measured.md").write_text("\n".join(L) + "\n")


# -------------------------------------------------------------------------------------------------- T1, T4, S1
def copy_tables():
    a = RES / "paper" / "assumption_checks.json"
    if a.exists():
        shutil.copy(use(RES / "paper" / "assumption_checks.md"), TAB / "T1_assumption_checks.md")
        shutil.copy(use(RES / "paper" / "assumption_checks.png"), FIG / "figS1_assumption_checks.png")
        for name, r in json.loads(use(a).read_text())["checks"].items():
            for cr, v in r.items():
                num(f"assumption[{name}]_{cr}_rms_mV", v["dV_rms_mV"], "mV", rel(a))
                num(f"assumption[{name}]_{cr}_max_mV", v["dV_max_mV"], "mV", rel(a))
    c = RES / "paper" / "cost_table.json"
    if c.exists():
        shutil.copy(use(RES / "paper" / "cost_table.md"), TAB / "T4_cost.md")
        cost = json.loads(use(c).read_text())
        for r in cost["fv_forward"]:
            num(f"cost_fv_{r['crate']:g}C_{r['grid'][0]}x{r['grid'][1]}_cpu_s", r["cpu_s"], "s", rel(c))
        for r in cost.get("pinn_step_timing", []):
            num(f"cost_pinn_training_{r['crate']:g}C_cpu_h", r["projected_cpu_h"], "h", rel(c))
        for r in cost.get("pinn_evaluation", []):
            num(f"cost_pinn_eval_voltage_{r['crate']:g}C_ms", r["voltage_1000_times_ms"], "ms", rel(c))
        inv = cost.get("inverse", {})
        if "finite_volume" in inv:
            num("cost_inverse_fv_cpu_h", inv["finite_volume"]["cpu_h"], "h", rel(c))
        if "pinn" in inv:
            num("cost_inverse_pinn_wall_h", inv["pinn"]["inverse_only_wall_h"], "h", rel(c))
            if inv["pinn"].get("forward_pretraining_wall_h"):
                num("cost_inverse_pinn_pretraining_wall_h", inv["pinn"]["forward_pretraining_wall_h"], "h", rel(c))


# ------------------------------------------------------------------------------------------------------- manuscript
def fmt(v):
    if not isinstance(v, (int, float)):
        return str(v)
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}" if a >= 1 else f"{v:.3g}"


def fill_manuscript():
    """Replace every {{key}} of paper/manuscript/manuscript.md by numbers.json[key] -> manuscript_filled.md; report
    unresolved keys (a number in the text without a traceable source is an error to fix, not to type by hand)."""
    import re
    src = OUT / "manuscript" / "manuscript.md"
    if not src.exists():
        return
    text = use(src).read_text()
    missing = []

    def sub(m):
        key = m.group(1)
        if key in NUMBERS:
            return fmt(NUMBERS[key]["value"])
        missing.append(key)
        return "**[" + key + "?]**"
    (OUT / "manuscript" / "manuscript_filled.md").write_text(re.sub(r"\{\{(.+?)\}\}", sub, text))
    print(f"manuscript: {len(re.findall(r'{{(.+?)}}', text))} placeholders, {len(missing)} unresolved"
          + (": " + "; ".join(sorted(set(missing))) if missing else ""))


# --------------------------------------------------------------------------------------------------------- manifest
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest():
    code = sorted(set(list((ROOT / "src" / "dfn_pinn" / "lispan").glob("*.py")) + list((ROOT / "scripts").glob("paper_*.py"))
                      + [ROOT / "scripts" / f for f in ("lispan_train.py", "lispan_inverse_multirate.py", "lispan_fit_experiment.py",
                                                       "lispan_make_inverse_data.py", "lispan_identifiability.py",
                                                       "lispan_discharge.py", "lispan_span500.py", "lispan_digitize_paper.py")]))
    import scipy
    return {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                         "torch": torch.__version__},
            "inputs": {rel(p): sha256(p) for p in sorted(INPUTS) if Path(p).is_file()},
            "code": {rel(p): sha256(p) for p in code if p.exists()},
            "outputs": {rel(p): sha256(p) for p in sorted(list(FIG.glob("*")) + list(TAB.glob("*")))}}


def recompute():
    py = [sys.executable, "-u"]
    cmds = [py + ["scripts/lispan_discharge.py", "--crates", "0.1", "1", "--zcc", "0", "--out", "results/lispan/ref_Zcc0", "--fig6a"],
            py + ["scripts/lispan_discharge.py", "--crates", "0.05", "0.1", "0.2", "1", "--zcc", "0.025", "--out",
                  "results/lispan/ref_Zcc0.025"],
            py + ["scripts/paper_assumption_checks.py"],
            py + ["scripts/paper_inverse_error_model.py"],
            py + ["scripts/paper_cost_table.py"]]
    for c in cmds:
        print("+", " ".join(c[2:]), flush=True)
        subprocess.run(c, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="rerun the cheap FV scripts first")
    ap.add_argument("--forward-runs", default=None, help='JSON {"0.1": [run, ...], "1": [...]} instead of the final-recipe runs')
    ap.add_argument("--only", nargs="*", default=None, help="subset of: reference forward identifiability inverse measured tables")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.recompute:
        recompute()
    for d in (FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    plt = plt_setup()
    steps = {"reference": lambda: fig_reference(plt),
             "forward": lambda: fig_forward(plt, json.loads(args.forward_runs) if args.forward_runs else None),
             "identifiability": lambda: fig_identifiability(plt), "inverse": lambda: fig_inverse(plt),
             "measured": lambda: fig_measured(plt), "tables": copy_tables}
    old = json.loads((OUT / "numbers.json").read_text()) if (args.only and (OUT / "numbers.json").exists()) else {}
    NUMBERS.update(old)
    for name, fn in steps.items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        fn()
        print(f"{name}: done in {time.time() - t0:.0f} s", flush=True)
    (OUT / "numbers.json").write_text(json.dumps(dict(sorted(NUMBERS.items())), indent=1))
    fill_manuscript()
    # the manifest describes a complete build only; partial builds (--only) write their own
    (OUT / ("MANIFEST_partial.json" if args.only else "MANIFEST.json")).write_text(json.dumps(manifest(), indent=1))
    print(f"{len(NUMBERS)} numbers, {len(INPUTS)} input files -> {rel(OUT)}")


if __name__ == "__main__":
    sys.exit(main())
