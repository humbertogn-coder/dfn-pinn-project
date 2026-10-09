"""Quantify the modelling choices of the Li-SPAN benchmark (paper step 4: explicit assumptions).

Each check compares the finite-volume discharge voltage of the PINN benchmark (reaction 1 reversible, 2 and 3
irreversible, no double layer, Z_CC = 0.025 Ohm m2, 30 s current ramp, N_c = 40 / N_s = 20) with a variant that
changes one choice, at 0.1 C and 1 C.  Voltages are compared at equal delivered charge (the axis of every discharge
plot), from the charge passed at t = 100 s up to 98 % of the smaller of the two capacities; the change of the capacity
to 1.0 V is reported separately.  The largest salt-concentration excursion of the benchmark is also recorded (it
bounds the error of using constant 1 M electrolyte properties).

    python scripts/paper_assumption_checks.py      ->  results/paper/assumption_checks.json (+ .md)
"""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.lispan.params import LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.model import solve  # noqa: E402

BASE = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)
VARIANTS = {
    "reaction 2 reversible (finite-volume reference, TTF)": dict(params=replace(BASE, reversible=(True, True, False))),
    # 0.1 C only: at 1 C the reverse of reaction 3 makes the BDF solve crawl (> 10 min CPU, stopped)
    "all three reactions reversible (literal mass action, TTT)": dict(params=replace(BASE, reversible=(True, True, True)),
                                                                       rates=(0.1,)),
    "double layer c_DL = 0.1 F/m2 (paper value)": dict(params=replace(BASE, c_DL=0.1)),
    "current ramp 1 s instead of 30 s": dict(ramp=1.0),
    "coarser grid N_c = 20, N_s = 10": dict(grid=(20, 10)),
    "Z_CC = 0.035 Ohm m2 (best fit quoted in the text)": dict(params=replace(BASE, Z_CC=0.035)),
    "S2- saturation c_sat = 1e-4 mol/m3 (10x)": dict(params=replace(BASE, c_S_sat=1e-4)),
    "S2- kinetic reference c_S_ref = 0.1 mol/m3 (10x)": dict(params=replace(BASE, c_S_ref=0.1)),
    # brackets for the concentration dependence of the 1 M electrolyte properties (constant here, Lundgren 2014
    # correlations in the paper): a 10 % lower conductivity and a 20 % lower salt diffusivity everywhere
    "electrolyte conductivity -10 %": dict(params=replace(BASE, kappa0=0.9 * BASE.kappa0)),
    "salt diffusivity -20 %": dict(params=replace(BASE, D_salt=0.8 * BASE.D_salt)),
    "temperature 308.15 K (RT/F terms only, no Arrhenius)": dict(params=replace(BASE, T=308.15)),
}


def run(p, crate, ramp=30.0, grid=(40, 20)):
    t0 = time.time()
    _, res = solve(p, LiSPANProtocol.from_crate(crate, ramp_s=ramp), N_c=grid[0], N_s=grid[1], n_out=1500)
    return res, time.time() - t0


def compare(ref, var):
    q0 = float(np.interp(100.0, ref["t"], ref["Q_mAh_gS"]))
    q = np.linspace(q0, 0.98 * min(ref["Q_mAh_gS"][-1], var["Q_mAh_gS"][-1]), 600)
    e = np.interp(q, var["Q_mAh_gS"], var["V"]) - np.interp(q, ref["Q_mAh_gS"], ref["V"])
    return 1e3 * float(np.sqrt(np.mean(e ** 2))), 1e3 * float(np.abs(e).max())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="substrings of variant names to (re)compute; the others are "
                    "kept from the existing results/paper/assumption_checks.json")
    args = ap.parse_args()
    d = ROOT / "results" / "paper"
    old = json.loads((d / "assumption_checks.json").read_text()) if (args.only and (d / "assumption_checks.json").exists()) else None
    out = {"benchmark": "reversible TFF, c_DL 1e-6, Z_CC 0.025, ramp 30 s, N_c 40 / N_s 20",
           "comparison": "V at equal delivered charge, Q(t = 100 s) .. 0.98 min(Q_end)", "checks": {}}
    curves = {}
    for crate in (0.1, 1.0):
        ref, wall = run(BASE, crate)
        c0 = ref["c_Li"][:, 0].mean()
        out[f"benchmark_{crate:g}C"] = {
            "Q_end_mAh_gS": float(ref["Q_mAh_gS"][-1]), "wall_s": wall,
            "max_rel_salt_excursion": float(np.abs(ref["c_PF6"] / ref["c_PF6"][:, :1] - 1.0).max()),
            "max_c_S2m_mol_m3": float(ref["c_S"].max()), "c_Li_initial": float(c0)}
        curves[crate] = {"benchmark": ref}
        for name, v in VARIANTS.items():
            if crate not in v.get("rates", (0.1, 1.0)):
                continue
            if old is not None and not any(o in name for o in args.only) and f"{crate:g}C" in old["checks"].get(name, {}):
                out["checks"].setdefault(name, {})[f"{crate:g}C"] = old["checks"][name][f"{crate:g}C"]
                continue
            res, wall = run(v.get("params", BASE), crate, v.get("ramp", 30.0), v.get("grid", (40, 20)))
            curves[crate][name] = res
            rms, mx = compare(ref, res)
            out["checks"].setdefault(name, {})[f"{crate:g}C"] = {
                "dV_rms_mV": rms, "dV_max_mV": mx, "dQ_end_mAh_gS": float(res["Q_mAh_gS"][-1] - ref["Q_mAh_gS"][-1])}
            print(f"{crate:g} C  {name:52s} dV rms {rms:7.2f} mV, max {mx:7.2f} mV, dQ {res['Q_mAh_gS'][-1] - ref['Q_mAh_gS'][-1]:+.1f} mAh/g",
                  flush=True)
    d.mkdir(parents=True, exist_ok=True)
    (d / "assumption_checks.json").write_text(json.dumps(out, indent=1))
    lines = ["| choice | 0.1 C: dV rms / max [mV], dQ [mAh/g] | 1 C: dV rms / max [mV], dQ [mAh/g] |", "| --- | --- | --- |"]
    for name, r in out["checks"].items():
        cells = [f"{r[c]['dV_rms_mV']:.2f} / {r[c]['dV_max_mV']:.2f}, {r[c]['dQ_end_mAh_gS']:+.1f}" if c in r else "not run"
                 for c in ("0.1C", "1C")]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    (d / "assumption_checks.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(curves, d / "assumption_checks.png")


def plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2), sharex="col")
    colors = {name: f"C{i % 10}" for i, name in enumerate(VARIANTS)}      # same colour for a variant in both panels
    for j, (crate, cs) in enumerate(curves.items()):
        ref = cs["benchmark"]
        axs[0, j].plot(ref["Q_mAh_gS"], ref["V"], "k", lw=2, label="benchmark")
        for name, res in cs.items():
            if name == "benchmark" or "grid" in name:
                continue
            axs[0, j].plot(res["Q_mAh_gS"], res["V"], lw=1, color=colors[name], label=name if j == 0 else None)
            q = np.linspace(float(np.interp(100.0, ref["t"], ref["Q_mAh_gS"])),
                            0.98 * min(ref["Q_mAh_gS"][-1], res["Q_mAh_gS"][-1]), 600)
            axs[1, j].plot(q, 1e3 * (np.interp(q, res["Q_mAh_gS"], res["V"]) - np.interp(q, ref["Q_mAh_gS"], ref["V"])), lw=1,
                           color=colors[name])
        axs[0, j].set_title(f"{crate:g} C")
        axs[0, j].set_ylabel("cell voltage [V]")
        axs[1, j].set_ylabel("variant - benchmark [mV]")
        axs[1, j].set_xlabel("capacity [mAh/g S]")
        axs[1, j].set_ylim(-40, 40)
        axs[1, j].grid(alpha=0.3)
    fig.legend(loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    sys.exit(main())
