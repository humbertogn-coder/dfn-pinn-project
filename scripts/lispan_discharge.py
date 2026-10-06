"""Li-SPAN reference discharges (Simanjuntak 2024 cell) at several C-rates, with comparison plots
against the digitized curves of the paper (results/lispan/paper_digitized/).

    python scripts/lispan_discharge.py --crates 0.05 0.1 0.2 1 --zcc 0.025 --out results/lispan/ref_Zcc0.025
    python scripts/lispan_discharge.py --crates 0.1 1 --zcc 0 --out results/lispan/ref_Zcc0 --fig6a

Writes one npz per rate (layout of dfn_pinn.lispan.model.export) and summary.json / figures.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.lispan import LiSPANParams, LiSPANProtocol, solve, export  # noqa: E402

DIG = ROOT / "results" / "lispan" / "paper_digitized"


def dig(name):
    f = DIG / name
    if not f.exists():
        return None
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    return d[np.argsort(d[:, 0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crates", nargs="+", type=float, default=[0.05, 0.1, 0.2, 1.0])
    ap.add_argument("--zcc", type=float, default=0.025)
    ap.add_argument("--k0", default=None, help="three effective frequency factors, comma separated")
    ap.add_argument("--Nc", type=int, default=20)
    ap.add_argument("--Ns", type=int, default=10)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--out", default=str(ROOT / "results" / "lispan" / "ref"))
    ap.add_argument("--fig6a", action="store_true", help="compare with Fig. 6a (Z_CC = 0) instead of Fig. 4b")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    kw = {"Z_CC": args.zcc}
    if args.k0:
        kw["k0"] = tuple(float(v) for v in args.k0.split(","))
    params = LiSPANParams(**kw)
    summary = {"params": params.to_dict(), "runs": []}
    results = {}
    for c in args.crates:
        prot = LiSPANProtocol.from_crate(c)
        t0 = time.perf_counter()
        _, res = solve(params, prot, N_c=args.Nc, N_s=args.Ns, rtol=args.rtol)
        dt = time.perf_counter() - t0
        name = f"discharge_{c:g}C"
        export(res, out / f"{name}.npz", params, prot)
        results[c] = res
        row = {"crate": c, "I_A_m2": prot.current, "Q_end_mAh_gS": float(res["Q_mAh_gS"][-1]), "Q_end_Ah_m2": float(res["Q_Ah_m2"][-1]),
               "V_end": float(res["V"][-1]), "t_end_s": float(res["t"][-1]), "eps_L_end": float(res["eps_L_avg"][-1]),
               "solver": res["solver"], "wall_s": dt}
        # paper comparison
        ref = dig(f"fig6a_k0_1e-2_{c:g}C.csv") if args.fig6a else dig(f"fig4b_{c:g}C.csv")
        if ref is not None:
            m = (ref[:, 0] >= 30) & (ref[:, 0] <= res["Q_mAh_gS"][-1])
            e = np.interp(ref[m, 0], res["Q_mAh_gS"], res["V"]) - ref[m, 1]
            row["paper_rms_mV"] = float(1e3 * np.sqrt(np.mean(e ** 2))); row["paper_max_mV"] = float(1e3 * np.abs(e).max())
            row["paper_Q_end"] = float(ref[-1, 0])
        summary["runs"].append(row)
        print(f"{c:g} C: Q_end {row['Q_end_mAh_gS']:.0f} mAh/g_S, t_end {row['t_end_s']:.0f} s, "
              + (f"vs paper rms {row['paper_rms_mV']:.1f} mV (max {row['paper_max_mV']:.0f}), paper Q_end {row['paper_Q_end']:.0f}; " if ref is not None else "")
              + f"{dt:.1f} s, nfev {res['solver']['nfev']}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax = axes[0]
    for i, c in enumerate(args.crates):
        res = results[c]
        ax.plot(res["Q_mAh_gS"], res["V"], color=colors[i % 10], label=f"{c:g} C (this work)")
        ref = dig(f"fig6a_k0_1e-2_{c:g}C.csv") if args.fig6a else dig(f"fig4b_{c:g}C.csv")
        if ref is not None:
            ax.plot(ref[:, 0], ref[:, 1], "o", ms=2.5, color=colors[i % 10], alpha=0.5,
                    label=f"{c:g} C paper {'Fig. 6a' if args.fig6a else 'Fig. 4b'}")
    ax.set_xlabel("specific capacity / mAh g$_S^{-1}$"); ax.set_ylabel("cell voltage / V"); ax.set_ylim(1.0, 3.0); ax.set_xlim(0, 1400)
    ax.set_title(f"Li-SPAN reference, Z_CC = {args.zcc} $\\Omega$ m$^2$"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax = axes[1]
    c = 0.1 if 0.1 in results else args.crates[0]
    res = results[c]
    for key, lab, col in (("c_S4", "PAN-S$_4$", "tab:blue"), ("c_S3", "PAN-S$_3$Li", "tab:red"), ("c_S2", "PAN-S$_2$Li", "tab:orange"), ("c_S1", "PAN-SLi", "tab:purple")):
        ax.plot(res["Q_mAh_gS"], res[key + "_avg"], color=col, label=lab)
        ref = dig(f"fig5a_{key}_0.1C.csv")
        if ref is not None and c == 0.1:
            ax.plot(ref[:, 0], ref[:, 1], "o", ms=2, color=col, alpha=0.4)
    ax.set_xlabel("specific capacity / mAh g$_S^{-1}$"); ax.set_ylabel("cathode-averaged concentration / mol m$^{-3}$")
    ax.set_title(f"SPAN species at {c:g} C (points: paper Fig. 5a)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax2 = ax.twinx(); ax2.plot(res["Q_mAh_gS"], res["eps_L_avg"], "k--", lw=1, label="$\\epsilon_{Li_2S}$"); ax2.set_ylabel("Li$_2$S volume fraction")
    ax2.legend(loc="center right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "discharge_vs_paper.png", dpi=150)
    # fields at 0.1 C
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    y_um = res["y"] * 1e6; yc_um = res["y_c"] * 1e6
    idx = np.linspace(0, len(res["t"]) - 1, 7).astype(int)[1:]
    for k, (key, lab, cath) in enumerate((("c_Li", "c$_{Li^+}$ / mol m$^{-3}$", False), ("c_S", "c$_{S^{2-}}$ / mol m$^{-3}$", False), ("phi_e", "$\\phi_e$ / V", False),
                                           ("dphi", "$\\Delta\\phi$ / V", True), ("c_S2", "c$_{PAN-S_2Li}$ / mol m$^{-3}$", True), ("eps_L", "$\\epsilon_{Li_2S}$", True))):
        ax = axes.flat[k]
        for j in idx:
            ax.plot(yc_um if cath else y_um, res[key][:, j], label=f"{res['Q_mAh_gS'][j]:.0f} mAh/g")
        if key == "c_S":
            ax.set_yscale("log")
        ax.set_xlabel("y / $\\mu$m"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
        if not cath:
            ax.axvline(params.L_cat * 1e6, color="gray", lw=0.8, ls=":")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Li-SPAN reference fields at {c:g} C"); fig.tight_layout(); fig.savefig(out / "fields_0.1C.png", dpi=130)
    print("written", out)


if __name__ == "__main__":
    main()
