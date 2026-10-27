"""Two-curve tool: fit measured Li-SPAN discharge curves (two or more C-rates) with the Phase 3b chemistry network
(src/dfn_pinn/lispan/two_curve.py).  Returns w_S, the five chemistry inputs (dU0_2, dU0_3rel2, b_12, b_3, k0_1), the
common U0 shift d and Z_CC with linearized standard deviations, correlations and validity flags; optionally checks
the solution with the finite-volume solver and predicts other rates.

    python scripts/lispan_two_curve_fit.py --dataset chen2019_ether --rates 0.4 1 --fv-check --plot
    python scripts/lispan_two_curve_fit.py --dataset hao2025_SPAN --rates 0.2 1 --predict 0.5 --fv-check --plot
    python scripts/lispan_two_curve_fit.py --experiment 0.1 1 --predict 0.2 --fv-check --plot
    python scripts/lispan_two_curve_fit.py --csv cellA_0.1C.csv:0.167 cellA_1C.csv:1.675 --name cellA --plot

Data sources
  --dataset NAME --rates ...   the literature curves of scripts/lispan_lit_rates.py (data/literature_span/; rates in
                               the dataset's units, A/g or C); n points uniform in capacity from the dataset's q_min to
                               frac (0.90) x the measured end capacity
  --experiment RATE ...        Simanjuntak et al. Fig. 4b (results/lispan/inverse_data/V_<rate>C_experiment.npz,
                               capacity per model sulfur, so w_S = 1 is the paper's normalization); the digitized points
                               with q < frac x the end capacity
  --csv FILE:I_SP[:LABEL] ...  your own curves: two columns capacity [mAh/g] and voltage [V] (one header line), at the
                               specific current I_SP [A/g] on the same capacity basis
Every curve must map inside the network's 0.1-1 C at the fitted w_S (model rate = 0.767 i / w_S for i in A/g): the
fastest/slowest current ratio must be <= 10.  With exactly 10, w_S is fixed by the range.

Parameters in natural units for --fix: w_S (factor), dU0_2, dU0_3rel2, d [V], b_12, b_3, k0_1 (multipliers), Z_CC
[Ohm m2], e.g. --fix w_S=1 k0_1=1.  Results: results/lispan/two_curve/<name>.json (+ .png with --plot).
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dfn_pinn.lispan.chem_pinn import load_chem_model  # noqa: E402
from dfn_pinn.lispan.two_curve import Curve, TwoCurveFit, fv_check, fv_curve, m_S_g  # noqa: E402

NETWORK = ROOT / "results" / "lispan_chem_runs" / "chem_v2_norar_s0" / "final.pt"
OUT = ROOT / "results" / "lispan" / "two_curve"


def dataset_curve(ds, rate, frac, n):
    import lispan_lit_rates as L
    cfg = L.DATASETS[ds]
    if rate not in cfg["rates"]:
        raise SystemExit(f"{ds}: no curve at {rate:g} {cfg['unit']} (have {sorted(cfg['rates'])})")
    d = np.loadtxt(L.DATA / cfg["dir"] / f"{cfg['prefix']}_{rate:g}{cfg['unit']}.csv", delimiter=",", skiprows=1)
    q, v = d[:, 0], d[:, 1]
    order = np.argsort(q)
    qq = np.linspace(cfg["q_min"], frac * q.max(), n)
    unit = {"Ag": "A/g", "C": "C"}.get(cfg["unit"], cfg["unit"])
    return Curve(cfg["rates"][rate], qq, np.interp(qq, q[order], v[order]), f"{rate:g} {unit}"), \
        {"q_raw": q, "v_raw": v}


def experiment_curve(rate, frac, params):
    z = np.load(ROOT / "results" / "lispan" / "inverse_data" / f"V_{rate:g}C_experiment.npz")
    q, V, I = np.asarray(z["Q_mAh_gS"], float), np.asarray(z["V"], float), float(np.median(z["I"]))
    keep = q < frac * q.max()
    return Curve(I / m_S_g(params), q[keep], V[keep], f"{rate:g} C"), {"q_raw": q, "v_raw": V}


def csv_curve(spec, frac, n):
    parts = spec.split(":")
    f, i_sp = parts[0], float(parts[1])
    label = parts[2] if len(parts) > 2 else Path(f).stem
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    q, v = d[:, 0], d[:, 1]
    order = np.argsort(q)
    qq = np.linspace(q.min() + 0.02 * (q.max() - q.min()), frac * q.max(), n)
    return Curve(i_sp, qq, np.interp(qq, q[order], v[order]), label), {"q_raw": q, "v_raw": v}


def parse_fix(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k.strip()] = float(v)
    return out


def plot(path, fit, rep, raw, preds, fvc, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID = "#1B2A3A", "#4A5568", "#E3E8EE"
    cols = ["#2A78D6", "#E0622C", "#2E8B57", "#7A4FB5"]
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160, "savefig.bbox": "tight", "axes.grid": True, "grid.color": GRID,
                         "axes.axisbelow": True})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1.6, 1]}, sharex=True)
    x = np.asarray(rep["x"])
    Vm = fit.model_curves(x)
    for k, (c, v, rw) in enumerate(zip(fit.curves, Vm, raw)):
        col = cols[k % len(cols)]
        a1.plot(rw["q_raw"], rw["v_raw"], ".", color=col, ms=3, alpha=0.35)
        a1.plot(c.q, v, "-", color=col, lw=1.8, label=f"{c.label}: network fit, rms {rep['rms_per_curve_mV'][k]:.1f} mV")
        a2.plot(c.q, 1e3 * (v - c.V), "-", color=col, lw=1.4)
        if fvc and fvc["curves"][k].get("rms_fv_vs_data_mV") is not None and "V" in fvc["curves"][k]:
            a1.plot(c.q, fvc["curves"][k]["V"], ":", color=INK, lw=1.2)
    for k, pr in enumerate(preds):
        col = cols[(len(fit.curves) + k) % len(cols)]
        if pr.get("q_raw") is not None:
            a1.plot(pr["q_raw"], pr["v_raw"], ".", color=col, ms=3, alpha=0.35)
        lab = f"{pr['label']}: prediction" + (f", rms {pr['rms_mV']:.1f} mV" if pr.get("rms_mV") is not None else "")
        a1.plot(pr["q"], pr["V"], "--", color=col, lw=1.8, label=lab)
        if pr.get("V_data") is not None:
            a2.plot(pr["q"], 1e3 * (np.asarray(pr["V"]) - np.asarray(pr["V_data"])), "--", color=col, lw=1.2)
    if fvc:
        a1.plot([], [], ":", color=INK, lw=1.2, label="FV solver at the fitted parameters")
    a1.set_ylabel("voltage [V]")
    a1.legend(frameon=False, fontsize=9, loc="lower left")
    p = rep["params"]
    a1.set_title(title, loc="left", fontweight="bold", color=INK, fontsize=11)
    txt = (f"w_S {p['w_S']:.3f}   d {1e3 * p['d']:+.0f} mV   Z_CC {1e4 * p['Z_CC']:.1f} Ohm cm2\n"
           f"dU0_2 {1e3 * p['dU0_2']:+.0f} mV   dU0_3rel2 {1e3 * p['dU0_3rel2']:+.0f} mV   b_12 x{p['b_12']:.2f}   "
           f"b_3 x{p['b_3']:.2f}   k0_1 x{p['k0_1']:.2g}")
    a1.text(0.99, 0.98, txt, transform=a1.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_ylabel("model - data [mV]"); a2.set_xlabel("capacity [mAh/g, data basis]")
    flags = [f["flag"] + (f" ({f['param']})" if "param" in f else "") for f in rep["flags"] if f["level"] != "info"]
    a2.set_title("flags: " + (", ".join(flags) if flags else "none"), loc="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset")
    src.add_argument("--experiment", nargs="+", type=float)
    src.add_argument("--csv", nargs="+")
    ap.add_argument("--rates", nargs="+", type=float, help="with --dataset: the curves to fit (dataset units)")
    ap.add_argument("--predict", nargs="*", type=float, default=[], help="rates to predict (dataset / experiment units)")
    ap.add_argument("--network", default=str(NETWORK))
    ap.add_argument("--fix", nargs="*", default=[])
    ap.add_argument("--frac", type=float, default=0.90, help="use the data up to frac x the end capacity")
    ap.add_argument("--n-points", type=int, default=80)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fv-check", action="store_true", help="FV curves at the solution and the network bias there")
    ap.add_argument("--fv-grid", nargs=2, type=int, default=[20, 10])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--name", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    t0 = time.time()
    model, cfg, params = load_chem_model(a.network)
    if a.dataset:
        if not a.rates:
            raise SystemExit("--dataset needs --rates")
        loaded = [dataset_curve(a.dataset, r, a.frac, a.n_points) for r in a.rates]
        name = a.name or f"{a.dataset}_{'+'.join(f'{r:g}' for r in a.rates)}"
    elif a.experiment:
        loaded = [experiment_curve(r, a.frac, params) for r in a.experiment]
        name = a.name or f"simanjuntak_{'+'.join(f'{r:g}' for r in a.experiment)}"
    else:
        loaded = [csv_curve(s, a.frac, a.n_points) for s in a.csv]
        name = a.name or "csv_" + "+".join(c.label for c, _ in loaded)
    curves, raw = [c for c, _ in loaded], [r for _, r in loaded]
    fit = TwoCurveFit(model, curves, fixed=parse_fix(a.fix))
    print(f"{name}: {len(curves)} curves ({', '.join(f'{c.label} = {c.i_sp:.4g} A/g, {len(c.q)} pts' for c in curves)}); "
          f"network {a.network}; free {[fit.names[j] for j in fit.free]}; w_S range "
          f"[{np.exp(fit.lo[0]):.3f}, {np.exp(fit.hi[0]):.3f}]", flush=True)
    sols = fit.fit(a.starts, a.seed, log=lambda s: print(s, flush=True))
    rep = fit.report(sols)
    x = np.asarray(rep["x"])
    print(f"best: rms {rep['rms_mV']:.2f} mV (per curve {', '.join(f'{v:.2f}' for v in rep['rms_per_curve_mV'])}), "
          f"model rates {', '.join(f'{r:.3f}' for r in rep['model_rates'])} C, T_max "
          f"{', '.join(f'{t:.3f}' for t in rep['T_max'])}")
    for n, v in rep["params"].items():
        print(f"  {n:10s} {v:12.5g}" + (f"   sd {rep['sd'][n]:.3g}" if n in rep["sd"] else "   (fixed)"))
    for f in rep["flags"]:
        print(f"  [{f['level']}] {f['flag']}: {f['msg']}")

    preds = []
    for r in a.predict:
        if a.dataset:
            import lispan_lit_rates as L
            cfg_ds = L.DATASETS[a.dataset]
            c, rw = dataset_curve(a.dataset, r, a.frac, a.n_points)
        else:
            c, rw = experiment_curve(r, a.frac, params)
        try:
            V, T_max = fit.predict(x, c.i_sp, c.q)
        except ValueError as exc:
            print(f"  predict {c.label}: {exc}")
            preds.append({"label": c.label, "error": str(exc)})
            continue
        rms = 1e3 * float(np.sqrt(np.mean((V - c.V) ** 2)))
        preds.append({"label": c.label, "i_sp": c.i_sp, "q": c.q.tolist(), "V": V.tolist(), "V_data": c.V.tolist(),
                      "rms_mV": rms, "T_max": T_max, "q_raw": rw["q_raw"].tolist(), "v_raw": rw["v_raw"].tolist()})
        print(f"  predict {c.label}: rms {rms:.2f} mV vs data, T_max {T_max:.3f}")

    fvc = None
    if a.fv_check:
        t1 = time.time()
        fvc = fv_check(fit, x, N=tuple(a.fv_grid), log=print)
        for c, row in zip(fit.curves, fvc["curves"]):
            V, _ = fv_curve(fit.p, fit.spec, x, c.i_sp, c.q, N=tuple(a.fv_grid), ramp_s=fit.model.ramp_s)
            row["V"] = None if V is None else [float(v) if np.isfinite(v) else None for v in V]
            print(f"  FV {c.label}: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                                  for k, v in row.items() if k not in ("label", "V")))
        for pr in preds:
            if "error" in pr:
                continue
            V, _ = fv_curve(fit.p, fit.spec, x, pr["i_sp"], np.asarray(pr["q"]), N=tuple(a.fv_grid),
                            ramp_s=fit.model.ramp_s)
            if V is not None:
                ok = np.isfinite(V)
                pr["rms_fv_vs_data_mV"] = 1e3 * float(np.sqrt(np.mean((V[ok] - np.asarray(pr["V_data"])[ok]) ** 2)))
                pr["rms_net_vs_fv_mV"] = 1e3 * float(np.sqrt(np.mean((V[ok] - np.asarray(pr["V"])[ok]) ** 2)))
                print(f"  FV prediction {pr['label']}: FV vs data {pr['rms_fv_vs_data_mV']:.2f} mV, network vs FV "
                      f"{pr['rms_net_vs_fv_mV']:.2f} mV")
        fvc["wall_s"] = time.time() - t1
        if "network_bias" in fvc:
            fvc["bias_over_sd"] = {n: fvc["network_bias"][n] / rep["sd"][n] if rep["sd"].get(n) else None
                                   for n in fvc["network_bias"]}
            big = [n for n, v in fvc["bias_over_sd"].items() if v is not None and abs(v) > 1.0]
            if big:
                rep["flags"].append({"flag": "network_bias", "level": "warn", "params": big,
                                     "msg": "the network's own error moves " + ", ".join(big) + " by more than 1 sd "
                                            "(FV self-check)"})
                print(f"  [warn] network_bias: {big}")

    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"name": name, "network": str(a.network), "args": vars(a), "curves": [
        {"label": c.label, "i_sp": c.i_sp, "q": c.q.tolist(), "V": c.V.tolist()} for c in curves],
        **rep, "predictions": preds, "fv_check": fvc, "wall_s": time.time() - t0}
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    print("written", OUT / f"{name}.json", f"({time.time() - t0:.0f} s)")
    if a.plot:
        plot(OUT / f"{name}.png", fit, rep, raw, preds, fvc, f"{name}: two-curve fit with the v2 network")
        print("written", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
