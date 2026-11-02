"""Compare the SPAN-DB two-curve fits of two networks (default: v2 chem_v2_norar_s0, 0.1-1 C, vs v4
chem_v4_warm_salt_s0, 0.08-3 C).

    python report/spandb/compare_networks.py
    python report/spandb/compare_networks.py --a results/lispan/spandb --b results/lispan/spandb_v4ws --tag v2_v4

Inputs : per-cell JSONs written by scripts/lispan_spandb_fit.py in the two result folders.
Outputs: report/spandb/figures/sdb_<tag>.png and report/spandb/compare_<tag>.json, plus a table on stdout.

  a  misfit per cell, network A vs network B; filled = same calibration pair (a direct check of the networks against
     each other on identical data), open = B uses a different (wider) pair; cells only B can fit on the right.
  b  cells described (misfit <= 25 mV) by both: parameter difference B - A in units of the combined sd_total
     (sqrt(sd_A^2 + sd_B^2)), per parameter; filled = same pair, open = B uses a wider pair (then the shift also
     measures how much the parameter depends on the rate span); box-edge parameters in grey.
  c  blind predictions inside the network window: rms error vs the current ratio to the nearest calibration current
     (extrapolation > 1 on the fast side, < 1 on the slow side, interpolation drawn at 1).
"""
import argparse
import json
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INK, INK2, GRID = "#1B2A3A", "#52514E", "#E4E2DC"
BLUE, ORANGE, AQUA, GREY = "#2A78D6", "#EB6834", "#1BAF7A", "#9A988F"
GOOD_MV = 25.0
NAMES = ["w_S", "dU0_2", "dU0_3rel2", "b_12", "b_3", "k0_1", "d", "Z_CC"]
# parameters in the tool's sd units: w_S, b, k0 relative (% of the value), dU0 and d in mV, Z_CC in Ohm cm2
UNIT = {"w_S": "rel", "b_12": "rel", "b_3": "rel", "k0_1": "rel", "dU0_2": 1e3, "dU0_3rel2": 1e3, "d": 1e3, "Z_CC": 1e4}


def load(folder):
    out = {}
    for p in sorted(Path(folder).glob("*.json")):
        if p.name.startswith("summary"):
            continue
        d = json.loads(p.read_text())
        if "cell" in d:
            out[d["cell"]] = d
    return out


def diff_in_sd(fa, fb, n):
    """(B - A) / sqrt(sd_A^2 + sd_B^2) in the tool's sd units, or None when a sd is missing (fixed parameter)."""
    a, b = fa["params"][n], fb["params"][n]
    sa, sb = fa["sd_total"].get(n), fb["sd_total"].get(n)
    if not sa or not sb:
        return None
    delta = 100.0 * np.log(b / a) if UNIT[n] == "rel" else UNIT[n] * (b - a)
    return float(delta / np.hypot(sa, sb))


def edge_params(fit):
    return {f.get("param") for f in fit["flags"] if f["flag"] == "box_edge"}


def preds(cell):
    """Scored blind predictions: (ratio to the nearest calibration current, rms mV)."""
    cal = cell["calibration"]
    cur = dict(zip(cell["labels"], cell["currents_A_g"]))
    lo, hi = min(cur[c] for c in cal), max(cur[c] for c in cal)
    out = []
    for p in cell["predictions"]:
        if "rms_mV" not in p:
            continue
        i = p["i_sp"]
        ratio = 1.0 if lo < i < hi else (i / hi if i >= hi else i / lo)
        out.append((ratio, p["rms_mV"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ROOT / "results" / "lispan" / "spandb"))
    ap.add_argument("--b", default=str(ROOT / "results" / "lispan" / "spandb_v4ws"))
    ap.add_argument("--label-a", default="v2 (0.1-1 C)")
    ap.add_argument("--label-b", default="v4 (0.08-3 C)")
    ap.add_argument("--tag", default="v2_v4")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    cells = sorted(set(A) | set(B))
    rows, same_pair, only_b, only_a = [], [], [], []
    for c in cells:
        ra, rb = A.get(c), B.get(c)
        fa = ra["fit"] if ra and ra.get("status") == "fitted" else None
        fb = rb["fit"] if rb and rb.get("status") == "fitted" else None
        row = {"cell": c, "family": (ra or rb)["electrolyte_family"],
               "pair_a": ra.get("calibration") if fa else None, "pair_b": rb.get("calibration") if fb else None,
               "rms_a": fa["rms_mV"] if fa else None, "rms_b": fb["rms_mV"] if fb else None,
               "status_a": ra.get("status") if ra else "missing", "status_b": rb.get("status") if rb else "missing",
               "reason_b": rb.get("reason") if rb else None}
        if fb and rb.get("fv_check"):
            row["net_vs_fv_b"] = float(np.median([x["rms_net_vs_fv_mV"] for x in rb["fv_check"]["curves"]]))
        if fa and fb:
            row["same_pair"] = row["pair_a"] == row["pair_b"]
            row["z"] = {n: diff_in_sd(fa, fb, n) for n in NAMES}
            row["edge"] = sorted(e for e in edge_params(fa) | edge_params(fb) if e)
            if row["same_pair"]:
                same_pair.append(row)
        elif fb:
            only_b.append(row)
        elif fa:
            only_a.append(row)
        row["pred_a"], row["pred_b"] = (preds(ra) if fa else []), (preds(rb) if fb else [])
        rows.append(row)

    fit_a = [r for r in rows if r["rms_a"] is not None]
    fit_b = [r for r in rows if r["rms_b"] is not None]
    good = lambda rs, k: [r for r in rs if r[k] <= GOOD_MV]  # noqa: E731
    summ = {
        "n_cells": len(cells), "fitted_a": len(fit_a), "fitted_b": len(fit_b),
        "good_a": len(good(fit_a, "rms_a")), "good_b": len(good(fit_b, "rms_b")),
        "median_rms_a": float(np.median([r["rms_a"] for r in fit_a])) if fit_a else None,
        "median_rms_b": float(np.median([r["rms_b"] for r in fit_b])) if fit_b else None,
        "only_b": [r["cell"] for r in only_b], "only_a": [r["cell"] for r in only_a],
        "same_pair": len(same_pair),
        "net_vs_fv_b_good_median": float(np.median([r["net_vs_fv_b"] for r in good(fit_b, "rms_b") if "net_vs_fv_b" in r]))
        if fit_b else None,
        "net_vs_fv_b_good_max": float(np.max([r["net_vs_fv_b"] for r in good(fit_b, "rms_b") if "net_vs_fv_b" in r]))
        if fit_b else None,
    }
    both = [r for r in rows if r["rms_a"] is not None and r["rms_b"] is not None
            and r["rms_a"] <= GOOD_MV and r["rms_b"] <= GOOD_MV]
    summ["both_described"] = [r["cell"] for r in both]
    zb = {n: [r["z"][n] for r in both if r["z"][n] is not None and n not in r["edge"]] for n in NAMES}
    summ["both_abs_z_median"] = {n: float(np.median(np.abs(v))) if v else None for n, v in zb.items()}
    summ["both_frac_abs_z_lt2"] = {n: float(np.mean(np.abs(v) < 2)) if v else None for n, v in zb.items()}
    if same_pair:
        d_rms = [r["rms_b"] - r["rms_a"] for r in same_pair]
        summ["same_pair_rms_diff_median"] = float(np.median(d_rms))
        summ["same_pair_rms_diff_range"] = [float(min(d_rms)), float(max(d_rms))]
        zs = {n: [r["z"][n] for r in same_pair if r["z"][n] is not None and n not in r["edge"]] for n in NAMES}
        summ["same_pair_abs_z_median"] = {n: float(np.median(np.abs(v))) if v else None for n, v in zs.items()}
        summ["same_pair_frac_abs_z_lt1"] = {n: float(np.mean(np.abs(v) < 1)) if v else None for n, v in zs.items()}
    for side in ("a", "b"):
        pr = [p for r in rows if r[f"rms_{side}"] is not None and r[f"rms_{side}"] <= GOOD_MV for p in r[f"pred_{side}"]]
        summ[f"pred_{side}"] = {"n": len(pr), "between": [p[1] for p in pr if p[0] == 1.0],
                               "extrapolated": [[p[0], p[1]] for p in pr if p[0] != 1.0]}

    print(f"{'cell':28s} {'family':14s} {'pair A':26s} {'rms A':>6s}  {'pair B':26s} {'rms B':>6s} {'netFV':>6s}")
    for r in rows:
        fmt = lambda v: f"{v:6.1f}" if v is not None else "     -"  # noqa: E731
        pb = r["pair_b"] or (r["status_b"] + (f" ({r['reason_b']})" if r["reason_b"] else ""))
        pa = r["pair_a"] or r["status_a"]
        same = "=" if r.get("same_pair") else " "
        print(f"{r['cell']:28s} {r['family'][:14]:14s} {str(pa)[:26]:26s} {fmt(r['rms_a'])} {same}{str(pb)[:26]:26s} "
              f"{fmt(r['rms_b'])} {fmt(r.get('net_vs_fv_b'))}")
    print(json.dumps({k: v for k, v in summ.items() if not k.startswith("pred_")}, indent=1))
    for side in ("a", "b"):
        p = summ[f"pred_{side}"]
        print(f"predictions {side} (described cells): n {p['n']}, between {np.round(p['between'], 1).tolist()}, "
              f"extrapolated {[(round(x, 2), round(y, 1)) for x, y in p['extrapolated']]}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.15, 1.3, 1]})
    lim = 100
    for r in rows:
        if r["rms_a"] is not None and r["rms_b"] is not None:
            same = r.get("same_pair")
            ax1.plot(min(r["rms_a"], lim), min(r["rms_b"], lim), "o", ms=8, color=BLUE,
                     mfc=BLUE if same else "white", mew=1.6, zorder=3)
    xb = 1.08 * lim
    for k, r in enumerate(only_b):
        ax1.plot(xb, min(r["rms_b"], lim), "D", ms=7, color=ORANGE, clip_on=False, zorder=3)
    ax1.plot([0, lim], [0, lim], color=INK2, lw=0.8)
    ax1.axhline(GOOD_MV, color=INK2, lw=0.8, ls=":"); ax1.axvline(GOOD_MV, color=INK2, lw=0.8, ls=":")
    ax1.set_xlim(0, 1.12 * lim); ax1.set_ylim(0, lim)
    ax1.set_xlabel(f"misfit with {a.label_a} [mV]"); ax1.set_ylabel(f"misfit with {a.label_b} [mV]")
    ax1.plot([], [], "o", color=BLUE, label="same calibration pair")
    ax1.plot([], [], "o", color=BLUE, mfc="white", mew=1.6, label="wider pair with B")
    ax1.plot([], [], "D", color=ORANGE, label=f"only B fits ({len(only_b)})")
    ax1.legend(loc="upper left", fontsize=9, frameon=False)
    ax1.set_title("a  Misfit per cell (capped at 100 mV)", loc="left", fontweight="bold", color=INK, fontsize=11)

    for j, n in enumerate(NAMES):
        for r in both:
            z = r["z"][n]
            if z is None:
                continue
            col = GREY if n in r["edge"] else BLUE
            x = j + np.random.default_rng(zlib.crc32((r["cell"] + n).encode())).uniform(-0.22, 0.22)
            ax2.plot(x, np.clip(z, -6, 6), "o", color=col, ms=6, alpha=0.85,
                     mfc=col if r["same_pair"] else "white", mew=1.4)
    ax2.axhspan(-2, 2, color=GRID, alpha=0.7, lw=0)
    ax2.set_xticks(range(len(NAMES))); ax2.set_xticklabels(NAMES, fontsize=9)
    ax2.set_ylim(-6.3, 6.3); ax2.set_ylabel("(B - A) / combined sd_total (clipped at 6)")
    ax2.set_title(f"b  Cells described by both ({len(both)}): parameter shift; open = wider pair,\n"
                  "    grey = box edge; band = 2 sd", loc="left", fontweight="bold", color=INK, fontsize=11)

    for side, col, mk, lab in (("a", BLUE, "o", a.label_a), ("b", ORANGE, "s", a.label_b)):
        p = summ[f"pred_{side}"]
        xj = 1.0 + (-0.04 if side == "a" else 0.04)
        ax3.plot([xj] * len(p["between"]), p["between"], mk, color=col, alpha=0.8, ms=7, label=f"{lab}: {p['n']}")
        if p["extrapolated"]:
            x, y = np.array(p["extrapolated"]).T
            ax3.plot(x, y, mk, color=col, alpha=0.8, ms=7)
    ax3.set_xscale("log"); ax3.axhline(GOOD_MV, color=INK2, lw=0.8, ls=":")
    ax3.set_xlabel("current / nearest calibration current (1 = between)"); ax3.set_ylabel("prediction rms [mV]")
    ax3.legend(fontsize=9, frameon=False, loc="upper center")
    ax3.set_title("c  Blind predictions, described cells", loc="left", fontweight="bold", color=INK, fontsize=11)
    fig.tight_layout()
    out = HERE / "figures" / f"sdb_{a.tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    (HERE / f"compare_{a.tag}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
