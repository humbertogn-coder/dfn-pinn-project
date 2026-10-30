"""Two-curve fits of the cells in SPAN-DB (Li-SPAN-Papers/span_db): every cell with discharge curves at two or more
specific currents is calibrated on its slowest current and the fastest one the network accepts, and the remaining
currents are predicted (docs/TWO_CURVE_TOOL.md).

    python scripts/lispan_spandb_fit.py --spandb <span_db folder> --workers 2 --fv-check
    python scripts/lispan_spandb_fit.py --spandb <span_db> --cells 2019_Chen_088186_C1 --network <checkpoint>

Curve choice per cell:
  - discharge curves with specific capacity (mAh/g) and a known specific current, qc_status ok or flagged (not
    rejected); the rate test (test_type 'rate') when it has two or more currents, else all tests of the cell (flagged
    'mixed_tests');
  - one curve per current, the lowest cycle number but never cycle 1 (SPAN's first discharge is an activation; kept
    only when every curve of the test is numbered 1, i.e. per-step numbering); partial curves (points starting
    beyond 15 % of the end capacity) are dropped;
  - rate order: over the common capacity window the mean voltage must fall with current (25 mV tolerance); the
    curve with most violations (ties: the slowest) is dropped until the order holds;
  - current and capacity on the same mass basis: if the current basis (tests.c_rate_basis) differs from the capacity
    basis (curves.x_basis) it is converted with the S content of the SPAN and the active fraction of the electrode;
    an unreported basis is taken equal to the other one ('basis_assumed' / 'capacity_basis_assumed'), both unreported
    is one unknown basis ('basis_unknown': w_S absorbs it and is not comparable with the S content).
Calibration pair: w_S is estimated from the slowest curve (end capacity / 1200 mAh/g); the slowest and fastest currents
whose estimated model rates sit 1.25x inside the network's range; fallback: the pair whose feasible w_S range is closest
to (or contains) the estimate, widest first.  Predictions are scored inside the network window (T <= 1).
Points: 80, uniform in capacity from 2 % to 90 % of the measured end capacity (the tool's CSV rule).
Output: results/lispan/spandb/<cell>.json, summary.csv and summary.json.
"""

import argparse
import csv
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dfn_pinn.lispan.chem_pinn import load_chem_model  # noqa: E402
from dfn_pinn.lispan.two_curve import Curve, TwoCurveFit, fv_check, model_time, network_sd_for, w_bounds, W_LIMITS  # noqa: E402

OUT = ROOT / "results" / "lispan" / "spandb"
_W = {}


def num(v):
    try:
        x = float(str(v).strip().split()[0])
        return x if math.isfinite(x) else None
    except (ValueError, IndexError):
        return None


def load_db(root):
    root = Path(root)
    rd = lambda f: list(csv.DictReader(open(root / "db" / f, encoding="utf-8")))       # noqa: E731
    db = {"papers": {r["paper_id"]: r for r in rd("papers.csv")}, "cells": {r["cell_id"]: r for r in rd("cells.csv")},
          "tests": {r["test_id"]: r for r in rd("tests.csv")}, "electrodes": {r["electrode_id"]: r for r in rd("electrodes.csv")},
          "materials": {r["material_id"]: r for r in rd("materials.csv")}, "curves": rd("curves.csv"),
          "scalars": rd("scalars.csv"), "root": root}
    return db


def basis_key(b):
    b = (b or "").strip().lower()
    if b in ("", "nr", "na", "n/a"):
        return "NR"
    if b.startswith("span"):
        return "SPAN"
    if b == "s" or b.startswith("s ") or b.startswith("s("):
        return "S"
    if b.startswith("electrode"):
        return "electrode" if "tap" not in b else None
    return None


def cell_curves(db, cell_id):
    """Chosen discharge curves of a cell: list of dicts (current A/g on the capacity basis, q, V, label, ...) and notes."""
    cell = db["cells"][cell_id]
    el = db["electrodes"].get(cell["electrode_id"], {})
    mat = db["materials"].get(el.get("material_id", ""), {})
    s_frac = num(mat.get("s_content_wt"))
    s_frac = s_frac / 100.0 if s_frac and s_frac > 1.0 else s_frac
    act = num(el.get("active_wt"))
    act = act / 100.0 if act and act > 1.0 else act
    per_span = {"SPAN": 1.0, "S": s_frac, "electrode": (1.0 / act) if act else None, "unknown": None}   # mass of the basis per g SPAN
    rows = []
    for k in db["curves"]:
        t = db["tests"].get(k["test_id"])
        if not t or t["cell_id"] != cell_id or k["direction"] != "discharge" or k["qc_status"] == "rejected":
            continue
        if k["x_quantity"] != "specific_capacity":
            continue
        i = num(k["current_specific_mA_g"])
        xb = basis_key(k["x_basis"])
        cb = basis_key(t["c_rate_basis"])
        if i is None or i <= 0 or xb is None or cb is None:
            continue
        notes = []
        if xb == "NR" and cb == "NR":
            xb = cb = "unknown"                          # same (unreported) basis assumed: w_S absorbs it, not comparable
            notes.append("basis_unknown")
        elif xb == "NR":
            xb = cb
            notes.append("capacity_basis_assumed")
        elif cb == "NR":
            cb = xb
            notes.append("basis_assumed")
        if cb != xb:
            fa, fb = per_span.get(cb), per_span.get(xb)
            if not fa or not fb:
                continue
            i = i * fa / fb
            notes.append(f"current converted {cb}->{xb}")
        p = db["root"] / (k["points_file"] or f"curves/{k['curve_id']}.csv")
        if not p.exists():
            continue
        d = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        if d.shape[0] < 10:
            continue
        cyc = num(k["cycle_number"])
        rows.append({"curve_id": k["curve_id"], "test_id": k["test_id"], "test_type": t["test_type"], "i_sp": i / 1000.0,
                     "label": k["legend_label"] or k["rate_reported"], "cycle": cyc, "q_raw": d[:, 0], "v_raw": d[:, 1],
                     "qc": k["qc_status"], "x_basis": xb, "notes": notes, "figure": k["figure"]})
    rate_rows = [r for r in rows if r["test_type"] == "rate"]
    by_test = {}
    for r in rate_rows:
        by_test.setdefault(r["test_id"], []).append(r)
    pick, mixed = None, False
    for tid, rs in sorted(by_test.items(), key=lambda kv: -len({round(r["i_sp"], 6) for r in kv[1]})):
        if len({round(r["i_sp"], 6) for r in rs}) >= 2:
            pick = rs
            break
    if pick is None:
        pick, mixed = rows, True
    chosen = {}
    for r in pick:
        key = round(r["i_sp"], 6)
        chosen.setdefault(key, []).append(r)
    out, first_dropped, partial = [], [], []
    per_step = all(r["cycle"] == 1 for r in pick)                        # cycle 1 at every step: per-step numbering
    is_full = lambda r: float(np.min(r["q_raw"])) <= PARTIAL_START * float(np.max(r["q_raw"]))   # noqa: E731
    n_full = sum(any(is_full(r) for r in rs) for rs in chosen.values())
    for key, rs in sorted(chosen.items()):
        if n_full >= 2:                                                  # partial curves only when nothing better exists
            rs = [r for r in rs if is_full(r)]
            if not rs:
                partial.append(chosen[key][0]["label"])
                continue
        rs2 = rs if per_step else [r for r in rs if r["cycle"] != 1]     # SPAN's first discharge is an activation cycle
        if not rs2:
            first_dropped.append(rs[0]["label"])
            continue
        rs2.sort(key=lambda r: (r["cycle"] if r["cycle"] is not None else 1e9, r["qc"] != "ok"))
        out.append(rs2[0])
    order_dropped = rate_order_check(out)
    out = [r for r in out if r["label"] not in order_dropped]
    info = {"s_frac": s_frac, "active_frac": act, "mixed_tests": mixed, "x_bases": sorted({r["x_basis"] for r in out}),
            "first_cycle_dropped": first_dropped, "partial_dropped": partial, "rate_order_dropped": order_dropped,
            "basis_notes": sorted({n for r in out for n in r["notes"]})}
    if len(info["x_bases"]) > 1:
        out = [r for r in out if r["x_basis"] == out[0]["x_basis"]]
    return out, info


PARTIAL_START = 0.15   # a curve whose points start beyond this fraction of its end capacity is partial (hidden start)
RATE_ORDER_TOL = 0.025  # V: a slower discharge may not sit this far below a faster one (mean over the common window)


def rate_order_check(rows):
    """Labels of curves that break the rate order: over the common capacity window (5-70 % of the smallest end
    capacity) the mean voltage must fall with current.  The curve with the most violations is dropped until none
    is left (an activation-like first step of a rate test, a mislabelled or misdigitized line)."""
    if len(rows) < 3:
        return []
    qmax = min(float(np.max(r["q_raw"])) for r in rows)
    qmin = max(float(np.min(r["q_raw"])) for r in rows)
    lo, hi = max(0.05 * qmax, qmin), 0.70 * qmax
    if hi <= lo:
        return []
    grid = np.linspace(lo, hi, 40)
    vbar = {}
    for r in rows:
        o = np.argsort(r["q_raw"])
        vbar[r["label"]] = float(np.mean(np.interp(grid, r["q_raw"][o], r["v_raw"][o])))
    live = sorted(rows, key=lambda r: r["i_sp"])
    dropped = []
    while len(live) >= 3:
        bad = {r["label"]: 0 for r in live}
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                if vbar[live[a]["label"]] < vbar[live[b]["label"]] - RATE_ORDER_TOL:
                    bad[live[a]["label"]] += 1
                    bad[live[b]["label"]] += 1
        worst = max(bad, key=bad.get)
        if bad[worst] == 0:
            break
        # ties: drop the slowest (the usual culprit is the activation-like first, slowest step of a rate test)
        worst = next(r["label"] for r in live if bad[r["label"]] == bad[worst])
        dropped.append(worst)
        live = [r for r in live if r["label"] != worst]
    return dropped


def to_curve(r, frac=0.90, n=80):
    q, v = r["q_raw"], r["v_raw"]
    order = np.argsort(q)
    q, v = q[order], v[order]
    qq = np.linspace(q.min() + 0.02 * (q.max() - q.min()), frac * q.max(), n)
    return Curve(r["i_sp"], qq, np.interp(qq, q, v), r["label"])


Q_REF = 1200.0          # mAh/g of model sulfur: FV capacity of the nominal chemistry at 0.1-0.3 C (1240 at 0.1 C), the
                        # yardstick for a first w_S estimate
RATE_MARGIN = 1.25      # keep the estimated model rates this factor inside the network's range (leaves w_S free)


def choose_pair(model, curves, q_ends):
    """Calibration pair (indices into curves, sorted by current): estimate w_S from the slowest curve's end capacity
    (q_end / Q_REF), map every current to a model rate, keep those inside [r_min, r_max] with RATE_MARGIN, and take the
    slowest and the fastest of them.  Fallback (fewer than two inside): the widest pair with a non-empty w_S range."""
    w_est = q_ends[0] / Q_REF
    lo, hi = model.rate_min * RATE_MARGIN, model.rate_max / RATE_MARGIN
    inside = [k for k, c in enumerate(curves) if lo <= c.i_sp * model.p.m_S_model * 1e3 / (w_est * 10.0) <= hi]
    if len(inside) >= 2:
        return inside[0], inside[-1], "estimated", w_est
    best = None
    for a in range(len(curves)):
        for b in range(a + 1, len(curves)):
            wl, wh = w_bounds(model.p, [curves[a], curves[b]], model.rate_min, model.rate_max, W_LIMITS)
            if wl > wh * (1 + 1e-9):
                continue
            dist = max(0.0, math.log(wl / w_est), math.log(w_est / wh))   # how far the estimate is from the range
            key = (round(dist, 3), -math.log(curves[b].i_sp / curves[a].i_sp))
            if best is None or key < best[0]:
                best = (key, a, b)
    if best is None:
        return None
    return best[1], best[2], "fallback_closest" if best[0][0] > 0 else "fallback_contains", w_est


def _init(network):
    torch.set_num_threads(1)
    _W["model"] = load_chem_model(network)[0]
    _W["net_sd"] = network_sd_for(network)


def run_cell(job):
    cell_id, args = job
    db = _W.setdefault("db", load_db(args["spandb"]))
    model = _W["model"]
    t0 = time.time()
    try:
        rows, info = cell_curves(db, cell_id)
        cell = db["cells"][cell_id]
        paper = db["papers"].get(cell_id.rsplit("_C", 1)[0], {})
        base = {"cell": cell_id, "paper": paper.get("paper_id", cell_id.rsplit("_C", 1)[0]), "doi": paper.get("doi"),
                "electrolyte_family": cell["electrolyte_family"], "electrolyte": f"{cell['electrolyte_salts']} in {cell['electrolyte_solvents']}",
                "info_level": cell["info_level"], **info, "n_currents": len(rows)}
        if len(rows) < 2:
            return {**base, "status": "skipped", "reason": "fewer than two usable currents"}
        curves = [to_curve(r, args["frac"], args["n_points"]) for r in rows]
        pair = choose_pair(model, curves, [float(np.max(r["q_raw"])) for r in rows])
        if pair is None:
            return {**base, "status": "skipped", "reason": "no pair inside the network's rate range",
                    "currents_A_g": [c.i_sp for c in curves]}
        a, b, pair_rule, w_est = pair
        base.update(pair_rule=pair_rule, w_est=w_est)
        fit = TwoCurveFit(model, [curves[a], curves[b]], network_sd=_W["net_sd"])
        sols = fit.fit(args["starts"], 0)
        rep = fit.report(sols)
        x = np.asarray(rep["x"])
        preds = []
        for k, c in enumerate(curves):
            if k in (a, b):
                continue
            try:
                V, T_max = fit.predict(x, c.i_sp, c.q)
                tt = model_time(c.q, c.i_sp, model.ramp_s)
                inside = tt / tt[-1] * T_max <= 1.0                       # T = t / t_end is proportional to t
                if inside.sum() < 5:
                    preds.append({"label": c.label, "i_sp": c.i_sp, "outside_window": float(T_max)})
                    continue
                preds.append({"label": c.label, "i_sp": c.i_sp,
                              "rms_mV": 1e3 * float(np.sqrt(np.mean((V[inside] - c.V[inside]) ** 2))),
                              "rms_all_mV": 1e3 * float(np.sqrt(np.mean((V - c.V) ** 2))), "coverage": float(inside.mean()),
                              "T_max": T_max, "between": curves[a].i_sp < c.i_sp < curves[b].i_sp,
                              "q": c.q.tolist(), "V": V.tolist(), "V_data": c.V.tolist()})
            except ValueError as exc:
                preds.append({"label": c.label, "i_sp": c.i_sp, "outside_range": str(exc)})
        fvc = fv_check(fit, x, refit=False) if args["fv_check"] else None
        Vm = fit.model_curves(x)
        return {**base, "status": "fitted", "calibration": [rows[a]["label"], rows[b]["label"]],
                "calibration_curves": [rows[a]["curve_id"], rows[b]["curve_id"]],
                "calibration_cycles": [rows[a]["cycle"], rows[b]["cycle"]], "curve_notes": sorted({n for r in rows for n in r["notes"]}),
                "currents_A_g": [c.i_sp for c in curves], "labels": [c.label for c in curves],
                "fit": {k: rep[k] for k in ("params", "sd", "sd_total", "rms_mV", "rms_per_curve_mV", "T_max", "model_rates",
                                            "U0", "b", "k0_1_mult", "flags", "spread", "x")},
                "curves": [{"label": c.label, "i_sp": c.i_sp, "q": c.q.tolist(), "V": c.V.tolist(), "V_fit": v.tolist()}
                           for c, v in zip([curves[a], curves[b]], Vm)],
                "predictions": preds,
                "fv_check": None if fvc is None else {"curves": [{k: v for k, v in r.items() if k != "V"} for r in fvc["curves"]]},
                "wall_s": time.time() - t0}
    except Exception as exc:                                            # noqa: BLE001 - record and go on
        return {"cell": cell_id, "status": "error", "reason": repr(exc), "trace": traceback.format_exc()[-1500:]}


def write_summary(out, results, network, n_cells, wall_s):
    """summary.csv / summary.json from the per-cell results (also rebuilt from the per-cell JSONs by --summary-only)."""
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "paper", "electrolyte_family", "info_level", "status", "calibration", "rms_mV", "w_S", "sd_w_S_pct",
                    "dU0_2_mV", "dU0_3rel2_mV", "b_12", "b_3", "k0_1", "d_mV", "Z_CC_ohm_cm2", "U0_1", "U0_2", "U0_3",
                    "n_pred", "pred_rms_median_mV", "flags", "s_frac", "x_basis"])
        for r in sorted(results, key=lambda r: r["cell"]):
            if r["status"] != "fitted":
                w.writerow([r["cell"], r.get("paper"), r.get("electrolyte_family"), r.get("info_level"), r["status"], r.get("reason")])
                continue
            f = r["fit"]; p = f["params"]
            pr = [x["rms_mV"] for x in r["predictions"] if "rms_mV" in x]
            sd_w = f["sd_total"].get("w_S", float("nan"))          # absent when w_S is fixed by the rate range
            w.writerow([r["cell"], r["paper"], r["electrolyte_family"], r["info_level"], "fitted", " + ".join(r["calibration"]),
                        f"{f['rms_mV']:.2f}", f"{p['w_S']:.4f}", f"{sd_w:.2f}", f"{1e3 * p['dU0_2']:.1f}",
                        f"{1e3 * p['dU0_3rel2']:.1f}", f"{p['b_12']:.3f}", f"{p['b_3']:.3f}", f"{p['k0_1']:.3g}",
                        f"{1e3 * p['d']:.1f}", f"{1e4 * p['Z_CC']:.1f}", *[f"{u:.4f}" for u in f["U0"]],
                        len(pr), f"{np.median(pr):.1f}" if pr else "", ";".join(sorted({fl['flag'] for fl in f['flags'] if fl['level'] != 'info'})),
                        r.get("s_frac"), ",".join(r.get("x_bases", []))])
    (out / "summary.json").write_text(json.dumps({"network": network, "n_cells": n_cells, "wall_s": wall_s,
                                                  "results": [{k: v for k, v in r.items() if k not in ("curves",)} for r in results]},
                                                 indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spandb", required=True)
    ap.add_argument("--network", default=str(ROOT / "results" / "lispan_chem_runs" / "chem_v2_norar_s0" / "final.pt"))
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--frac", type=float, default=0.90)
    ap.add_argument("--n-points", type=int, default=80)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--fv-check", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--summary-only", action="store_true", help="rebuild summary.csv/json from the per-cell JSONs in --out")
    a = ap.parse_args()
    if a.summary_only:
        out = Path(a.out)
        res = [json.loads(f.read_text()) for f in sorted(out.glob("*.json")) if not f.name.startswith("summary")]
        write_summary(out, res, a.network, len(res), float("nan"))
        print("rebuilt", out / "summary.csv", f"({len(res)} cells)")
        return
    db = load_db(a.spandb)
    if a.cells:
        cells = a.cells
    else:
        cnt = {}
        for k in db["curves"]:
            t = db["tests"].get(k["test_id"])
            if t and k["direction"] == "discharge" and num(k["current_specific_mA_g"]):
                cnt.setdefault(t["cell_id"], set()).add(k["current_specific_mA_g"])
        cells = sorted(c for c, s in cnt.items() if len(s) >= 2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = {"spandb": a.spandb, "frac": a.frac, "n_points": a.n_points, "starts": a.starts, "fv_check": a.fv_check}
    print(f"{len(cells)} cells, network {a.network}", flush=True)
    results, t0 = [], time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.network,)) as pool:
        for r in pool.imap_unordered(run_cell, [(c, args) for c in cells]):
            results.append(r)
            (out / f"{r['cell']}.json").write_text(json.dumps(r, indent=1, default=float))
            if r["status"] == "fitted":
                f = r["fit"]
                pr = [p for p in r["predictions"] if "rms_mV" in p]
                print(f"{r['cell']:34s} {r['electrolyte_family'][:12]:12s} L{r['info_level']:2s} rms {f['rms_mV']:6.1f} mV | "
                      f"w_S {f['params']['w_S']:.3f} d {1e3 * f['params']['d']:+5.0f} Z {1e4 * f['params']['Z_CC']:5.0f} | "
                      f"pred {len(pr)}: " + ", ".join(f"{p['rms_mV']:.0f}" for p in pr) + " mV | "
                      + ",".join(sorted({fl['flag'] for fl in f['flags'] if fl['level'] != 'info'})) + f" ({r['wall_s']:.0f} s)", flush=True)
            else:
                print(f"{r['cell']:34s} {r['status']}: {r.get('reason')}", flush=True)
    write_summary(out, results, a.network, len(cells), time.time() - t0)
    print("written", out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
