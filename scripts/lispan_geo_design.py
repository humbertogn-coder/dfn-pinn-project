"""FV design, time window and evaluation references for network v5 (chemistry + cathode geometry; docs/GEOMETRY.md).

Conditions z = (rate, 5 chemistry inputs of the v2/v3 box, ln L_cat, f) with the fixed polyolefin separator
(geometry.polyolefin_cell) and the network's contact resistance 0.  Points outside the transport domain
(geometry.in_domain: Theta = I L / porosity^1.5 <= 0.03 A/m, loading <= 6 mg S/cm2) are not simulated.

    # 1. cut-off times (sharded; on Grace one array task per shard, ~1 h each at 1 core):
    python scripts/lispan_geo_design.py samples --shard 0 --nshards 40
    # 2. merge the shards, fit the window, write the design JSON and the evaluation cases:
    python scripts/lispan_geo_design.py fit --out configs/lispan_geo_design_v5.json
    # 3. evaluation references (40/20 cells; sharded too):
    python scripts/lispan_geo_design.py refs --out configs/lispan_geo_design_v5.json --shard 0 --nshards 8

Window: ln(r t_cut) = P6(ln r, theta_hat) + C_d(ln r, theta_hat, gamma_hat), where P6 is the degree-6 window of the v3
chemistry design (glass-fibre reference cell) kept as it is and C_d a degree-d polynomial correction fitted here
(gamma_hat = normalized ln L_cat and f); d is chosen by the error on held-out points.
"""

import argparse
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from dfn_pinn.lispan import LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.chem_pinn import CHEM_SPECS, apply_chem, check_spec, normalize, window_features  # noqa: E402
from dfn_pinn.lispan.geometry import GEO_INPUTS_V5, in_domain, mass_ratio, polyolefin_cell, transport_load  # noqa: E402
from lispan_chem_design import base_params, lhs, solve_retry  # noqa: E402

SAMPLES_DIR = ROOT / "results" / "lispan" / "geo_design"
RATES = [0.08, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
EVAL_GEOMS = {"g1": (30.0, 2.0), "g2": (60.0, 4.0), "g3": (100.0, 1.5), "g4": (140.0, 3.0)}
EVAL_RATES = [0.1, 0.5, 2.0]


def geo_box():
    return [dict(s) for s in GEO_INPUTS_V5]


def conditions(n_lhs, seed):
    """Chemistry x geometry points in the box (unit-cube coordinates mapped): corners of the 7-d box, the centre and a
    Latin hypercube."""
    spec = check_spec(CHEM_SPECS["v2"])
    box = spec + geo_box()
    k = len(box)
    rng = np.random.default_rng(seed)
    corners = np.array([[(i >> j) & 1 for j in range(k)] for i in range(2 ** k)], float)
    U = np.vstack([corners, np.full((1, k), 0.5), lhs(n_lhs, k, rng)])
    lo = np.array([s["lo"] for s in box]); hi = np.array([s["hi"] for s in box])
    return spec, lo + (hi - lo) * U


def cutoff_job(job):
    chem, lnL, f, rate = job
    spec = check_spec(CHEM_SPECS["v2"])
    L = math.exp(lnL)
    p = polyolefin_cell(apply_chem(base_params(0.0), spec, chem), L, f)
    prot = LiSPANProtocol(current=10.0 * rate * mass_ratio(L, f), ramp_s=30.0)
    t0 = time.perf_counter()
    out = {"rate": rate, "chem": chem, "lnL": lnL, "f": f}
    try:
        res, att = solve_retry(p, prot, 20, 10, 200, timeout=300)
        V = np.asarray(res["V"])
        c = np.asarray(res["c_PF6"])
        out.update(t_cut=float(res["t"][-1]), ok=bool(res["solver"]["success"]) and bool(V[-1] <= prot.V_min + 1e-3),
                   Q_end=float(res["Q_mAh_gS"][-1]), c_min=float(c.min()), c_max=float(c.max()), attempt=att,
                   message=res["solver"]["message"])
    except Exception as exc:                                     # noqa: BLE001 - record and go on
        out.update(t_cut=float("nan"), ok=False, message=repr(exc))
    out["wall_s"] = time.perf_counter() - t0
    return out


def all_jobs(n_lhs, seed):
    from dfn_pinn.lispan.params import LiSPANParams
    p0 = LiSPANParams()
    spec, pts = conditions(n_lhs, seed)
    k = len(spec)
    jobs = []
    for x in pts:
        chem, lnL, f = [float(v) for v in x[:k]], float(x[k]), float(x[k + 1])
        for r in RATES:
            if in_domain(p0, r, math.exp(lnL), f):
                jobs.append((chem, lnL, f, r))
    return jobs


def cmd_samples(a):
    jobs = all_jobs(a.n_lhs, a.seed)
    mine = jobs[a.shard::a.nshards]
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / f"samples_{a.shard:03d}_of_{a.nshards:03d}.jsonl"
    key = lambda d: (round(d[0] if isinstance(d, tuple) else d["rate"], 9),)        # noqa: E731
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            d = json.loads(line)
            done.add((round(d["rate"], 9), tuple(round(v, 9) for v in d["chem"]), round(d["lnL"], 9), round(d["f"], 9)))
    todo = [j for j in mine if (round(j[3], 9), tuple(round(v, 9) for v in j[0]), round(j[1], 9), round(j[2], 9)) not in done]
    print(f"{len(jobs)} FV discharges in the design; shard {a.shard}/{a.nshards}: {len(mine)} ({len(mine) - len(todo)} "
          f"cached), {a.workers} workers", flush=True)
    del key
    t0 = time.perf_counter()
    with Pool(a.workers) as pool, open(path, "a", encoding="utf-8") as fh:
        for i, res in enumerate(pool.imap_unordered(cutoff_job, todo)):
            fh.write(json.dumps(res) + "\n"); fh.flush()
            if (i + 1) % 20 == 0 or not res["ok"]:
                print(f"  {i + 1}/{len(todo)} {'' if res['ok'] else 'FAILED '}r {res['rate']:g} L {math.exp(res['lnL']):.0f} "
                      f"f {res['f']:.2f} t_cut {res['t_cut']:.0f} s ({res['wall_s']:.1f} s, {time.perf_counter() - t0:.0f} s)",
                      flush=True)
    print("done", path, flush=True)


def load_samples():
    out = []
    for path in sorted(SAMPLES_DIR.glob("samples_*.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def features(spec, samples):
    th = normalize(spec, [s["chem"] for s in samples])
    g = normalize(geo_box(), [[s["lnL"], s["f"]] for s in samples])
    r = np.array([s["rate"] for s in samples])
    return r, th, g


def base_window():
    d = json.loads((ROOT / "configs" / "lispan_chem_design_v3.json").read_text())
    return d["window"]["coef"], d["window"]["degree"]


def window_predict(coef_base, deg_base, coef_corr, deg_corr, r, th, g):
    x = np.log(r)
    y = window_features(x, th, deg_base) @ np.asarray(coef_base)
    if coef_corr is not None:
        y = y + window_features(x, np.column_stack([th, g]), deg_corr) @ np.asarray(coef_corr)
    return np.exp(y) / r


def cmd_fit(a):
    spec = check_spec(CHEM_SPECS["v2"])
    samples = load_samples()
    good = [s for s in samples if s.get("ok")]
    print(f"{len(samples)} samples, {len(good)} reached the cut-off", flush=True)
    r, th, g = features(spec, good)
    tc = np.array([s["t_cut"] for s in good])
    cb, db = base_window()
    y = np.log(r * tc) - window_features(np.log(r), th, db) @ np.asarray(cb)
    best = None
    rng = np.random.default_rng(7)
    for deg in a.degrees:
        A = window_features(np.log(r), np.column_stack([th, g]), deg)
        if A.shape[1] >= 0.5 * len(good):
            print(f"  degree {deg}: {A.shape[1]} coefficients for {len(good)} points - skipped")
            continue
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rel = np.exp(A @ coef - y) - 1.0
        holds = []
        for rep in range(5):
            idx = rng.permutation(len(good)); nt = len(good) // 5
            te, tr = idx[:nt], idx[nt:]
            c_tr, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
            holds.append(float(np.max(np.abs(np.exp(A[te] @ c_tr - y[te]) - 1.0))))
        row = {"degree": deg, "n_coef": A.shape[1], "max_rel_err": float(np.max(np.abs(rel))),
               "rms_rel_err": float(np.sqrt(np.mean(rel ** 2))), "holdout_max_rel_err": max(holds), "coef": coef.tolist()}
        print(f"  correction degree {deg} ({A.shape[1]} coef): max {100 * row['max_rel_err']:.2f} %, rms "
              f"{100 * row['rms_rel_err']:.2f} %, held-out max {100 * row['holdout_max_rel_err']:.2f} %", flush=True)
        if best is None or row["holdout_max_rel_err"] < best["holdout_max_rel_err"]:
            best = row
    base_only = np.exp(-y) - 1.0
    print(f"  base window alone (v3, glass-fibre cell): max {100 * np.max(np.abs(base_only)):.1f} %", flush=True)
    frac = 0.99 / (1 + max(best["max_rel_err"], best["holdout_max_rel_err"]))
    cases = eval_cases(spec, a.out)
    design = {"spec": spec, "geo_spec": geo_box(), "Z_CC": 0.0, "reversible": [True, False, False], "window_rates": RATES,
              "rate_min": min(RATES), "rate_max": max(RATES), "separator": "polyolefin (geometry.SEP_POLYOLEFIN)",
              "domain": {"theta_max_A_m": 0.03, "loading_max_mg_cm2": 6.0},
              "window": {"coef_base": cb, "degree_base": db, "coef": best["coef"], "degree": best["degree"],
                         "max_rel_err": best["max_rel_err"], "rms_rel_err": best["rms_rel_err"],
                         "holdout_max_rel_err": best["holdout_max_rel_err"], "n_samples": len(good),
                         "fraction_1pct_clear": frac},
              "n_samples_failed": len(samples) - len(good), "eval_cases": cases}
    Path(a.out).write_text(json.dumps(design, indent=1))
    print(f"written {a.out}: window fraction keeping 1 % clear {frac:.3f}; {len(cases)} evaluation cases", flush=True)


def eval_cases(spec, out):
    """Evaluation chemistries of the v3 design (p0 nominal + p1..p9) on the EVAL_GEOMS at EVAL_RATES, inside the domain;
    p0-p4 on g1/g3 are 'select' cases (best.pt), the rest test cases."""
    from dfn_pinn.lispan.params import LiSPANParams
    p0 = LiSPANParams()
    v3 = json.loads((ROOT / "configs" / "lispan_chem_design_v3.json").read_text())
    chems = {}
    for c in v3["eval_cases"]:
        chems.setdefault(c["id"].split("_")[0], c["chem"])
    pick = ["p0", "p1", "p2", "p3", "p5", "p6", "p7", "litHao"]
    name = Path(out).stem
    cases = []
    for cid in pick:
        if cid not in chems:
            continue
        for gname, (L, f) in EVAL_GEOMS.items():
            for r in EVAL_RATES:
                if not in_domain(p0, r, L, f):
                    continue
                case = f"{cid}-{gname}_{r:g}C"
                cases.append({"id": case, "rate": r, "chem": chems[cid], "geo": {"lnL_cat": math.log(L), "f": f},
                              "L_cat_um": L, "theta_A_m": transport_load(p0, r, L, f),
                              "select": cid in ("p0", "p1", "p2", "p3") and gname in ("g1", "g3"),
                              "ref": f"data/lispan_geo_refs/{name}/{case}.npz"})
    return cases


def ref_job(job):
    case, spec = job
    path = ROOT / case["ref"]
    if path.exists():
        return case["id"], "cached", 0.0
    L, f = math.exp(case["geo"]["lnL_cat"]), case["geo"]["f"]
    p = polyolefin_cell(apply_chem(base_params(0.0), spec, [case["chem"].get(s["name"], 0.0) for s in spec]), L, f)
    prot = LiSPANProtocol(current=10.0 * case["rate"] * mass_ratio(L, f), ramp_s=30.0)
    t0 = time.perf_counter()
    res, att = solve_retry(p, prot, 40, 20, 800, timeout=900)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")
    meta = {"purpose": "geo PINN evaluation reference (c_DL -> 0, Z_CC 0, polyolefin separator)", **case}
    np.savez_compressed(tmp, t=np.asarray(res["t"]), V=np.asarray(res["V"]), Q=np.asarray(res["Q_mAh_gS"]), meta=json.dumps(meta))
    tmp.replace(path)
    return case["id"], res["solver"]["message"], time.perf_counter() - t0


def cmd_refs(a):
    design = json.loads(Path(a.out).read_text())
    jobs = [(c, design["spec"]) for c in design["eval_cases"]][a.shard::a.nshards]
    with Pool(a.workers) as pool:
        for cid, msg, wall in pool.imap_unordered(ref_job, jobs):
            print(f"  ref {cid}: {msg} ({wall:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["samples", "fit", "refs", "count"])
    ap.add_argument("--out", default=str(ROOT / "configs" / "lispan_geo_design_v5.json"))
    ap.add_argument("--n-lhs", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--degrees", type=int, nargs="*", default=[2, 3, 4])
    a = ap.parse_args()
    if a.cmd == "count":
        jobs = all_jobs(a.n_lhs, a.seed)
        print(f"{len(jobs)} FV discharges ({len({(tuple(j[0]), j[1], j[2]) for j in jobs})} conditions); per rate: "
              + ", ".join(f"{r:g} C {sum(j[3] == r for j in jobs)}" for r in RATES))
        print(f"evaluation cases: {len(eval_cases(check_spec(CHEM_SPECS['v2']), a.out))}")
        return
    {"samples": cmd_samples, "fit": cmd_fit, "refs": cmd_refs}[a.cmd](a)


if __name__ == "__main__":
    main()
