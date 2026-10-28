"""Window design and evaluation references for the chemistry-conditioned Li-SPAN PINN (Phase 3b, chem_pinn.py).

    python scripts/lispan_chem_design.py --out configs/lispan_chem_design_v1.json --workers 2
    python scripts/lispan_chem_design.py --check-invariances          # FV check of the analytic parameters only
    python scripts/lispan_chem_design.py --refs-only --out configs/lispan_chem_design_v1.json   # (re)build the refs

1. Cut-off times of the finite-volume solver at the window rates for a design of chemistries: the 2^k corners of the
   box, its centre and a Latin hypercube; ln(r t_cut) is fitted by a full quadratic in (ln r, theta_hat)
   (chem_pinn.fit_chem_window), with the error on 20 % of the points held out reported too.
2. Evaluation cases: the nominal chemistry plus Latin-hypercube points (another seed), each at the evaluation rates;
   the first half are 'select' cases (best.pt is chosen on them), the rest are test cases.  FV references (40/20
   cells, c_DL -> 0, at the network's Z_CC; voltage curve only, ~15 kB) are written to
   data/lispan_chem_refs/<design>/<case>.npz.
The JSON (window coefficients, samples, cases) and the light references are small and belong in the repository; a
missing reference is rebuilt by scripts/lispan_chem_train.py.
"""

import argparse
from dataclasses import replace
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from dfn_pinn.lispan import LiSPANParams, LiSPANProtocol, solve  # noqa: E402
from dfn_pinn.lispan.chem_pinn import CHEM_SPECS, apply_chem, check_spec, fit_chem_window, normalize  # noqa: E402


def base_params(Z_CC, reversible=(True, False, False)):
    return replace(LiSPANParams(), c_DL=1e-6, Z_CC=Z_CC, reversible=tuple(reversible))


class SolveTimeout(Exception):
    pass


def _alarm(*_):
    raise SolveTimeout()


# FV attempts in order: a few discharges stall at the default tolerance (seen 2026-10-27: the litHao chemistry at 1.5 C
# hangs at rtol 1e-6 on 20/10 cells, but solves in 6 s at rtol 1e-5 with the same cut-off time to 0.01 %)
ATTEMPTS = ({}, {"rtol": 1e-5}, {"rtol": 1e-6, "dN": (4, 2)})


def solve_retry(p, prot, N_c, N_s, n_out, timeout=120):
    """solve() with a time limit per attempt and the fallbacks of ATTEMPTS; returns (res, attempt index)."""
    last = None
    for k, att in enumerate(ATTEMPTS):
        dn = att.get("dN", (0, 0))
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
        try:
            _, res = solve(p, prot, N_c=N_c + dn[0], N_s=N_s + dn[1], n_out=n_out, rtol=att.get("rtol", 1e-6))
            return res, k
        except Exception as exc:                               # noqa: BLE001 - try the next attempt
            last = exc
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    raise RuntimeError(f"all FV attempts failed (last: {last!r})")


def cutoff_case(job):
    """One FV discharge -> cut-off time (coarse grid: the cut-off time is grid-insensitive, 2838 s vs 2838 s at 1 C)."""
    spec, values, rate, Z_CC = job
    p = apply_chem(base_params(Z_CC), spec, values)
    prot = LiSPANProtocol.from_crate(rate, ramp_s=30.0)
    t0 = time.perf_counter()
    try:
        res, att = solve_retry(p, prot, 20, 10, 200)
        V = np.asarray(res["V"])
        ok = bool(res["solver"]["success"]) and bool(V[-1] <= prot.V_min + 1e-3)
        return {"rate": rate, "chem": values, "t_cut": float(res["t"][-1]), "ok": ok, "V_end": float(V[-1]),
                "wall_s": time.perf_counter() - t0, "message": res["solver"]["message"], "attempt": att}
    except Exception as exc:                                   # noqa: BLE001 - record and go on
        return {"rate": rate, "chem": values, "t_cut": float("nan"), "ok": False, "V_end": float("nan"),
                "wall_s": time.perf_counter() - t0, "message": repr(exc)}


def reference_case(job):
    spec, values, rate, Z_CC, path = job
    path = Path(path)
    if path.exists():
        return str(path), "cached", 0.0
    p = apply_chem(base_params(Z_CC), spec, values)
    prot = LiSPANProtocol.from_crate(rate, ramp_s=30.0)
    t0 = time.perf_counter()
    res, att = solve_retry(p, prot, 40, 20, 800, timeout=600)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")         # parallel jobs may build the same reference
    meta = {"purpose": "chem PINN evaluation reference (c_DL -> 0), voltage only", "rate": rate, "Z_CC": Z_CC,
            "chem": dict(zip([s["name"] for s in spec], values)), "solver": res["solver"]["message"],
            "fv_attempt": ATTEMPTS[att]}
    np.savez_compressed(tmp, t=np.asarray(res["t"]), V=np.asarray(res["V"]), meta=json.dumps(meta))
    tmp.replace(path)
    return str(path), res["solver"]["message"], time.perf_counter() - t0


def lhs(n, k, rng):
    """Latin hypercube in [0, 1]^k."""
    u = (rng.permuted(np.tile(np.arange(n), (k, 1)), axis=1).T + rng.uniform(size=(n, k))) / n
    return u


def box_points(spec, u):
    lo = np.array([s["lo"] for s in spec]); hi = np.array([s["hi"] for s in spec])
    return lo + (hi - lo) * np.asarray(u)


def build_refs(design, workers):
    jobs = [(design["spec"], c["chem_values"], c["rate"], design["Z_CC"], str(ROOT / c["ref"])) for c in design["eval_cases"]]
    with Pool(workers) as pool:
        for path, msg, wall in pool.imap_unordered(reference_case, jobs):
            print(f"  ref {Path(path).name}: {msg} ({wall:.0f} s)", flush=True)


def check_invariances(Z_CC):
    p0 = base_params(Z_CC)
    prot = LiSPANProtocol.from_crate(1.0, ramp_s=30.0)
    _, r0 = solve(p0, prot, N_c=20, N_s=10, n_out=400)
    _, ra = solve(replace(p0, U0=tuple(u + 0.05 for u in p0.U0)), prot, N_c=20, N_s=10, n_out=400)
    _, rz = solve(replace(p0, Z_CC=Z_CC + 0.015), prot, N_c=20, N_s=10, n_out=400)
    t = r0["t"]; m = (t > 100) & (t < 0.97 * min(r0["t"][-1], ra["t"][-1], rz["t"][-1]))
    I = prot.current * np.tanh(t[m] / 30.0)
    ea = np.abs(np.interp(t[m], ra["t"], ra["V"]) - r0["V"][m] - 0.05).max()
    ez = np.abs(np.interp(t[m], rz["t"], rz["V"]) - r0["V"][m] + 0.015 * I).max()
    print(f"common U0 shift +50 mV -> V + 50 mV: max error {ea:.2e} V")
    print(f"Z_CC + 0.015 Ohm m2 -> V - 0.015 I: max error {ez:.2e} V")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "configs" / "lispan_chem_design_v1.json"))
    ap.add_argument("--spec", default="", help="JSON file with a list of {name, lo, hi} (instead of --spec-name)")
    ap.add_argument("--spec-name", default="v1", choices=sorted(CHEM_SPECS), help="built-in box (chem_pinn.CHEM_SPECS)")
    ap.add_argument("--eval-extra", default="", help="JSON file: extra evaluation chemistries [{id (no '_'), chem: "
                    "{input: value}, select: bool}], e.g. points of literature fits")
    ap.add_argument("--Z-CC", type=float, default=0.010, help="the network's (window) contact resistance [Ohm m2]")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--n-lhs", type=int, default=32)
    ap.add_argument("--eval-points", type=int, default=8)
    ap.add_argument("--eval-rates", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window-degree", type=int, default=2, help="total degree of the ln(r t_cut) polynomial")
    ap.add_argument("--levels", type=int, default=2, help="levels per input of the full factorial (2 = corners)")
    ap.add_argument("--extra-lhs", type=int, default=0, help="a second Latin hypercube (seed + 1) on top")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--refs-only", action="store_true")
    ap.add_argument("--check-invariances", action="store_true")
    a = ap.parse_args()
    if a.check_invariances:
        check_invariances(a.Z_CC)
        return
    out = Path(a.out)
    if a.refs_only:
        build_refs(json.loads(out.read_text()), a.workers)
        return
    spec = check_spec(json.loads(Path(a.spec).read_text()) if a.spec else CHEM_SPECS[a.spec_name])
    k = len(spec)
    rng = np.random.default_rng(a.seed)
    corners = list(itertools.product(list(np.linspace(0.0, 1.0, a.levels)), repeat=k))
    U = np.vstack([np.array(corners), np.full((1, k), 0.5), lhs(a.n_lhs, k, rng)]
                  + ([lhs(a.extra_lhs, k, np.random.default_rng(a.seed + 1))] if a.extra_lhs else []))
    U = np.unique(np.round(U, 12), axis=0)
    pts = box_points(spec, U)
    jobs = [(spec, [float(v) for v in th], float(r), a.Z_CC) for th in pts for r in a.rates]
    cache = out.with_suffix(".samples.jsonl")                 # every FV result as it comes (refits need no reruns)
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            d = json.loads(line)
            done[(round(d["rate"], 9), tuple(round(v, 9) for v in d["chem"]), d["Z_CC"])] = d
    key = lambda j: (round(j[2], 9), tuple(round(v, 9) for v in j[1]), j[3])       # noqa: E731
    todo = [j for j in jobs if key(j) not in done]
    print(f"{len(pts)} chemistries x {len(a.rates)} rates = {len(jobs)} FV discharges ({len(jobs) - len(todo)} cached), "
          f"{a.workers} workers", flush=True)
    t0 = time.perf_counter()
    with Pool(a.workers) as pool, open(cache, "a", encoding="utf-8") as fh:
        for i, res in enumerate(pool.imap(cutoff_case, todo)):
            res["Z_CC"] = a.Z_CC
            fh.write(json.dumps(res) + "\n"); fh.flush()
            done[(round(res["rate"], 9), tuple(round(v, 9) for v in res["chem"]), a.Z_CC)] = res
            if (i + 1) % 20 == 0 or not res["ok"]:
                print(f"  {i + 1}/{len(jobs)} {'' if res['ok'] else 'FAILED '}r {res['rate']:g} chem "
                      f"{[round(v, 3) for v in res['chem']]} t_cut {res['t_cut']:.0f} s ({res['wall_s']:.1f} s) "
                      f"{'' if res['ok'] else res['message']}", flush=True)
    samples = [done[key(j)] for j in jobs]
    good = [s for s in samples if s["ok"]]
    print(f"{len(good)}/{len(samples)} reached the cut-off; {time.perf_counter() - t0:.0f} s", flush=True)
    th_hat = normalize(spec, [s["chem"] for s in good])
    rates = [s["rate"] for s in good]; tc = [s["t_cut"] for s in good]
    deg = a.window_degree
    coef, emax, erms = fit_chem_window(rates, th_hat, tc, deg)
    from dfn_pinn.lispan.chem_pinn import window_features
    holds = []
    for rep_ in range(5):                                     # 5 random 80/20 splits
        idx = np.random.default_rng(1000 + rep_).permutation(len(good)); ntest = max(1, len(good) // 5)
        tr, te = idx[ntest:], idx[:ntest]
        coef_tr, _, _ = fit_chem_window(np.array(rates)[tr], th_hat[tr], np.array(tc)[tr], deg)
        pred = np.exp(window_features(np.log(np.array(rates)[te]), th_hat[te], deg) @ np.array(coef_tr)) / np.array(rates)[te]
        holds.append(float(np.max(np.abs(pred / np.array(tc)[te] - 1.0))))
    hold = max(holds)
    print(f"window fit (degree {deg}, {len(coef)} coefficients): max {100 * emax:.2f} %, rms {100 * erms:.2f} %, "
          f"held-out max {100 * hold:.2f} % (5 splits); window fraction keeping 1 % clear of the cut-off: "
          f"{0.99 / (1 + max(emax, hold)):.3f}")
    ev_pts = np.vstack([np.zeros((1, k)), box_points(spec, lhs(a.eval_points - 1, k, np.random.default_rng(a.seed + 101)))])
    ev_pts[0] = 0.0                                             # the nominal chemistry
    name = out.stem
    cases = []
    for j, th in enumerate(ev_pts):
        for r in a.eval_rates:
            cid = f"p{j}_{r:g}C"
            cases.append({"id": cid, "rate": float(r), "chem": {s["name"]: float(v) for s, v in zip(spec, th)},
                          "chem_values": [float(v) for v in th], "select": j < a.eval_points // 2,
                          "ref": f"data/lispan_chem_refs/{name}/{cid}.npz"})
    if a.eval_extra:
        lo = np.array([s_["lo"] for s_ in spec]); hi = np.array([s_["hi"] for s_ in spec])
        for e in json.loads(Path(a.eval_extra).read_text()):
            if "_" in e["id"]:
                raise SystemExit(f"extra evaluation id {e['id']!r}: no '_' (case ids are <chemistry>_<rate>C)")
            th = np.array([float(e["chem"].get(s_["name"], 0.0)) for s_ in spec])
            if np.any(th < lo) or np.any(th > hi):
                raise SystemExit(f"extra evaluation chemistry {e['id']} lies outside the box: {e['chem']}")
            for r in a.eval_rates:
                cid = f"{e['id']}_{r:g}C"
                cases.append({"id": cid, "rate": float(r), "chem": {s_["name"]: float(v) for s_, v in zip(spec, th)},
                              "chem_values": [float(v) for v in th], "select": bool(e.get("select", False)),
                              "ref": f"data/lispan_chem_refs/{name}/{cid}.npz", "note": e.get("note", "")})
    design = {"spec": spec, "Z_CC": a.Z_CC, "reversible": [True, False, False], "window_rates": a.rates,
              "rate_min": min(a.rates), "rate_max": max(a.rates),
              "window": {"coef": coef, "degree": deg, "max_rel_err": emax, "rms_rel_err": erms, "holdout_max_rel_err": hold,
                         "n_samples": len(good), "fraction_1pct_clear": 0.99 / (1 + max(emax, hold))},
              "samples": samples, "eval_cases": cases}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(design, indent=1))
    print("written", out, flush=True)
    build_refs(design, a.workers)


if __name__ == "__main__":
    main()
