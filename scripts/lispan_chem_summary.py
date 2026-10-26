"""Summarize chemistry-conditioned Li-SPAN PINN runs (scripts/lispan_chem_train.py) and check the Phase 3b gate
(docs/PHASE3B_CHEM_PINN.md, "Acceptance").  Standard library only.

    python scripts/lispan_chem_summary.py                                   # results/lispan_chem_runs/*
    python scripts/lispan_chem_summary.py results/lispan_chem_runs/chem_v1_* --json summary_chem_v1.json

Per run: status, steps, wall hours, rollbacks, skipped batches, the select / test means and the worst case of final.pt
(final_metrics.json, else the last evaluation) and of best.pt (best_metrics.json); per case the final errors.
Strict gate (v1 proposal):
  1. test cases: mean < 2 mV and every case < 5 mV;
  2. test mean <= 1.5 x select mean;
  3. at least two runs pass 1-2 (on final.pt or best.pt).
Revised gate (docs/PHASE3B_CHEM_PINN.md, after v1; needs the cliff-aware metrics that evaluate_chem writes since v2):
  1. test cases OUTSIDE the voltage cliffs: mean < 2 mV, every case < 5 mV;
  2. cliff positions within 1 % of the capacity (every case);
  3. outside-cliff test mean <= 1.5 x outside-cliff select mean;
  4. at least two runs pass 1-3.
The parameter-recovery part of the revised gate is scripts/lispan_chem_recovery.py.
"""

import argparse
import glob
import json
import re
from pathlib import Path

GATE_MEAN, GATE_CASE, GATE_RATIO = 2.0, 5.0, 1.5


GATE_SHIFT = 1.0          # % of the capacity


def metrics_of(m, select_ids):
    cases = {k.split("@", 1)[1]: v for k, v in m.items() if k.startswith("V_rmse_mV@")}
    test = {c: v for c, v in cases.items() if c not in select_ids}
    sel = {c: v for c, v in cases.items() if c in select_ids}
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")      # noqa: E731
    out = {"select_mean": mean(list(sel.values())), "test_mean": mean(list(test.values())),
           "test_worst": max(test.values()) if test else float("nan"), "worst": max(cases.values()) if cases else float("nan"),
           "cases": cases}
    out["pass"] = bool(test) and out["test_mean"] < GATE_MEAN and all(v < GATE_CASE for v in test.values()) \
        and out["test_mean"] <= GATE_RATIO * out["select_mean"]
    co = {k.split("@", 1)[1]: v for k, v in m.items() if k.startswith("V_rmse_out_mV@")}
    sh = {k.split("@", 1)[1]: v for k, v in m.items() if k.startswith("cliff_shift_pct@")}
    if co:
        t_o = [v for c, v in co.items() if c not in select_ids]; s_o = [v for c, v in co.items() if c in select_ids]
        out.update(out_select=mean(s_o), out_test=mean(t_o), out_test_worst=max(t_o) if t_o else float("nan"),
                   shift_max=max(sh.values()) if sh else 0.0, cases_out=co, shifts=sh)
        out["pass_revised"] = bool(t_o) and out["out_test"] < GATE_MEAN and all(v < GATE_CASE for v in t_o) \
            and out["shift_max"] < GATE_SHIFT and out["out_test"] <= GATE_RATIO * out["out_select"]
    else:
        out["pass_revised"] = None
    return out


def load_run(d: Path, root: Path):
    run = {"name": d.name}
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    design_file = cfg.get("design_file") or cfg.get("train", {}).get("design_file", "")
    select_ids = set()
    if design_file:
        dp = Path(design_file)
        dp = dp if dp.is_absolute() else root / dp
        if dp.exists():
            select_ids = {c["id"] for c in json.loads(dp.read_text())["eval_cases"] if c["select"]}
    run["status"] = "finished" if (d / "final.pt").exists() else ("stopped" if (d / "latest.pt").exists() else "no checkpoint")
    hist = json.loads((d / "history.json").read_text()) if (d / "history.json").exists() else {"history": [], "evals": []}
    fm = json.loads((d / "final_metrics.json").read_text()) if (d / "final_metrics.json").exists() else \
        (hist["evals"][-1] if hist.get("evals") else {})
    run["step"] = fm.get("step", hist["history"][-1]["step"] if hist.get("history") else None)
    run["wall_h"] = fm.get("time_s", hist["history"][-1]["time_s"] if hist.get("history") else float("nan")) / 3600
    run["final"] = metrics_of(fm, select_ids)
    bm = json.loads((d / "best_metrics.json").read_text()) if (d / "best_metrics.json").exists() else None
    run["best"] = metrics_of(bm, select_ids) if bm else None
    if bm:
        run["best"]["step"] = bm.get("step")
    log = (d / "train.log").read_text(errors="replace") if (d / "train.log").exists() else ""
    m = re.findall(r"finished: (\d+) batches skipped", log)
    run["skipped"] = int(m[-1]) if m else len(re.findall(r"non-finite", log))
    run["rollbacks"] = [int(s) for s in re.findall(r"step (\d+): ROLLBACK", log)]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--json")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = a.runs or sorted(glob.glob(str(root / "results" / "lispan_chem_runs" / "*")))
    runs = [load_run(Path(p), root) for p in paths if Path(p).is_dir()]
    if not runs:
        raise SystemExit("no run folders found")
    print("Voltage rms error vs FV [mV] (select / test cases)")
    print("| Run | status | steps | wall h | final select | final test | final worst | best.pt (step) select / test / worst "
          "| rollbacks | skipped | gate |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    n_pass = 0
    for r in runs:
        f, b = r["final"], r["best"]
        ok = f["pass"] or (b is not None and b["pass"])
        n_pass += ok
        bs = "-" if b is None else f"{b['select_mean']:.2f} / {b['test_mean']:.2f} / {b['worst']:.2f} ({b['step']})"
        print(f"| {r['name']} | {r['status']} | {r['step']} | {r['wall_h']:.1f} | {f['select_mean']:.2f} | "
              f"{f['test_mean']:.2f} | {f['worst']:.2f} | {bs} | {len(r['rollbacks'])} | {r['skipped']} | "
              f"{'PASS' if ok else 'FAIL'} |")
    print(f"\nStrict gate 3 (multi-start): {n_pass} of {len(runs)} runs pass 1-2 -> {'PASS' if n_pass >= 2 else 'FAIL'}")
    rev = [r for r in runs if r["final"]["pass_revised"] is not None]
    if rev:
        print("\nRevised gate (outside the voltage cliffs; cliff positions) [mV, %]")
        print("| Run | final out select | out test | out worst test | cliff shift max | best.pt out select / test / shift | revised |")
        print("|---|---|---|---|---|---|---|")
        n_rev = 0
        for r in rev:
            f, b = r["final"], r["best"]
            ok = bool(f["pass_revised"]) or bool(b and b.get("pass_revised"))
            n_rev += ok
            bs = "-" if not b or b.get("pass_revised") is None else \
                f"{b['out_select']:.2f} / {b['out_test']:.2f} / {b['shift_max']:.2f} ({b['step']})"
            print(f"| {r['name']} | {f['out_select']:.2f} | {f['out_test']:.2f} | {f['out_test_worst']:.2f} | "
                  f"{f['shift_max']:.2f} | {bs} | {'PASS' if ok else 'FAIL'} |")
        print(f"Revised gate 4 (multi-start): {n_rev} of {len(rev)} runs pass 1-3 -> {'PASS' if n_rev >= 2 else 'FAIL'}")
    for r in runs:
        f = r["final"]
        print(f"\n{r['name']} final per case: " + ", ".join(
            f"{c} {v:.2f}" + (f" (out {f['cases_out'][c]:.2f}, shift {f['shifts'].get(c, 0):.2f} %)" if f.get("cases_out") else "")
            for c, v in sorted(f["cases"].items())))
    if a.json:
        Path(a.json).write_text(json.dumps(runs, indent=1))
        print("\nwritten", a.json)


if __name__ == "__main__":
    main()
