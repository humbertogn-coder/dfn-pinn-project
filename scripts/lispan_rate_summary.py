"""Summarize rate-parametric Li-SPAN PINN runs (scripts/lispan_rate_train.py) and check the Phase 3a gate
(docs/PHASE3_PARAMETRIC_PINN.md, "Acceptance").  Standard library only, so it also runs with the system python.

    python scripts/lispan_rate_summary.py                                   # results/lispan_rate_runs/rate_v2_*
    python scripts/lispan_rate_summary.py results/lispan_rate_runs/rate_w96_* --json summary_wave1.json

Per run: status (finished = final.pt), last step, wall hours, voltage rms error vs FV at every evaluation rate (from
final_metrics.json, else the last evaluation in history.json), V max, c_e max at the highest rate, skipped batches and
rollbacks (counted in train.log, with their steps).  Then the gate:
  1. V rms < 2 mV at every evaluation rate;
  2. no error growth at the non-anchor rates: mean over non-anchor rates <= 1.5 x mean over the anchors (window_rates);
  3. seed spread below the error: runs whose configs differ only in the seed form a group; at every rate the standard
     deviation over the group must stay below the group mean.
"""

import argparse
import glob
import json
import math
import re
from pathlib import Path

IGNORED_KEYS = {"seed", "threads", "device", "stop_step", "eval_every", "checkpoint_every", "latest_every",
                "log_every", "history_every"}
GATE_V_MV = 2.0
GATE_GROWTH = 1.5


def fmt_rate(r):
    return ("%g" % r)


def load_run(d: Path):
    run = {"name": d.name, "dir": str(d)}
    cfg_path = d / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    train = cfg.get("train", {})
    run["train"] = train
    run["eval_rates"] = [float(r) for r in train.get("eval_rates", [])]
    run["anchors"] = [float(r) for r in train.get("window_rates", [])]
    run["status"] = "finished" if (d / "final.pt").exists() else ("stopped" if (d / "latest.pt").exists() else "no checkpoint")
    hist_path = d / "history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else {"history": [], "evals": []}
    if hist.get("history"):
        run["step"] = hist["history"][-1]["step"]
        run["wall_h"] = hist["history"][-1].get("time_s", float("nan")) / 3600
    fm_path = d / "final_metrics.json"
    if fm_path.exists():
        metrics, run["metrics_from"] = json.loads(fm_path.read_text()), "final_metrics.json"
    elif hist.get("evals"):
        metrics, run["metrics_from"] = hist["evals"][-1], "last eval (step %s)" % hist["evals"][-1].get("step", "?")
    else:
        metrics, run["metrics_from"] = {}, "none"
    run["metrics"] = metrics
    if "step" in metrics:
        run["step"] = metrics["step"]
    if "time_s" in metrics:
        run["wall_h"] = metrics["time_s"] / 3600
    run["V"] = {}
    for r in run["eval_rates"]:
        v = metrics.get("V_rmse_mV@" + fmt_rate(r))
        if v is not None:
            run["V"][r] = float(v)
    rmax = max(run["eval_rates"]) if run["eval_rates"] else None
    run["ce_max_top"] = metrics.get("c_e_max@" + fmt_rate(rmax)) if rmax is not None else None
    run["V_max_worst"] = metrics.get("V_max_worst_mV")
    log_path = d / "train.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    m = re.findall(r"finished: (\d+) batches skipped", log)
    run["skipped"] = int(m[-1]) if m else len(re.findall(r"non-finite", log))
    run["rollback_steps"] = [int(s) for s in re.findall(r"step (\d+): ROLLBACK", log)]
    run["resumes"] = len(re.findall(r"resum", log, flags=re.I))
    return run


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def gate(run):
    v = run["V"]
    if not v or len(v) < len(run["eval_rates"]):
        return {"c1": None, "c2": None, "growth": float("nan")}
    anchors = [r for r in v if any(abs(r - a) < 1e-9 for a in run["anchors"])]
    others = [r for r in v if r not in anchors]
    growth = mean(v[r] for r in others) / mean(v[r] for r in anchors) if anchors and others else float("nan")
    return {"c1": all(x < GATE_V_MV for x in v.values()), "c2": bool(growth <= GATE_GROWTH), "growth": growth}


def seed_groups(runs):
    groups = {}
    for run in runs:
        key = json.dumps({k: v for k, v in sorted(run["train"].items()) if k not in IGNORED_KEYS}, sort_keys=True)
        groups.setdefault(key, []).append(run)
    return [g for g in groups.values() if len(g) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run folders (default: results/lispan_rate_runs/rate_v2_*)")
    ap.add_argument("--json", help="also write the summary to this JSON file")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = a.runs or sorted(glob.glob(str(root / "results" / "lispan_rate_runs" / "rate_v2_*")))
    runs = [load_run(Path(p)) for p in paths if Path(p).is_dir()]
    if not runs:
        raise SystemExit("no run folders found")

    rates = sorted({r for run in runs for r in run["eval_rates"]})
    head = ["Run", "status", "steps", "wall h"] + [fmt_rate(r) + " C" for r in rates] + \
           ["mean", "worst", "V max worst", "c_e max @top", "skipped", "rollbacks"]
    print("Voltage rms error vs FV [mV]")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for run in runs:
        v = run["V"]
        cells = [run["name"], run["status"], str(run.get("step", "-")), "%.1f" % run.get("wall_h", float("nan"))]
        cells += ["%.2f" % v[r] if r in v else "-" for r in rates]
        cells += ["%.2f" % mean(v.values()) if v else "-", "%.2f" % max(v.values()) if v else "-",
                  "%.1f" % run["V_max_worst"] if run["V_max_worst"] is not None else "-",
                  "%.1f" % run["ce_max_top"] if run["ce_max_top"] is not None else "-",
                  str(run["skipped"]), str(len(run["rollback_steps"]))]
        print("| " + " | ".join(cells) + " |")

    print("\nDetails")
    for run in runs:
        g = run["gate"] = gate(run)
        rb = run["rollback_steps"]
        print("- %s: metrics from %s; rollbacks at steps %s; anchors %s; gate 1 (< %g mV) %s, gate 2 (non-anchor/anchor "
              "= %.2f, limit %g) %s" % (run["name"], run["metrics_from"], rb if rb else "none",
                                        [fmt_rate(r) for r in run["anchors"]], GATE_V_MV,
                                        {True: "PASS", False: "FAIL", None: "n/a"}[g["c1"]], g["growth"], GATE_GROWTH,
                                        {True: "PASS", False: "FAIL", None: "n/a"}[g["c2"]]))

    print("\nGate 3 (seed spread: std over seeds < mean over seeds at every rate)")
    groups = seed_groups(runs)
    summary_groups = []
    if not groups:
        print("- no group of runs that differ only in the seed")
    for grp in groups:
        names = [r["name"] for r in grp]
        common = sorted(set.intersection(*[set(r["V"]) for r in grp]))
        rows, ok = [], True
        for r in common:
            vals = [run["V"][r] for run in grp]
            m, s = mean(vals), std(vals)
            ok = ok and s < m
            rows.append({"rate": r, "mean": m, "std": s, "min": min(vals), "max": max(vals)})
        means = [mean(run["V"].values()) for run in grp]
        print("- %s: %s; mean V rms per seed %s mV; worst std/mean %.2f" % (
            ", ".join(names), "PASS" if ok and common else "FAIL",
            ", ".join("%.2f" % x for x in means), max(x["std"] / x["mean"] for x in rows) if rows else float("nan")))
        for x in rows:
            print("    %5s C: mean %.2f  std %.2f  range %.2f-%.2f mV" % (fmt_rate(x["rate"]), x["mean"], x["std"],
                                                                         x["min"], x["max"]))
        summary_groups.append({"runs": names, "pass": bool(ok and common), "per_rate": rows})

    if a.json:
        out = [{k: v for k, v in run.items() if k not in ("train", "metrics")} for run in runs]
        for o in out:
            o["V"] = {fmt_rate(r): x for r, x in o["V"].items()}
        Path(a.json).write_text(json.dumps({"runs": out, "seed_groups": summary_groups}, indent=1))
        print("\nwritten", a.json)


if __name__ == "__main__":
    main()
