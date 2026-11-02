"""Progress of PINN training runs (rate / chemistry networks) from their run folders: step reached, speed, time left,
latest evaluation and whether the run looks alive.  Reads only config.json and train.log, so it is cheap on a login
node and safe while the jobs are running.

    python scripts/lispan_run_progress.py                                   # results/lispan_chem_runs/chem_v4_*
    python scripts/lispan_run_progress.py results/lispan_chem_runs/chem_v3_*
    python scripts/lispan_run_progress.py --stale-min 30                    # flag runs whose log is older than 30 min

Columns: step / total, percent, seconds per step (last ~2000 logged steps, wall clock of the training loop), time
left, expected end (local time), minutes since the log was last written, rollbacks and skipped batches so far,
and the latest evaluation line (select / test / worst, and the same outside the voltage cliffs) for chemistry runs.
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import time
from pathlib import Path

STEP = re.compile(r"^step\s+(\d+)\s.*\|\s*([\d.]+)s\s*$")
EVAL = re.compile(r"eval V rmse \[mV\]:\s*(.*)$")
OUT = re.compile(r"outside cliffs select ([\d.e+-]+) test ([\d.e+-]+) \(worst test ([\d.e+-]+)\)")
SEL = re.compile(r"select ([\d.e+-]+) test ([\d.e+-]+) \| mean ([\d.e+-]+) worst ([\d.e+-]+)")


def total_steps(run):
    try:
        cfg = json.loads((run / "config.json").read_text())
        t = cfg.get("train", cfg)
        return int(t.get("adam_steps") or t.get("steps") or 0)
    except (OSError, ValueError):
        return 0


def scan(run, window=2000):
    log = run / "train.log"
    steps, evals, rollbacks, finished = [], [], 0, None
    if not log.exists():
        return None
    with open(log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP.match(line.strip())
            if m:
                steps.append((int(m.group(1)), float(m.group(2))))
                continue
            m = EVAL.search(line)
            if m:
                evals.append((steps[-1][0] if steps else 0, m.group(1).strip()))
                continue
            if "ROLLBACK #" in line:
                rollbacks += 1
            if line.startswith("finished:"):
                finished = line.strip()
    if not steps:
        return {"step": 0, "spd": None, "evals": evals, "rollbacks": rollbacks, "finished": finished,
                "age_min": (time.time() - log.stat().st_mtime) / 60}
    last_step, last_t = steps[-1]
    # time_s restarts at a resume: use only the stretch after the last drop of the clock
    seg = [steps[-1]]
    for s, t in reversed(steps[:-1]):
        if t > seg[-1][1] or s >= seg[-1][0]:
            break
        seg.append((s, t))
        if last_step - s >= window:
            break
    s0, t0 = seg[-1]
    spd = (last_t - t0) / (last_step - s0) if last_step > s0 and last_t > t0 else None
    return {"step": last_step, "spd": spd, "evals": evals, "rollbacks": rollbacks, "finished": finished,
            "age_min": (time.time() - log.stat().st_mtime) / 60}


def short_eval(text):
    m1, m2 = SEL.search(text), OUT.search(text)
    parts = []
    if m1:
        parts.append(f"all {m1.group(1)}/{m1.group(2)} worst {m1.group(4)}")
    if m2:
        parts.append(f"out-cliff {m2.group(1)}/{m2.group(2)} worst {m2.group(3)}")
    return "; ".join(parts) or text[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=["results/lispan_chem_runs/chem_v4_*"])
    ap.add_argument("--stale-min", type=float, default=60.0, help="flag a running job whose log is older than this")
    ap.add_argument("--evals", type=int, default=1, help="how many of the latest evaluation lines to show per run")
    a = ap.parse_args()
    runs = sorted({Path(p) for pat in a.runs for p in glob.glob(pat) if os.path.isdir(p)})
    if not runs:
        print("no run folders match", a.runs)
        return
    now = dt.datetime.now()
    print(f"{'run':24s} {'step':>15s} {'%':>5s} {'s/step':>7s} {'left':>8s} {'expected end':>16s} {'log age':>8s} "
          f"{'rb':>3s}  status / latest eval [mV, select/test]")
    for run in runs:
        tot = total_steps(run)
        r = scan(run)
        if r is None:
            print(f"{run.name:24s} no train.log yet")
            continue
        done = (run / "final.pt").exists() or r["finished"] is not None
        pct = 100.0 * r["step"] / tot if tot else float("nan")
        if done:
            left, end, status = "-", "-", "FINISHED"
        elif r["spd"]:
            sec = (tot - r["step"]) * r["spd"]
            left = f"{sec / 3600:6.1f} h"
            end = (now + dt.timedelta(seconds=sec)).strftime("%m-%d %H:%M")
            status = "STALE?" if r["age_min"] > a.stale_min else "running"
        else:
            left, end, status = "?", "?", "starting"
        spd = f"{r['spd']:.3f}" if r["spd"] else "-"
        ev = short_eval(r["evals"][-1][1]) if r["evals"] else "no eval yet"
        print(f"{run.name:24s} {r['step']:>7d}/{tot:<7d} {pct:5.1f} {spd:>7s} {left:>8s} {end:>16s} "
              f"{r['age_min']:6.0f} m {r['rollbacks']:>3d}  {status}: {ev}")
        for s, text in r["evals"][-a.evals:-1] if a.evals > 1 else []:
            print(f"{'':24s}   step {s:>7d}: {short_eval(text)}")
        if r["finished"]:
            print(f"{'':24s}   {r['finished']}")


if __name__ == "__main__":
    main()
