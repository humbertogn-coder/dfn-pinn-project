"""Computational cost of the finite-volume reference and of the PINN (paper step 2).

Everything is single-threaded CPU (OMP_NUM_THREADS=1, torch.set_num_threads(1)).  Finite-volume solves are timed
here (best of --repeats, CPU seconds of this process, so a busy machine does not inflate them much); PINN training
times are the wall times logged in each run's history.json (pauses between resumes excluded; two trainings shared
the two cores of the machine, one per core).  Rows:

  A  finite-volume discharge, 0.1 C and 1 C, paper grid (20/10 volumes) and benchmark grid (40/20)
  B  PINN forward training with the final recipe from scratch (configs/lispan_forward_final_*.json, all seeds found)
  B' CPU time per training step of the same recipe, measured here (steps 11-60 of a fresh run, before and after the
     inventory residual switches on), and the implied CPU cost of the full schedule - robust to other jobs sharing
     the machine, unlike the logged wall times
  C  evaluation of a trained PINN (voltage at 1000 times; all fields on 20 x 1000 points)
  D  8-parameter inverse from 0.1 C + 1 C synthetic data (1 mV noise): classical least squares with the finite-volume
     model (scripts/lispan_fit_experiment.py --data nominal) vs the PINN route (stage-1 re-adaptation of both rate
     networks at the initial guess + multi-rate Levenberg-Marquardt run)

    python scripts/paper_cost_table.py      ->  results/paper/cost_table.json (+ .md)
"""

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataclasses import replace  # noqa: E402
from dfn_pinn.lispan.params import LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.model import solve  # noqa: E402
from dfn_pinn.lispan import runs  # noqa: E402

RUNS = ROOT / "results" / "lispan_runs"
BASE = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)
INVERSE = {"stage1": ["lispan_inv8p_stage1_0.1C_20261007T060146Z", "lispan_inv8p_stage1_1C_20261007T070125Z"],
           "multirate": "lispan_inv8p_multirate_lm_20261007T083347Z",
           "fv_fit": ROOT / "results" / "lispan" / "fit_experiment" / "fit_0.1+1C_nominal.json"}


def fv_timing(repeats):
    rows = []
    for crate in (0.1, 1.0):
        for grid in ((20, 10), (40, 20)):
            cpu, wall = [], []
            for _ in range(repeats):
                c0, w0 = time.process_time(), time.perf_counter()
                _, res = solve(BASE, LiSPANProtocol.from_crate(crate, ramp_s=30.0), N_c=grid[0], N_s=grid[1], n_out=1500)
                cpu.append(time.process_time() - c0); wall.append(time.perf_counter() - w0)
            rows.append({"crate": crate, "grid": list(grid), "cpu_s": min(cpu), "wall_s": min(wall),
                         "rhs_evals": res["solver"]["nfev"], "states": int(grid[0] * 6 + (grid[0] + grid[1]) * 2)})
            print(f"FV {crate:g} C grid {grid}: {min(cpu):.2f} s CPU", flush=True)
    return rows


def pinn_training():
    rows = []
    for cr in ("0.1", "1"):
        for d in sorted(RUNS.glob(f"lispan_final_{cr}C_seed*_*")):
            if runs.is_stopped(d) or not (d / "history.json").exists():
                continue
            h = runs.history(d)
            last = h["history"][-1]
            ev = h["evals"][-1] if h["evals"] else {}
            done = (d / "final.pt").exists()
            cfg = json.loads((d / "config.json").read_text())["train"]
            t1, s1 = runs.time_to(d, threshold=1.0)
            t05, s05 = runs.time_to(d, threshold=0.5)
            rows.append({"run": d.name, "crate": float(cr), "seed": cfg["seed"], "finished": done,
                         "steps": last["step"], "steps_planned": cfg["adam_steps"], "train_wall_h": last["time_s"] / 3600,
                         "s_per_step": last["time_s"] / last["step"], "V_rmse_mV": ev.get("V_rmse_mV"),
                         "V_max_mV": ev.get("V_max_mV"), "time_to_1mV_h": None if t1 is None else t1 / 3600,
                         "time_to_0.5mV_h": None if t05 is None else t05 / 3600})
            print(f"PINN {d.name}: {rows[-1]['train_wall_h']:.2f} h, {rows[-1]['steps']} steps, "
                  f"V rms {rows[-1]['V_rmse_mV']}", flush=True)
    return rows


def pinn_step_timing(n_steps=60, n_skip=10):
    import tempfile
    from dfn_pinn.lispan.pinn import LiSPANTrainConfig, train
    from dfn_pinn.lispan.model import load
    torch.set_num_threads(1)
    rows = []
    for cr in ("0.1", "1"):
        spec = json.loads((ROOT / "configs" / f"lispan_forward_final_{cr}C.json").read_text())
        params, prot = runs._params(spec.get("params", {})), runs._protocol(spec["protocol"])
        rev = "".join("r" if r else "i" for r in params.reversible)
        ref = load(ROOT / "results" / "lispan" / f"pinn_reference_{prot.crate:g}C_{rev}_Zcc{params.Z_CC:g}.npz")
        t_end = float(ref["t"][-1]) * 0.98
        per = {}
        for phase, start in (("before_inventory", 10 ** 9), ("with_inventory", 0)):
            cpu = {}
            for n in (n_skip, n_steps):
                tc = dict(spec["train"], adam_steps=n, eval_every=10 ** 9, checkpoint_every=0, latest_every=0,
                          log_every=10 ** 9, charge_total_start=start, ema_decay=0.0)
                with tempfile.TemporaryDirectory() as tmp:
                    c0 = time.process_time()
                    train(LiSPANTrainConfig(**tc), params, prot, t_end, Path(tmp), reference=None, log=lambda m: None)
                    cpu[n] = time.process_time() - c0
            per[phase] = (cpu[n_steps] - cpu[n_skip]) / (n_steps - n_skip)
        steps, start = spec["train"]["adam_steps"], spec["train"].get("charge_total_start", 0)
        total = min(start, steps) * per["before_inventory"] + max(steps - start, 0) * per["with_inventory"]
        rows.append({"crate": float(cr), "cpu_s_per_step": per, "steps": steps, "projected_cpu_h": total / 3600})
        print(f"PINN step timing {cr} C: {per}, projected {total / 3600:.2f} h CPU", flush=True)
    return rows


def pinn_evaluation(train_rows, repeats):
    rows = []
    torch.set_num_threads(1)
    for r in train_rows:
        if not r["finished"] or r["seed"] != 0:
            continue
        model = runs.load_model(RUNS / r["run"] / "final.pt")
        T = torch.linspace(0.0, 1.0, 1000).view(-1, 1)
        Yc = torch.linspace(0.0, 1.0, 20)
        Yq = Yc.view(1, -1, 1).expand(1000, 20, 1).reshape(-1, 1)
        Tq = T.view(-1, 1, 1).expand(1000, 20, 1).reshape(-1, 1)
        tv, tf = [], []
        with torch.no_grad():
            for _ in range(repeats):
                c0 = time.process_time(); model.voltage(T); tv.append(time.process_time() - c0)
                c0 = time.process_time(); model.species(Yq, Tq); model.ce(Yq * model.Yi, Tq); tf.append(time.process_time() - c0)
        rows.append({"run": r["run"], "crate": r["crate"], "voltage_1000_times_ms": 1e3 * min(tv),
                     "fields_20x1000_ms": 1e3 * min(tf)})
    return rows


def inverse_costs(train_rows, fv_rows):
    out = {}
    st = []
    for name in INVERSE["stage1"]:
        h = runs.history(RUNS / name)
        st.append({"run": name, "steps": h["history"][-1]["step"], "wall_h": h["history"][-1]["time_s"] / 3600})
    h = runs.history(RUNS / INVERSE["multirate"])
    res = json.loads((RUNS / INVERSE["multirate"] / "inverse_result.json").read_text())
    fwd = [r for r in train_rows if r["finished"] and r["seed"] == 0]
    out["pinn"] = {"stage1": st, "multirate": {"run": INVERSE["multirate"], "steps": h["history"][-1]["step"],
                                               "wall_h": h["history"][-1]["time_s"] / 3600},
                   "forward_pretraining_wall_h": sum(r["train_wall_h"] for r in fwd) if len(fwd) == 2 else None,
                   "errors_pct_or_mV": res["errors_pct_or_mV"]}
    out["pinn"]["inverse_only_wall_h"] = sum(s["wall_h"] for s in st) + out["pinn"]["multirate"]["wall_h"]
    if INVERSE["fv_fit"].exists():
        fv = json.loads(INVERSE["fv_fit"].read_text())
        # CPU cost = solves x measured CPU time of one 0.1 C + one 1 C discharge on the fit's grid (the fit itself
        # shared the machine and was resumed after restarts, so its own timers are not a clean measurement)
        per = sum(r["cpu_s"] for r in fv_rows if r["grid"] == list(fv["grid"]))
        out["finite_volume"] = {"solves_per_rate": fv["n_solves_per_rate"], "cpu_h": fv["n_solves_per_rate"] * per / 3600,
                                "cpu_s_per_solve_pair": per, "logged_wall_h": fv["wall_s"] / 3600, "grid": fv["grid"],
                                "rms_fit_mV": fv["rms_fit_mV"], "errors_pct_or_mV": fv.get("errors_pct_or_mV")}
    return out


def machine():
    try:
        cpu = [l.split(":", 1)[1].strip() for l in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines()
               if l.startswith("Model name")][0]
    except Exception:  # noqa: BLE001
        cpu = platform.processor()
    import scipy
    return {"cpu": cpu, "cores": os.cpu_count(), "threads_per_process": 1, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}


def markdown(out):
    L = [f"Machine: {out['machine']['cpu']}, one thread per process.", "",
         "| task | cost | accuracy |", "| --- | --- | --- |"]
    for r in out["fv_forward"]:
        L.append(f"| FV discharge {r['crate']:g} C, {r['grid'][0]}/{r['grid'][1]} volumes | {r['cpu_s']:.1f} s CPU | reference |")
    for r in out["pinn_training"]:
        if not r["finished"]:                     # in-progress runs are in the JSON only
            continue
        tag = ""
        _, s1 = runs.time_to(RUNS / r["run"], threshold=1.0)
        _, s05 = runs.time_to(RUNS / r["run"], threshold=0.5)
        L.append(f"| PINN training {r['crate']:g} C, seed {r['seed']}{tag} | {r['train_wall_h']:.2f} h wall, machine shared "
                 f"(evaluations every 2500 steps: < 1 mV from step {s1}, < 0.5 mV from step {s05}) | "
                 f"V rms {r['V_rmse_mV']:.2f} mV, max {r['V_max_mV']:.2f} mV |")
    for r in out["pinn_step_timing"]:
        L.append(f"| PINN training {r['crate']:g} C, CPU projection from measured step time ({r['steps']} steps) | "
                 f"{r['projected_cpu_h']:.2f} h CPU ({1e3 * r['cpu_s_per_step']['before_inventory']:.0f} / "
                 f"{1e3 * r['cpu_s_per_step']['with_inventory']:.0f} ms per step) | - |")
    for r in out["pinn_evaluation"]:
        L.append(f"| trained PINN {r['crate']:g} C: voltage at 1000 times / all fields 20 x 1000 | "
                 f"{r['voltage_1000_times_ms']:.0f} ms / {r['fields_20x1000_ms']:.0f} ms | as trained |")
    inv = out["inverse"]
    if "finite_volume" in inv:
        fv = inv["finite_volume"]
        L.append(f"| 8-parameter inverse, FV least squares ({fv['solves_per_rate']} solves per rate) | {fv['cpu_h']:.2f} h CPU | "
                 + ", ".join(f"{k} {v:+.2f}" for k, v in (fv["errors_pct_or_mV"] or {}).items()) + " |")
    p = inv["pinn"]
    L.append(f"| 8-parameter inverse, PINN (stage 1 + multi-rate LM) | {p['inverse_only_wall_h']:.2f} h"
             + (f" + {p['forward_pretraining_wall_h']:.1f} h forward pre-training" if p["forward_pretraining_wall_h"] else "")
             + " | " + ", ".join(f"{k} {v:+.2f}" for k, v in p["errors_pct_or_mV"].items()) + " |")
    return "\n".join(L) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--md-only", action="store_true", help="rewrite cost_table.md from the existing cost_table.json")
    args = ap.parse_args()
    d = ROOT / "results" / "paper"
    if args.md_only:
        md = markdown(json.loads((d / "cost_table.json").read_text()))
        (d / "cost_table.md").write_text(md)
        print(md)
        return
    out = {"machine": machine(), "fv_forward": fv_timing(args.repeats)}
    out["pinn_training"] = pinn_training()
    out["pinn_step_timing"] = pinn_step_timing()
    out["pinn_evaluation"] = pinn_evaluation(out["pinn_training"], args.repeats)
    out["inverse"] = inverse_costs(out["pinn_training"], out["fv_forward"])
    d = ROOT / "results" / "paper"
    d.mkdir(parents=True, exist_ok=True)
    (d / "cost_table.json").write_text(json.dumps(out, indent=1))
    md = markdown(out)
    (d / "cost_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    sys.exit(main())
