"""Train the chemistry-conditioned Li-SPAN PINN (Phase 3b, src/dfn_pinn/lispan/chem_pinn.py): one network for every
C-rate in [rate_min, rate_max] and every chemistry in the box of the design, evaluated against FV references.

    python scripts/lispan_chem_train.py --config configs/lispan_chem_v1.json --run-dir results/lispan_chem_runs/chem_v1_s0
    python scripts/lispan_chem_train.py --config configs/lispan_chem_v1.json --name chem_v1 --set seed=1

The design file (train.design_file, written by scripts/lispan_chem_design.py) fixes the chemistry box, the network's
contact resistance (Z_CC), the training window t_end(r, theta) and the evaluation cases; its FV references are built
here when missing (data/lispan_chem_refs/<design>/, ~1 min each).  As in lispan_rate_train.py, a --run-dir that holds
latest.pt (and no final.pt) RESUMES with the configuration stored in it, and a folder with final.pt is left untouched.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dfn_pinn.lispan.chem_pinn import (ChemRARSampler, ChemSampler, ChemTrainConfig, build_chem_model, check_spec,  # noqa: E402
                                       chem_eval_summary, chem_window_t_end, evaluate_chem, warm_start_from_rate)
from dfn_pinn.lispan.rate_pinn import params_from_dict, train_rate  # noqa: E402
from lispan_chem_design import reference_case  # noqa: E402


def load_cases(design, log):
    cases = {}
    for c in design["eval_cases"]:
        path = ROOT / c["ref"]
        if not path.exists():
            _, msg, wall = reference_case((design["spec"], c["chem_values"], c["rate"], design["Z_CC"], str(path)))
            log(f"  built reference {path.name}: {msg} ({wall:.0f} s)")
        with np.load(path) as z:
            ref = {"t": np.array(z["t"]), "V": np.array(z["V"])}
        cases[c["id"]] = {"rate": c["rate"], "chem": c["chem"], "ref": ref, "select": c["select"]}
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lispan_chem_v1.json"))
    ap.add_argument("--name", default=None, help="run name (folder results/lispan_chem_runs/<name>_<UTC time>)")
    ap.add_argument("--run-dir", default=None, help="explicit run folder (resumed if it holds latest.pt)")
    ap.add_argument("--set", nargs="*", default=[], help="train overrides key=value (JSON values)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    resume = None
    if run_dir is not None and (run_dir / "final.pt").exists():
        print(f"{run_dir} already has final.pt - nothing to do")
        return
    if run_dir is not None and (run_dir / "latest.pt").exists():
        spec = json.loads((run_dir / "config.json").read_text())
        tcfg, pdict = spec["train"], spec["params"]
        resume = run_dir / "latest.pt"
    else:
        spec = json.loads(Path(args.config).read_text())
        tcfg, pdict = dict(spec["train"]), dict(spec.get("params", {}))
        for item in args.set:
            k, v = item.split("=", 1)
            try:
                tcfg[k] = json.loads(v)
            except json.JSONDecodeError:
                tcfg[k] = v
        if run_dir is None:
            name = args.name or spec.get("name", "lispan_chem")
            run_dir = ROOT / "results" / "lispan_chem_runs" / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    design_path = Path(tcfg["design_file"])
    design = json.loads((design_path if design_path.is_absolute() else ROOT / design_path).read_text())
    # the design fixes the box, the window and the network's contact resistance: the config may not contradict it
    tcfg["chem_inputs"] = check_spec(design["spec"])
    tcfg["rate_min"], tcfg["rate_max"] = design["rate_min"], design["rate_max"]
    tcfg["window_rates"] = design["window_rates"]
    pdict = {**pdict, "Z_CC": design["Z_CC"], "reversible": design["reversible"], "c_DL": 1e-6}
    params = params_from_dict(pdict)
    cfg = ChemTrainConfig.from_dict(tcfg)
    if resume is None and cfg.init_from and not (ROOT / cfg.init_from).exists():
        raise SystemExit(f"init_from {cfg.init_from} not found (a Phase 3a rate checkpoint, e.g. "
                         "results/lispan_rate_runs/rate_w96_120k_s0/final.pt)")
    window_coef = design["window"]["coef"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()

    log(f"Output: {run_dir}" if resume is None else f"Resuming {run_dir} (configuration from the run folder)")
    w = design["window"]
    log(f"design {design_path.name}: {len(cfg.chem_inputs)} chemistry inputs "
        + ", ".join(f"{s['name']} [{s['lo']:.3g}, {s['hi']:.3g}]" for s in cfg.chem_inputs)
        + f"; rates [{cfg.rate_min:g}, {cfg.rate_max:g}] C; Z_CC {params.Z_CC:g} Ohm m2; window fit max "
        f"{100 * w['max_rel_err']:.2f} % (held out {100 * w['holdout_max_rel_err']:.2f} %) over {w['n_samples']} FV runs")
    cases = load_cases(design, log)
    for cid, c in sorted(cases.items()):
        vals = [c["chem"].get(s["name"], 0.0) for s in cfg.chem_inputs]
        te = chem_window_t_end(window_coef, cfg.chem_inputs, c["rate"], vals, cfg.window_fraction)
        tc = float(c["ref"]["t"][-1])
        flag = "" if te < tc else "  <-- WINDOW PASSES THE CUT-OFF"
        log(f"  {cid}{' (select)' if c['select'] else ''}: FV cut-off {tc:.0f} s, window {te:.0f} s "
            f"({100 * te / tc:.1f} %){flag}")
    if cfg.rar:
        log(f"residual-based sampling: {100 * cfg.rar_fraction:g} % of the cathode points from the {cfg.rar_pool} worst "
            f"of {cfg.rar_candidates} candidates, refreshed every {cfg.rar_every} steps")
    # the initial weights follow the seed (v1 built the model before train_rate seeded torch, so all v1 runs started
    # from torch's default seed and the seed only changed the batches)
    torch.manual_seed(cfg.seed)
    model = build_chem_model(cfg, params, window_coef)
    if resume is None and cfg.init_from:
        src = Path(cfg.init_from)
        info = warm_start_from_rate(model, src if src.is_absolute() else ROOT / src)
        log(f"warm start from {cfg.init_from}: {info['layers']} layers (rate network at step {info['rate_step']}); "
            f"chemistry columns zero")
    _, history, evals = train_rate(cfg, params, window_coef, run_dir, refs=cases, log=log, resume=resume, model=model,
                                   sampler_factory=ChemRARSampler if cfg.rar else ChemSampler, evaluate_fn=evaluate_chem,
                                   score_fn=lambda ev: ev["V_rmse_select_mV"], eval_summary_fn=chem_eval_summary,
                                   kind="lispan_chem", extra_save={"design_file": str(design_path)})
    if (run_dir / "final.pt").exists():
        (run_dir / "final_metrics.json").write_text(json.dumps(evals[-1] if evals else {}, indent=1))
        log("done")


if __name__ == "__main__":
    main()
