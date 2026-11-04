"""Train network v5 (chemistry + cathode geometry, src/dfn_pinn/lispan/geo_pinn.py) against FV references.

    python scripts/lispan_geo_train.py --config configs/lispan_geo_v5.json --run-dir results/lispan_geo_runs/geo_v5_s0
    python scripts/lispan_geo_train.py --config configs/lispan_geo_v5.json --name geo_v5 --set seed=1 adam_steps=200

The design file (train.design_file, scripts/lispan_geo_design.py) fixes the chemistry and geometry boxes, the window
(v3 base + geometry correction) and the evaluation cases (FV references in data/lispan_geo_refs/<design>/; a missing
reference is built here, ~1-3 min each).  A --run-dir holding latest.pt (and no final.pt) RESUMES with the stored
configuration; a folder with final.pt is left untouched (as lispan_chem_train.py).
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dfn_pinn.lispan.chem_pinn import chem_eval_summary, check_spec  # noqa: E402
from dfn_pinn.lispan.geo_pinn import (GeoSampler, GeoTrainConfig, build_geo_model, evaluate_geo, geo_residuals,  # noqa: E402
                                      geo_vector, geo_window_t_end, network_params, warm_start_from_chem)
from dfn_pinn.lispan.params import LiSPANParams  # noqa: E402
from dfn_pinn.lispan.rate_pinn import params_from_dict, train_rate  # noqa: E402


def load_cases(design, log):
    from lispan_geo_design import ref_job
    cases = {}
    for c in design["eval_cases"]:
        path = ROOT / c["ref"]
        if not path.exists():
            _, msg, wall = ref_job((c, design["spec"]))
            log(f"  built reference {path.name}: {msg} ({wall:.0f} s)")
        with np.load(path) as z:
            ref = {"t": np.array(z["t"]), "V": np.array(z["V"])}
        cases[c["id"]] = {"rate": c["rate"], "chem": c["chem"], "geo": c["geo"], "ref": ref, "select": c["select"]}
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lispan_geo_v5.json"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="train overrides key=value (JSON values)")
    ap.add_argument("--no-eval", action="store_true", help="skip the FV references (smoke tests)")
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
            name = args.name or spec.get("name", "geo_v5")
            run_dir = ROOT / "results" / "lispan_geo_runs" / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    design_path = Path(tcfg["design_file"])
    design = json.loads((design_path if design_path.is_absolute() else ROOT / design_path).read_text())
    tcfg["chem_inputs"] = check_spec(design["spec"])
    tcfg["geo_inputs"] = design["geo_spec"]
    tcfg["rate_min"], tcfg["rate_max"] = design["rate_min"], design["rate_max"]
    tcfg["window_rates"] = design["window_rates"]
    if resume is None:
        base = replace(LiSPANParams(), **{k: v for k, v in pdict.items() if k in LiSPANParams.__dataclass_fields__})
        params = network_params(replace(base, Z_CC=design["Z_CC"], reversible=tuple(design["reversible"]), c_DL=1e-6))
    else:
        params = params_from_dict(pdict)
    cfg = GeoTrainConfig.from_dict(tcfg)
    window = design["window"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()

    log(f"Output: {run_dir}" if resume is None else f"Resuming {run_dir} (configuration from the run folder)")
    log(f"design {design_path.name}: chemistry " + ", ".join(f"{s['name']} [{s['lo']:.3g}, {s['hi']:.3g}]" for s in cfg.chem_inputs)
        + "; geometry " + ", ".join(f"{s['name']} [{s['lo']:.3g}, {s['hi']:.3g}]" for s in cfg.geo_inputs)
        + f"; rates [{cfg.rate_min:g}, {cfg.rate_max:g}] C; separator L {params.L_sep * 1e6:.0f} um porosity "
        f"{params.eps_e_sep:.2f} beta {params.beta_sep:g}; window correction degree {window['degree']} (max "
        f"{100 * window['max_rel_err']:.2f} %, held out {100 * window['holdout_max_rel_err']:.2f} %, {window['n_samples']} FV runs)")
    cases = {} if args.no_eval else load_cases(design, log)
    for cid, c in sorted(cases.items()):
        vals = [c["chem"].get(s["name"], 0.0) for s in cfg.chem_inputs]
        te = geo_window_t_end(window, cfg.chem_inputs, cfg.geo_inputs, c["rate"], vals, geo_vector(c["geo"]), cfg.window_fraction)
        tc = float(c["ref"]["t"][-1])
        flag = "" if te < tc else "  <-- WINDOW PASSES THE CUT-OFF"
        log(f"  {cid}{' (select)' if c['select'] else ''}: FV cut-off {tc:.0f} s, window {te:.0f} s ({100 * te / tc:.1f} %){flag}")
    torch.manual_seed(cfg.seed)
    model = build_geo_model(cfg, params, window)
    if resume is None and cfg.init_from:
        src = Path(cfg.init_from)
        info = warm_start_from_chem(model, src if src.is_absolute() else ROOT / src)
        log(f"warm start from {cfg.init_from}: {info['layers']} layers (step {info['rate_step']}); geometry columns zero")
    _, history, evals = train_rate(cfg, params, window["coef_base"], run_dir, refs=cases, log=log, resume=resume, model=model,
                                   sampler_factory=GeoSampler, residual_fn=geo_residuals,
                                   evaluate_fn=evaluate_geo if cases else (lambda m, r: {}),
                                   score_fn=(lambda ev: ev["V_rmse_select_mV"]) if cases else None,
                                   eval_summary_fn=chem_eval_summary if cases else (lambda ev: "no evaluation"),
                                   kind="lispan_geo", extra_save={"design_file": str(design_path), "window": window})
    if (run_dir / "final.pt").exists():
        (run_dir / "final_metrics.json").write_text(json.dumps(evals[-1] if evals else {}, indent=1))
        log("done")


if __name__ == "__main__":
    main()
