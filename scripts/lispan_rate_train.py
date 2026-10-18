"""Train the rate-parametric Li-SPAN PINN (Phase 3a, src/dfn_pinn/lispan/rate_pinn.py): one network for every C-rate in
[rate_min, rate_max], evaluated against finite-volume references at several rates.

    python scripts/lispan_rate_train.py --config configs/lispan_rate_0.1-1C.json --name rate_w96_s0
    python scripts/lispan_rate_train.py --config configs/lispan_rate_0.1-1C.json --run-dir results/lispan_rate_runs/x \
        --set width=128 seed=1

If --run-dir already holds latest.pt (and no final.pt) the run RESUMES from it with the configuration stored in the run
folder (--config/--set are ignored then), so a cluster job can simply be resubmitted after a time limit.  A run folder
with final.pt is left untouched.

The FV references (one per rate, dfn_pinn.lispan.model.solve with a negligible double layer) are cached in
results/lispan/pinn_reference_<rate>C_<rev>_Zcc<Z>.npz, the same files the single-rate runs use; the cut-off times of
the window_rates define the training window t_end(r) (rate_pinn module docstring).
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dfn_pinn.lispan import LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.rate_pinn import (RateTrainConfig, fit_window, params_from_dict, train_rate,  # noqa: E402
                                       window_t_end)
from lispan_train import reference_for  # noqa: E402


def references(params, rates, ramp_s):
    rev_tag = "".join("r" if r else "i" for r in params.reversible)
    out = {}
    for rate in sorted(set(float(r) for r in rates)):
        prot = LiSPANProtocol(current=10.0 * rate, ramp_s=ramp_s)
        path = ROOT / "results" / "lispan" / f"pinn_reference_{prot.crate:g}C_{rev_tag}_Zcc{params.Z_CC:g}.npz"
        out[rate] = reference_for(params, prot, path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lispan_rate_0.1-1C.json"))
    ap.add_argument("--name", default=None, help="run name (folder results/lispan_rate_runs/<name>_<UTC time>)")
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
        tcfg, pdict, window_coef = spec["train"], spec["params"], spec["window_coef"]
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
        window_coef = None
        if run_dir is None:
            name = args.name or spec.get("name", "lispan_rate")
            run_dir = ROOT / "results" / "lispan_rate_runs" / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    params = params_from_dict(pdict)
    cfg = RateTrainConfig.from_dict(tcfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()

    refs_all = references(params, list(cfg.window_rates) + list(cfg.eval_rates), cfg.ramp_s)
    t_cut = {r: float(ref["t"][-1]) for r, ref in refs_all.items()}
    if window_coef is None:
        window_coef, fit_err = fit_window(cfg.window_rates, [t_cut[float(r)] for r in cfg.window_rates], cfg.window_degree)
        if fit_err > 0.02:
            raise SystemExit(f"window fit misses an anchor cut-off time by {100 * fit_err:.1f} % - use more window_rates "
                             f"or a narrower rate range")
        log(f"Output: {run_dir}")
        log(f"window: ln(r t_cut) = {window_coef[0]:.5f} + {window_coef[1]:.5f} ln r + {window_coef[2]:.5f} (ln r)^2, "
            f"anchors {list(cfg.window_rates)}, max anchor error {100 * fit_err:.2f} %")
    else:
        log(f"Resuming {run_dir} (configuration and window from the run folder)")
    for r in sorted(t_cut):
        te = window_t_end(window_coef, r, cfg.window_fraction)
        flag = "" if te < t_cut[r] else "  <-- WINDOW PASSES THE CUT-OFF"
        log(f"  {r:g} C: FV cut-off {t_cut[r]:.0f} s, PINN window {te:.0f} s ({100 * te / t_cut[r]:.1f} %){flag}")
    refs = {float(r): refs_all[float(r)] for r in cfg.eval_rates}
    model, history, evals = train_rate(cfg, params, window_coef, run_dir, refs=refs, log=log, resume=resume)
    if (run_dir / "final.pt").exists():
        (run_dir / "final_metrics.json").write_text(json.dumps(evals[-1] if evals else {}, indent=1))
        log("done")


if __name__ == "__main__":
    main()
