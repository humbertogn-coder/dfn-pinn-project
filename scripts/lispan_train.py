"""Train the Li-SPAN forward PINN (src/dfn_pinn/lispan/pinn.py) and evaluate it against the finite-volume
reference of the same discharge.

    python scripts/lispan_train.py --config configs/lispan_forward_01C.json --name lispan_f01
    python scripts/lispan_train.py --resume results/lispan_runs/<run>          # continue after a restart

The reference (dfn_pinn.lispan.model.solve with a negligible double layer) is generated once per
protocol and cached in results/lispan/pinn_reference_<crate>C.npz.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dfn_pinn.lispan import LiSPANParams, LiSPANProtocol, solve, export, load  # noqa: E402
from dfn_pinn.lispan.pinn import LiSPANTrainConfig, train  # noqa: E402


def reference_for(params: LiSPANParams, protocol: LiSPANProtocol, path: Path, t_end=None):
    if path.exists():
        return load(path)
    from dataclasses import replace
    ref_params = replace(params, c_DL=1e-6)       # the PINN has no double layer
    _, res = solve(ref_params, protocol, N_c=40, N_s=20, n_out=800, t_end=t_end)
    export(res, path, ref_params, protocol, {"purpose": "PINN evaluation reference (c_DL -> 0)"})
    return load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lispan_forward_01C.json"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="train overrides key=value (JSON values)")
    ap.add_argument("--resume", default=None, help="run folder or latest.pt")
    ap.add_argument("--init", default=None, help="checkpoint whose networks warm-start this run")
    args = ap.parse_args()
    if args.resume:
        rp = Path(args.resume)
        run_dir = rp if rp.is_dir() else rp.parent
        spec = json.loads((run_dir / "config.json").read_text())
        tcfg, pdict, prot_d, t_end = spec["train"], spec["params"], spec["protocol"], spec["t_end"]
        resume = str(run_dir / "latest.pt")
    else:
        spec = json.loads(Path(args.config).read_text())
        tcfg = dict(spec["train"]); pdict = dict(spec.get("params", {})); prot_d = dict(spec["protocol"])
        for item in args.set:
            k, v = item.split("=", 1)
            try:
                tcfg[k] = json.loads(v)
            except json.JSONDecodeError:
                tcfg[k] = v
        name = args.name or spec.get("name", "lispan")
        run_dir = ROOT / "results" / "lispan_runs" / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        t_end = spec.get("t_end")
        resume = None
    known = {f.name for f in LiSPANParams.__dataclass_fields__.values()}
    params = LiSPANParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pdict.items() if k in known})
    protocol = LiSPANProtocol(**{k: v for k, v in prot_d.items() if k in ("current", "ramp_s", "t_end_s", "V_min", "V_max")})
    cfg = LiSPANTrainConfig(**tcfg)
    rev_tag = "".join("r" if r else "i" for r in params.reversible)
    ref_path = ROOT / "results" / "lispan" / f"pinn_reference_{protocol.crate:g}C_{rev_tag}_Zcc{params.Z_CC:g}.npz"
    ref = reference_for(params, protocol, ref_path, t_end=None)
    if t_end is None:
        t_end = float(ref["t"][-1]) * 0.98       # stop before the cut-off collapse
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()
    log(f"Output: {run_dir}; reference {ref_path} (t_end ref {float(ref['t'][-1]):.0f} s, PINN t_end {t_end:.0f} s)")
    init_state = torch.load(args.init, weights_only=False)["model"] if args.init else None
    model, history, evals = train(cfg, params, protocol, t_end, run_dir, reference=ref, log=log,
                                  init_state=init_state, resume=resume)
    (run_dir / "final_metrics.json").write_text(json.dumps(evals[-1] if evals else {}, indent=1))
    if cfg.inverse_params:
        est = model.parameter_values()
        raw = np.load(cfg.data_path, allow_pickle=False)
        truth = json.loads(str(raw["truth_json"])) if "truth_json" in raw.files else {}
        res = {"estimates": {n: est[n] for n in cfg.inverse_params}, "truth": {n: truth.get(n) for n in cfg.inverse_params},
               "initial": cfg.inverse_init, "data": cfg.data_path, "noise_mV": cfg.data_noise_mV,
               "final_eval": evals[-1] if evals else {}}
        err = {n: (1e3 * (est[n] - truth[n]) if n.startswith("U0") else 100.0 * (est[n] / truth[n] - 1.0))
               for n in cfg.inverse_params if truth.get(n) is not None}
        res["errors_pct_or_mV"] = err
        (run_dir / "inverse_result.json").write_text(json.dumps(res, indent=1))
        log("Estimates: " + ", ".join(f"{n} {est[n]:.4g}" for n in cfg.inverse_params))
        log("Errors vs truth [% (mV for U0)]: " + ", ".join(f"{n} {v:+.2f}" for n, v in err.items()))
    log("done")


if __name__ == "__main__":
    main()
