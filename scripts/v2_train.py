"""Train the v2 DFN PINN (forward problem) and evaluate it against PyBaMM.

Examples (Anaconda Prompt, repository root, env dfn-pinn):
    python scripts/v2_make_reference.py
    python scripts/v2_train.py --config configs/v2_forward_1C.json
    python scripts/v2_train.py --config configs/v2_forward_1C.json --set adam_steps=2000 --name quick

Outputs go to results/v2_runs/<name>_<timestamp>/ (config, history, checkpoints,
final metrics). The PyBaMM reference is used ONLY for evaluation.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dfn_pinn.v2.params import CellParams, Protocol  # noqa: E402
from dfn_pinn.v2.reference import load  # noqa: E402
from dfn_pinn.v2.train import TrainConfig, train  # noqa: E402
from dfn_pinn.v2.evaluate import compare, screen  # noqa: E402


def parse_value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "v2_forward_1C.json"))
    ap.add_argument("--set", nargs="*", default=[], help="override train keys, e.g. adam_steps=5000")
    ap.add_argument("--name", default=None)
    ap.add_argument("--init", default=None, help="optional checkpoint to warm start from")
    ap.add_argument("--fresh-optimizer", action="store_true", help="ignore the Adam state stored in --init")
    ap.add_argument("--resume", default=None,
                    help="run folder (or its latest.pt) of an interrupted run: continue it in place with the "
                         "config stored in that folder (--config/--set are ignored)")
    args = ap.parse_args()

    resume = None
    if args.resume:
        resume = Path(args.resume)
        if resume.is_dir():
            resume = resume / "latest.pt"
        stored = json.loads((resume.parent / "config.json").read_text())
        spec = {"train": stored["train"], "cell": stored["cell"], "protocol": stored["protocol"],
                "reference": json.loads(Path(args.config).read_text()).get("reference")}
        args.set = []
    else:
        spec = json.loads(Path(args.config).read_text())
    tcfg = dict(spec.get("train", {}))
    for item in args.set:
        k, v = item.split("=", 1)
        tcfg[k] = parse_value(v)
    if tcfg.get("data_path") and not Path(tcfg["data_path"]).is_absolute():
        tcfg["data_path"] = str(ROOT / tcfg["data_path"])
    cfg = TrainConfig(**tcfg)
    cell = CellParams(**spec.get("cell", {}))
    protocol = Protocol(**spec.get("protocol", {}))
    ref_path = ROOT / spec["reference"] if spec.get("reference") else None
    reference = load(ref_path) if ref_path and ref_path.exists() else None
    if reference is None:
        print("WARNING: reference not found; run scripts/v2_make_reference.py first. Training without evaluation.")
    if resume is not None:
        out = resume.parent
    else:
        name = args.name or spec.get("name", "v2")
        out = ROOT / "results" / "v2_runs" / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    init = torch.load(args.init, weights_only=False) if args.init else {}
    for key in ("collector_bc", "fourier_period", "fourier_t", "width", "depth", "short_t", "inventory"):
        stored = init.get("train", {}).get(key, TrainConfig().__dict__[key]) if init else None
        if init and stored != getattr(cfg, key):
            print(f"WARNING: --init checkpoint has {key}={stored!r} but the config has {getattr(cfg, key)!r}; "
                  "the architectures differ (pass --set to match the checkpoint).")
    init_state, init_weights, init_opt = init.get("model"), init.get("group_weights"), init.get("optimizer")
    if args.fresh_optimizer:
        init_opt = None
    log_file = None

    def log(msg):
        nonlocal log_file
        print(msg, flush=True)
        if log_file is None:
            out.mkdir(parents=True, exist_ok=True)
            log_file = open(out / "train.log", "a", encoding="utf-8")
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Output: {out}")
    model, history, evals = train(cfg, cell, protocol, out, reference=reference, log=log,
                                  init_state=init_state, init_weights=init_weights, init_optimizer=init_opt,
                                  resume=resume)
    if reference is not None:
        metrics, _ = compare(model, reference, kinetics=cfg.kinetics)
        (out / "final_metrics.json").write_text(json.dumps(metrics, indent=1))
        log("Final screen (value, limit, pass):")
        for k, v in screen(metrics).items():
            log(f"  {k:14s} {v[0]:10.4g}  {v[1]:8.3g}  {'PASS' if v[2] else 'fail'}")


if __name__ == "__main__":
    main()
