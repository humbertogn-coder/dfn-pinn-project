"""Multi-C-rate synthetic inverse problem (shared parameters, one network set per rate).

    python scripts/v2_make_reference.py --current 2.5 --t-end 6500 --nx 80 --nr 120
    python scripts/v2_make_reference.py --current 10  --t-end 1400 --nx 80 --nr 120
    python scripts/v2_inverse_multirate.py --config configs/v2_inverse_multirate.json

Each protocol in the config needs a reference npz (evaluation) and a data npz
(t, V); the data file is created from the reference if it does not exist.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from dfn_pinn.v2.multirate import train_multirate  # noqa: E402
from dfn_pinn.v2.params import CellParams, Protocol  # noqa: E402
from dfn_pinn.v2.reference import load  # noqa: E402
from dfn_pinn.v2.train import TrainConfig  # noqa: E402


def parse_value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "v2_inverse_multirate.json"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", default=None, help="run folder (or latest.pt) of an interrupted run to continue")
    ap.add_argument("--init", default=None,
                    help="checkpoint of a previous multi-rate run: warm start (networks + current parameter "
                         "estimates, fresh optimizer); use with a config that releases further parameters")
    args = ap.parse_args()
    spec = json.loads(Path(args.config).read_text())
    tcfg = dict(spec["train"])
    for item in args.set:
        k, v = item.split("=", 1)
        tcfg[k] = parse_value(v)
    cfg = TrainConfig(**tcfg)
    cell = CellParams(**spec.get("cell", {}))
    protocols, references, data_paths = [], [], []
    for entry in spec["protocols"]:
        protocols.append(Protocol(**entry["protocol"]))
        ref_path = ROOT / entry["reference"]
        references.append(load(ref_path) if ref_path.exists() else None)
        data_path = ROOT / entry["data"]
        if not data_path.exists():
            if references[-1] is None:
                raise FileNotFoundError(f"neither {data_path} nor {ref_path} exists")
            r = references[-1]
            np.savez(data_path, t=r["t"], V=r["V"], I=r["I"])
        data_paths.append(data_path)
    resume = None
    if args.resume:
        resume = Path(args.resume)
        if resume.is_dir():
            resume = resume / "latest.pt"
        out = resume.parent
    else:
        name = args.name or spec.get("name", "multirate")
        out = ROOT / "results" / "v2_runs" / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    init_states = None
    if args.init:
        import torch
        ck = torch.load(args.init, weights_only=False)
        init_states = ck["models"]
        if len(init_states) != len(protocols):
            raise ValueError(f"{args.init} holds {len(init_states)} models, config has {len(protocols)} protocols")
    log(f"Output: {out}")
    if args.init:
        log(f"Warm start from {args.init} (parameters there: {ck.get('parameters')})")
    train_multirate(cfg, cell, protocols, data_paths, out, references=references, log=log,
                    init_states=init_states, resume=resume)


if __name__ == "__main__":
    main()
