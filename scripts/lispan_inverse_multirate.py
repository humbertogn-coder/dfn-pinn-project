"""Multi-rate Li-SPAN inverse problem (src/dfn_pinn/lispan/multirate.py).

    python scripts/lispan_inverse_multirate.py --config configs/lispan_inverse_multirate_8p.json
    python scripts/lispan_inverse_multirate.py --resume results/lispan_runs/<run>

Config: {"name", "params": {LiSPANParams overrides}, "rates": [{"current": A/m2, "ramp_s", "init": checkpoint,
"data_path": npz}, ...], "train": {LiSPANTrainConfig fields}}.  The time scale of each rate is 0.98 of the
finite-volume reference discharge time at the nominal parameters (as in scripts/lispan_train.py), so the
stage-1 checkpoints of lispan_train.py can be used as initial states.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dfn_pinn.lispan import LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.pinn import LiSPANTrainConfig  # noqa: E402
from dfn_pinn.lispan.multirate import train_multirate  # noqa: E402
from lispan_train import reference_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    if args.resume:
        run_dir = Path(args.resume)
        spec = json.loads((run_dir / "spec.json").read_text())
        resume = str(run_dir / "latest.pt")
    else:
        spec = json.loads(Path(args.config).read_text())
        for item in args.set:
            k, v = item.split("=", 1)
            try:
                spec["train"][k] = json.loads(v)
            except json.JSONDecodeError:
                spec["train"][k] = v
        name = args.name or spec.get("name", "lispan_multirate")
        run_dir = ROOT / "results" / "lispan_runs" / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        resume = None
    known = {f for f in LiSPANParams.__dataclass_fields__}
    params = LiSPANParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in spec.get("params", {}).items() if k in known})
    cfg = LiSPANTrainConfig(**spec["train"])
    protocols, t_ends, refs, inits, data_paths = [], [], [], [], []
    rev_tag = "".join("r" if r else "i" for r in params.reversible)
    for r in spec["rates"]:
        prot = LiSPANProtocol(current=r["current"], ramp_s=r.get("ramp_s", 30.0))
        ref_path = ROOT / "results" / "lispan" / f"pinn_reference_{prot.crate:g}C_{rev_tag}_Zcc{params.Z_CC:g}.npz"
        ref = reference_for(params, prot, ref_path)
        protocols.append(prot)
        t_ends.append(float(ref["t"][-1]) * 0.98)
        refs.append(ref)
        inits.append(torch.load(ROOT / r["init"], weights_only=False)["model"] if (r.get("init") and not resume) else None)
        data_paths.append(str(ROOT / r["data_path"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=2))
    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()
    log(f"Output: {run_dir}; rates {[p.crate for p in protocols]}, t_end {[round(t) for t in t_ends]} s")
    train_multirate(cfg, params, protocols, t_ends, data_paths, run_dir, init_states=inits, references=refs,
                    log=log, resume=resume)
    log("done")


if __name__ == "__main__":
    sys.exit(main())
