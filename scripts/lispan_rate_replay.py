"""Replay a rate-parametric PINN run from one of its checkpoints into a NEW folder, optionally with changed settings,
up to a given step.  Used to test the wave-2 stabilization on the wave-1 excursions (docs/PHASE3_PARAMETRIC_PINN.md):
the checkpoint carries the network, the Adam state, the adaptive weights and the random generator, so the replay draws
the same batches as the original run until a changed setting makes it diverge.

    python scripts/lispan_rate_replay.py --checkpoint results/lispan_rate_runs/rate_w96_2xpts_s0/step_10000.pt \
        --out results/lispan_rate_replays/2xpts_fix --stop-step 11000 --set weight_guard=true rollback=true

The original run folder is not touched (its history.json up to the checkpoint step is copied into --out so that the
rollback monitor has its baseline).  No FV evaluation during the replay; evaluate latest.pt afterwards
(dfn_pinn.lispan.rate_pinn.load_rate_model + evaluate_rates).
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dfn_pinn.lispan.rate_pinn import RateTrainConfig, params_from_dict, train_rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-step", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (JSON values)")
    a = ap.parse_args()
    ck_path = Path(a.checkpoint)
    out = Path(a.out)
    if (out / "start.pt").exists():
        raise SystemExit(f"{out} already holds a replay - choose a new --out (replays are never overwritten)")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    tcfg = dict(ck["train"])
    tcfg.update(stop_step=a.stop_step, eval_every=10 ** 9, latest_every=0, checkpoint_every=0, log_every=100)
    for item in a.set:
        k, v = item.split("=", 1)
        try:
            tcfg[k] = json.loads(v)
        except json.JSONDecodeError:
            tcfg[k] = v
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ck_path, out / "start.pt")
    hist = ck_path.parent / "history.json"
    if hist.exists():
        h = json.loads(hist.read_text())
        (out / "history.json").write_text(json.dumps({"history": [r for r in h.get("history", []) if r["step"] <= ck["step"]],
                                                      "evals": []}))
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()
    log(f"replay of {ck_path} (step {ck['step']}) to step {a.stop_step}, overrides {a.set}")
    train_rate(RateTrainConfig.from_dict(tcfg), params_from_dict(ck["params"]), ck["window_coef"], out, refs=None,
               log=log, resume=out / "start.pt")


if __name__ == "__main__":
    main()
