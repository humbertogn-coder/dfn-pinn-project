"""Inverse problem on one aged-cell discharge (step C: commercial-cell data).

    python scripts/v2_inverse_aging.py --cycle results/aging_synthetic/cellA/cycles/cycle_0200.npz \
        --init results/v2_runs/<fresh forward run>/final.pt --name aging_c200

The cycle npz holds t, I, V of a 1C discharge (CC + CV tail) and, for synthetic data, the true aged
state. The CC segment defines the protocol (current, duration); the voltage for t >= --t-min is the
data. The aging parameters of configs/v2_inverse_aging.json (theta_n0, theta_p0, eps_am_n, eps_am_p,
R0) are estimated with the fields warm-started from the fresh forward model (--init). The result
(estimates, truth when available, misfit) is written to aging_result.json in the run folder.
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

from dfn_pinn.v2.aging import cc_segment, load_cycle, protocol_for_cycle, truth_multipliers, write_inverse_data  # noqa: E402
from dfn_pinn.v2.params import CellParams  # noqa: E402
from dfn_pinn.v2.train import TrainConfig, train  # noqa: E402


def parse_value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, help="npz with t, I, V of one discharge")
    ap.add_argument("--config", default=str(ROOT / "configs" / "v2_inverse_aging.json"))
    ap.add_argument("--init", default=None, help="fresh forward checkpoint (networks warm start)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--name", default=None)
    ap.add_argument("--t-min", type=float, default=100.0, help="first data time compared [s] (ramp excluded)")
    ap.add_argument("--ramp", type=float, default=30.0)
    ap.add_argument("--align", default="charge", choices=["charge", "none"],
                    help="charge: place the data at t + ramp ln 2 so that ramped model and stepped data have passed "
                         "the same charge (default); none: compare at equal clock time (20.8 s lag for ramp 30 s)")
    ap.add_argument("--pretrain-steps", type=int, default=0,
                    help="stage 1: re-adapt the fields to the cycle's duration with the fresh parameters and NO data "
                         "(forward problem only), then stage 2 releases the aging parameters with the data")
    args = ap.parse_args()

    spec = json.loads(Path(args.config).read_text())
    tcfg = dict(spec["train"])
    init = torch.load(args.init, weights_only=False) if args.init else {}
    if init:   # architecture follows the checkpoint
        for key in ("width", "depth", "act", "fourier_t", "fourier_period", "collector_bc", "inventory", "short_t", "width_scalar"):
            if key in init.get("train", {}):
                tcfg[key] = init["train"][key]
    for item in args.set:
        k, v = item.split("=", 1)
        tcfg[k] = parse_value(v)

    cyc = load_cycle(args.cycle)
    protocol = protocol_for_cycle(cyc, ramp_s=args.ramp, align=args.align)
    name = args.name or spec.get("name", "aging")
    out = ROOT / "results" / "v2_runs" / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=True)
    data_path, n_pts = write_inverse_data(cyc, out / "data_V.npz", t_min=args.t_min, ramp_s=args.ramp, align=args.align)
    tcfg["data_path"] = str(data_path)
    cfg = TrainConfig(**tcfg)
    cell = CellParams(**spec.get("cell", {}))
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()

    log(f"Output: {out}")
    log(f"Cycle {args.cycle}: CC {protocol.current_A:.3f} A for {protocol.t_end_s:.0f} s (incl. charge shift "
        f"{protocol.t_end_s - (cc_segment(cyc['t'], cyc['I'], cyc['V'])[0][-1]):.1f} s); {n_pts} voltage samples from t >= {args.t_min} s")
    truth = truth_multipliers(cyc, cell)
    if truth:
        log("Truth (multipliers of the fresh cell): " + ", ".join(f"{k} {v:.4f}" for k, v in truth.items())
            + "; " + ", ".join(f"{k} {v:.4g}" for k, v in cyc["truth"].items() if k.startswith(("Loss", "SEI", "Capacity"))))
    init_state, init_weights = init.get("model"), init.get("group_weights")
    if args.pretrain_steps > 0:
        from dataclasses import replace
        cfg_pre = replace(cfg, data_path="", inverse_params=[], inverse_init={}, adam_steps=args.pretrain_steps,
                          checkpoint_every=10 ** 9, latest_every=0, eval_every=10 ** 9)
        pre_dir = out / "pretrain"
        log(f"Stage 1: forward re-adaptation to t_end = {protocol.t_end_s:.0f} s, {args.pretrain_steps} steps, no data")
        model_pre, _, _ = train(cfg_pre, cell, protocol, pre_dir, reference=None, log=log,
                                init_state=init_state, init_weights=init_weights, init_optimizer=None)
        init_state = model_pre.state_dict()
        init_weights = None
        log("Stage 2: aging parameters released with the voltage data")
    model, history, evals = train(cfg, cell, protocol, out, reference=None, log=log,
                                  init_state=init_state, init_weights=init_weights, init_optimizer=None)
    est = model.parameter_values()
    fresh = CellParams()
    mult = {"theta_n0": est["theta_n0"] / fresh.theta_n0, "theta_p0": est["theta_p0"] / fresh.theta_p0,
            "eps_am_n": est["eps_am_n"] / fresh.eps_am_n, "eps_am_p": est["eps_am_p"] / fresh.eps_am_p, "R0_mOhm": 1e3 * est["R0"]}
    misfit = history[-1].get("data_V", float("nan")) ** 0.5 * cfg.data_scale_mV
    log("Estimates (multipliers): " + ", ".join(f"{k} {v:.4f}" for k, v in mult.items()) + f"; final misfit {misfit:.2f} mV rms")
    if truth:
        log("Errors vs truth [%]: " + ", ".join(f"{k} {100 * (mult[k] / truth[k] - 1):+.2f}" for k in truth))
    (out / "aging_result.json").write_text(json.dumps({"cycle": args.cycle, "protocol": protocol.to_dict(),
                                                        "estimates": mult, "truth": truth, "raw_truth": cyc["truth"],
                                                        "misfit_mV": misfit, "n_points": n_pts}, indent=1))


if __name__ == "__main__":
    main()
