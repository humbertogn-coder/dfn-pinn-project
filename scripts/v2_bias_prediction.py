"""Predict the parameter bias of the inverse problem from a forward run.

The inverse PINN fits voltage data with a model whose forward solution has a
systematic error e(t) = V_pinn(t) - V_ref(t). To first order the optimizer
absorbs the part of e that lies in the span of the voltage sensitivities
S_i(t) = dV/d ln p_i:   S delta = -e  (least squares),  bias_i = exp(delta_i) - 1.

    python scripts/v2_bias_prediction.py results/v2_runs/<forward run> \
        --ref results/v2_reference/ref_I5A_t3000s_ramp30s_x80_r120.npz \
        --sens results/v2_identifiability/sensitivities_1C.npz --params D_p k_n D_n

docs/V2_RESULTS.md 3.3: the observed biases of the bias test (D_n -4.1 %,
D_p +0.5 %, k_n -0.2 %) are the quantity this script predicts.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dfn_pinn.v2.model import DFNPINN  # noqa: E402
from dfn_pinn.v2.params import CellParams, Protocol, Scales  # noqa: E402
from dfn_pinn.v2.reference import load  # noqa: E402
from dfn_pinn.v2.train import TrainConfig  # noqa: E402


def load_model(run: Path, checkpoint="final.pt"):
    spec = json.loads((run / "config.json").read_text())
    cfg = TrainConfig(**spec["train"])
    cell, protocol = CellParams(**spec["cell"]), Protocol(**spec["protocol"])
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    model = DFNPINN(Scales(cell, protocol), cfg.width, cfg.depth, cfg.act, cfg.projection, ic_tau_s=cfg.ic_tau_s,
                    inventory=cfg.inventory, fourier_t=cfg.fourier_t, width_scalar=cfg.width_scalar or None,
                    short_t=tuple(cfg.short_t), collector_bc=cfg.collector_bc,
                    fourier_period=cfg.fourier_period).to(dtype)
    ck = torch.load(run / checkpoint, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="final.pt")
    ap.add_argument("--ref", default=str(ROOT / "results/v2_reference/ref_I5A_t3000s_ramp30s_x80_r120.npz"))
    ap.add_argument("--sens", default=str(ROOT / "results/v2_identifiability/sensitivities_1C.npz"))
    ap.add_argument("--params", nargs="*", default=["D_p", "k_n", "D_n"])
    ap.add_argument("--t-min", type=float, default=0.0, help="ignore times below this [s] (data window)")
    ap.add_argument("--t-max", type=float, default=1e12, help="ignore times above this [s] (data window)")
    args = ap.parse_args()

    run = Path(args.run)
    model, cfg = load_model(run, args.checkpoint)
    ref = load(args.ref)
    sens = np.load(args.sens, allow_pickle=True)
    names = [str(p) for p in sens["params"]]
    t = np.asarray(sens["t"], float)
    keep = (t >= args.t_min) & (t <= args.t_max)
    t, S_all = t[keep], np.asarray(sens["S"], float)[:, keep]          # V per unit ln p
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        V_pinn = model.voltage(torch.as_tensor(t / model.sc.t_end, dtype=dtype).view(-1, 1)).numpy().ravel()
    V_ref = np.interp(t, ref["t"], ref["V"])
    e = V_pinn - V_ref
    idx = [names.index(p) for p in args.params]
    S = S_all[idx].T                                                     # (n_t, n_p)
    delta, *_ = np.linalg.lstsq(S, -e, rcond=None)
    e_after = e + S @ delta
    rms = lambda a: 1e3 * np.sqrt(np.mean(a ** 2))
    print(f"{run.name} / {args.checkpoint}: forward voltage error {rms(e):.3f} mV rms, {1e3 * np.abs(e).max():.3f} mV max "
          f"({len(t)} times from {t[0]:.0f} to {t[-1]:.0f} s)")
    print(f"part of the error absorbable by {args.params}: residual after fit {rms(e_after):.3f} mV rms")
    print("predicted bias (exp(delta) - 1):")
    for p, d in zip(args.params, delta):
        print(f"  {p:8s} {100 * (np.exp(d) - 1):+7.2f} %")
    # correlation structure of the fitted directions, for reference
    C = np.linalg.inv(S.T @ S)
    corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    print("correlation of the fitted directions:", np.round(corr, 2).tolist())


if __name__ == "__main__":
    main()
