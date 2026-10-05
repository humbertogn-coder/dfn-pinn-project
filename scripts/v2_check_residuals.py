"""Consistency check of the v2 residual implementation against PyBaMM.

Fits every network to the PyBaMM fields by plain regression (NO physics), then
evaluates the physics residuals at random points. If signs, units and scales
in residuals.py agree with PyBaMM's DFN, the residuals of the data-fitted
model are much smaller than those of an untrained model. This is a
verification of the equations, not a PINN result.

    python scripts/v2_check_residuals.py --ref results/v2_reference/<file>.npz
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.v2.params import CellParams, Protocol, Scales  # noqa: E402
from dfn_pinn.v2.model import DFNPINN  # noqa: E402
from dfn_pinn.v2.residuals import Residuals  # noqa: E402
from dfn_pinn.v2.reference import load  # noqa: E402
from dfn_pinn.v2.train import TrainConfig, Sampler, residual_terms, term_losses  # noqa: E402
from dfn_pinn.v2.evaluate import compare  # noqa: E402


def build_targets(ref, cell, t_end, dtype):
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype).reshape(-1, 1)
    th = ref["t"] / t_end
    tg = {}
    X = ref["x"] / cell.L
    XX, TT = np.meshgrid(X, th, indexing="ij")
    tg["ce"] = (T(XX), T(TT), T(ref["c_e"] / cell.c_e0))
    tg["phie"] = (T(XX), T(TT), T(ref["phi_e"]))
    for k, xs, L0, Lk, R in (("n", ref["x_n"], 0.0, cell.L_n, cell.R_n),
                             ("p", ref["x_p"], cell.L_n + cell.L_s, cell.L_p, cell.R_p)):
        xk = (xs - L0) / Lk
        XK, TK = np.meshgrid(xk, th, indexing="ij")
        tg[f"phis_{k}"] = (T(XK), T(TK), T(ref[f"phi_s_{k}"]))
        tg[f"j_{k}"] = (T(XK), T(TK), T(ref[f"j_{k}"]))
        rr = ref[f"r_{k}"] / R
        RR, XR, TR = np.meshgrid(rr, xk, th, indexing="ij")
        tg[f"theta_{k}"] = (T(RR ** 2), T(XR), T(TR), T(ref[f"theta_{k}"]))
        # volume-averaged stoichiometry (finite-volume shells) for the cheap theta_bar fit
        nr = len(rr)
        edges = np.linspace(0, 1, nr + 1)
        vol = (edges[1:] ** 3 - edges[:-1] ** 3)[:, None, None]
        thbar = (ref[f"theta_{k}"] * vol).sum(0) / vol.sum()
        tg[f"thbar_{k}"] = (T(XK), T(TK), T(thbar))
    tg["V"] = (T(th), T(ref["V"]))
    return tg


def data_loss(model, tg, gen, n=2048, fit_theta=True):
    c = model.sc.cell
    def pick(*arrs):
        idx = torch.randint(0, arrs[0].shape[0], (n,), generator=gen)
        return [a[idx] for a in arrs]
    L = {}
    X, t, y = pick(*tg["ce"])
    s1, s2 = (X > c.X1).to(X.dtype), (X > c.X2).to(X.dtype)
    L["ce"] = ((model.ce(X, t, s1, s2) - y) ** 2).mean() / 0.1 ** 2
    X, t, y = pick(*tg["phie"])
    s1, s2 = (X > c.X1).to(X.dtype), (X > c.X2).to(X.dtype)
    L["phie"] = ((model.phie(X, t, s1, s2) - y) ** 2).mean() / 1e-3 ** 2
    for k in ("n", "p"):
        x, t, y = pick(*tg[f"phis_{k}"])
        scale = 1e-6 if k == "n" else 1e-3
        L[f"phis_{k}"] = ((model.phis(k, x, t) - y) ** 2).mean() / scale ** 2
        x, t, y = pick(*tg[f"j_{k}"])
        L[f"j_{k}"] = ((model.j(k, x, t) - y) ** 2).mean() / 0.01 ** 2
        if fit_theta:
            s, x, t, y = pick(*tg[f"theta_{k}"])
            L[f"theta_{k}"] = ((model.theta(k, s, x, t) - y) ** 2).mean() / 1e-3 ** 2
        elif model.inventory == "soft":
            x, t, y = pick(*tg[f"thbar_{k}"])
            L[f"thbar_{k}"] = ((model.theta_bar(k, x, t) - y) ** 2).mean() / 1e-3 ** 2
    t, y = pick(*tg["V"])
    L["V"] = ((model.voltage(t) - y) ** 2).mean() / 1e-3 ** 2
    return L


def residual_rms(model, cell, seed=7, dtype=torch.float64):
    cfg = TrainConfig(n_electrode=2048, n_particle=4096, n_separator=512, n_boundary=512)
    terms = residual_terms(Residuals(model), Sampler(cfg, dtype, torch.Generator().manual_seed(seed)).draw(), cell)
    return {k: float(v.detach().square().mean().sqrt()) for k, v in terms.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--width-scalar", type=int, default=0)
    ap.add_argument("--fourier", type=int, default=0, help="Fourier time-feature frequencies")
    ap.add_argument("--act", default="tanh")
    ap.add_argument("--short-t", nargs="*", type=float, default=[], help="short-time feature constants tau [s]")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--skip-residuals", action="store_true")
    ap.add_argument("--no-theta", action="store_true", help="fit theta_bar instead of the full particle field (fast)")
    ap.add_argument("--points", type=int, default=2048)
    args = ap.parse_args()
    ref = load(args.ref)
    meta = ref["meta"]
    cell, protocol = CellParams(**meta["cell"]), Protocol(**meta["protocol"])
    dtype = torch.float64
    torch.manual_seed(0)
    model = DFNPINN(Scales(cell, protocol), width=args.width, act=args.act, fourier_t=args.fourier,
                    width_scalar=args.width_scalar or None, short_t=tuple(args.short_t)).to(dtype)
    print(f"architecture: width {args.width}, scalar width {args.width_scalar or args.width // 2}, "
          f"fourier_t {args.fourier}, short_t {args.short_t}, act {args.act}, parameters {sum(p.numel() for p in model.parameters())}")
    before = None if args.skip_residuals else residual_rms(model, cell)
    tg = build_targets(ref, cell, protocol.t_end_s, dtype)
    gen = torch.Generator().manual_seed(3)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, (1e-5 / args.lr) ** (1 / args.steps))
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        L = data_loss(model, tg, gen, n=args.points, fit_theta=not args.no_theta)
        loss = sum(L.values())
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0:
            print(f"fit step {step} loss {float(loss):.3e} ({time.perf_counter()-t0:.0f}s) "
                  + " ".join(f"{k}:{float(v):.2e}" for k, v in L.items()), flush=True)
    after = None if args.skip_residuals else residual_rms(model, cell)
    metrics, _ = compare(model, ref, max_times=60)
    if before is not None:
        print("\nterm               RMS untrained   RMS data-fitted")
        for k in before:
            print(f"{k:18s} {before[k]:14.4e} {after[k]:14.4e}")
    print("\nfield errors of the data fit:", json.dumps({k: round(v, 5) for k, v in metrics.items()}))
    print(f"FIT SUMMARY width={args.width} fourier={args.fourier} short_t={args.short_t} act={args.act}: V rmse {metrics['V_rmse_mV']:.2f} mV, "
          f"phi_e rmse {metrics['phie_rmse_mV']:.2f} mV, phi_s,p rmse {metrics['phisp_rmse_mV']:.2f} mV, c_e rmse {metrics['ce_rmse']:.1f}, "
          f"j_n rmse {metrics['j_n_rmse_rel']:.4f}, j_p rmse {metrics['j_p_rmse_rel']:.4f}; last data terms "
          + " ".join(f"{k}:{float(v):.2e}" for k, v in L.items()))
    if args.out:
        Path(args.out).write_text(json.dumps({"before": before, "after": after, "fit_metrics": metrics,
                                              "args": vars(args)}, indent=1))
        torch.save(model.state_dict(), Path(args.out).with_suffix(".pt"))


if __name__ == "__main__":
    main()
