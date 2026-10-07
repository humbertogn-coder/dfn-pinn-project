"""Plot a Li-SPAN PINN run against its finite-volume reference.

    python scripts/lispan_plot_run.py results/lispan_runs/<run> [--checkpoint latest.pt]
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dfn_pinn.lispan import LiSPANParams, LiSPANProtocol, load  # noqa: E402
from dfn_pinn.lispan.pinn import LiSPANPINN, LiSPANTrainConfig  # noqa: E402


def load_run(run_dir, checkpoint="latest.pt"):
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "config.json").read_text())
    cfg = LiSPANTrainConfig(**spec["train"])
    known = {f for f in LiSPANParams.__dataclass_fields__}
    params = LiSPANParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in spec["params"].items() if k in known})
    pd = spec["protocol"]
    protocol = LiSPANProtocol(current=pd["current_A_m2"], ramp_s=pd["ramp_s"], t_end_s=pd.get("t_end_s"))
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    model = LiSPANPINN(params, protocol, spec["t_end"], cfg.width, cfg.depth, cfg.act, cfg.fourier_t, cfg.fourier_period,
                       tuple(cfg.short_t), cfg.ic_tau_s, cfg.quad_order).to(dtype)
    ck = torch.load(run_dir / checkpoint, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    return model, cfg, params, protocol, ck.get("step"), dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="latest.pt")
    args = ap.parse_args()
    model, cfg, params, protocol, step, dtype = load_run(args.run, args.checkpoint)
    rev_tag = "".join("r" if r else "i" for r in params.reversible)
    ref = load(ROOT / "results" / "lispan" / f"pinn_reference_{protocol.crate:g}C_{rev_tag}_Zcc{params.Z_CC:g}.npz")
    t = np.asarray(ref["t"]); keep = t <= model.t_end; t = t[keep]
    T = torch.as_tensor(t / model.t_end, dtype=dtype).view(-1, 1)
    yc = np.asarray(ref["y_c"]); ny = len(yc); nt = len(t)
    with torch.no_grad():
        V = model.voltage(T).view(-1).numpy()
        Yq = torch.as_tensor(yc / params.L_cat, dtype=dtype).view(1, ny, 1).expand(nt, ny, 1).reshape(-1, 1)
        Tq = T.view(nt, 1, 1).expand(nt, ny, 1).reshape(-1, 1)
        sp = model.species(Yq, Tq)
        a_Li = torch.clamp(model.ce(Yq * params.L_cat / params.L_tot, Tq) / params.c_Li0, 1e-6)
        R, i_e = model.current_fields(Yq, Tq, sp)
        d = model.dphi(R, sp, a_Li)
        f = model.rates(d, sp, a_Li)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    Q = np.asarray(ref["Q_mAh_gS"])[keep]
    ax = axes[0, 0]; ax.plot(Q, ref["V"][keep], "k", label="reference"); ax.plot(Q, V, "r--", label=f"PINN step {step}")
    ax.set_xlabel("Q / mAh/g"); ax.set_ylabel("V"); ax.legend(); ax.set_title("voltage"); ax.grid(alpha=.3)
    ax = axes[0, 1]
    for key, col in (("c_S4", "C0"), ("c_S3", "C3"), ("c_S2", "C1"), ("c_S1", "C4")):
        ax.plot(Q, np.asarray(ref[key])[:, keep].mean(0), color=col, label=key)
        ax.plot(Q, sp[key].view(nt, ny).numpy().mean(1), "--", color=col)
    ax.set_title("species (solid ref, dashed PINN)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axes[0, 2]
    for m in range(3):
        ax.plot(Q, np.asarray(ref[f"r{m + 1}"])[:, keep].mean(0) * 1e9, color=f"C{m}", label=f"r{m + 1} ref")
        ax.plot(Q, f[m].view(nt, ny).numpy().mean(1) * 1e9, "--", color=f"C{m}")
    ax.set_title("rates x1e9 (cathode mean)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axes[1, 0]
    ax.plot(Q, np.asarray(ref["dphi"])[0, keep], "k", label="dphi(0) ref"); ax.plot(Q, d.view(nt, ny).numpy()[:, 0], "r--", label="PINN")
    ax.set_title("Delta phi at the collector"); ax.legend(); ax.grid(alpha=.3)
    ax = axes[1, 1]
    y = np.asarray(ref["y"])
    for j in np.linspace(0, nt - 1, 5).astype(int)[1:]:
        ax.plot(y * 1e6, np.asarray(ref["c_Li"])[:, keep][:, j], color="k", lw=.8)
        with torch.no_grad():
            ce = model.ce(torch.as_tensor(y / params.L_tot, dtype=dtype).view(-1, 1), torch.full((len(y), 1), float(T[j]), dtype=dtype)).view(-1).numpy()
        ax.plot(y * 1e6, ce, "--", color="C1", lw=.8)
    ax.set_title("c_e profiles (black ref, dashed PINN)"); ax.grid(alpha=.3)
    ax = axes[1, 2]
    ax.plot(Q, np.asarray(ref["eps_L"])[:, keep].mean(0), "k", label="eps_L ref"); ax.plot(Q, sp["eps_L"].view(nt, ny).numpy().mean(1), "r--", label="PINN")
    ax.set_title("Li2S volume fraction"); ax.legend(); ax.grid(alpha=.3)
    fig.suptitle(f"{Path(args.run).name} @ step {step}"); fig.tight_layout()
    out = Path(args.run) / f"plot_{args.checkpoint.replace('.pt', '')}.png"
    fig.savefig(out, dpi=120); print("written", out)


if __name__ == "__main__":
    main()
