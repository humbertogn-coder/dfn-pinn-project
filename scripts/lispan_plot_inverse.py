"""Figure of a Li-SPAN inverse run: parameter trajectories, data vs PINN voltage, residuals.

    python scripts/lispan_plot_inverse.py results/lispan_runs/<inverse run> [--checkpoint final.pt]
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lispan_plot_run import load_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="final.pt")
    args = ap.parse_args()
    run = Path(args.run)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, cfg, params, protocol, step, dtype = load_run(run, args.checkpoint)
    ck = torch.load(run / args.checkpoint, weights_only=False)
    model.set_parameter_values({n: ck["parameters"][n] for n in cfg.inverse_params})
    hist = json.loads((run / "history.json").read_text())["history"]
    raw = np.load(ROOT / cfg.data_path if not Path(cfg.data_path).is_absolute() else cfg.data_path, allow_pickle=False)
    truth = json.loads(str(raw["truth_json"])) if "truth_json" in raw.files else {}
    t_d, V_d = np.asarray(raw["t"]), np.asarray(raw["V"])
    keep = t_d <= model.t_end
    t_d, V_d = t_d[keep], V_d[keep]
    noise = np.random.default_rng(cfg.seed + 7).normal(0.0, cfg.data_noise_mV * 1e-3, np.asarray(raw["V"]).shape)[keep] \
        if cfg.data_noise_mV > 0 else 0.0
    with torch.no_grad():
        V_p = model.voltage(torch.as_tensor(t_d / model.t_end, dtype=dtype).view(-1, 1)).view(-1).numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    steps = np.array([h["step"] for h in hist])
    for n in cfg.inverse_params:
        v = np.array([h["params"][n] for h in hist])
        tr = truth.get(n, 1.0)
        ax[0].plot(steps, (v - tr) * 1e3 if n.startswith("U0") else 100 * (v / tr - 1), label=n)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_yscale("symlog", linthresh=1.0)
    ax[0].set_xlabel("step"); ax[0].set_ylabel("error vs truth [%] (mV for U0)")
    ax[0].legend(); ax[0].set_title("parameters")
    h = t_d / 3600.0
    ax[1].plot(h, V_d + noise, ".", ms=3, label="data (with noise)")
    ax[1].plot(h, V_p, "-", lw=1, label="PINN")
    ax[1].set_xlabel("t [h]"); ax[1].set_ylabel("V [V]"); ax[1].legend(); ax[1].set_title(f"{protocol.crate:g} C")
    ax[2].plot(h, 1e3 * (V_p - V_d - noise), ".", ms=3, label="PINN - noisy data")
    ax[2].plot(h, 1e3 * (V_p - V_d), "-", lw=1, label="PINN - noise-free data")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlabel("t [h]"); ax[2].set_ylabel("mV"); ax[2].legend(); ax[2].set_title("residuals")
    fig.tight_layout()
    out = run / "inverse_summary.png"
    fig.savefig(out, dpi=130)
    print("written", out)


if __name__ == "__main__":
    sys.exit(main())
