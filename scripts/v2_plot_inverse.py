"""Plot parameter trajectories of an inverse v2 run (true multipliers = 1).

    python scripts/v2_plot_inverse.py results/v2_runs/<inverse run>
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dfn_pinn.v2.params import CellParams  # noqa: E402


def main(run):
    run = Path(run)
    hist = json.loads((run / "history.json").read_text())["history"]
    cfg = json.loads((run / "config.json").read_text())["train"]
    names = cfg["inverse_params"]
    cell = CellParams()
    true = {"D_n": cell.D_n, "D_p": cell.D_p, "k_n": cell.k_n, "k_p": cell.k_p, "sigma_p": cell.sigma_p,
            "D_e": 1.0, "kappa_e": 1.0}                 # electrolyte functions: reported as multipliers
    rows = [h for h in hist if "parameters" in h]
    steps = [h["step"] for h in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n in names:
        ax[0].plot(steps, [h["parameters"][n] / true[n] for h in rows], label=n)
    ax[0].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[0].set_xlabel("Adam step"); ax[0].set_ylabel("estimate / true value"); ax[0].set_yscale("log")
    ax[0].legend(); ax[0].set_title("Parameter recovery")
    if "misfit_mV" in rows[-1]:                      # multi-rate run: one misfit per protocol
        for label in rows[-1]["misfit_mV"]:
            ax[1].semilogy(steps, [h["misfit_mV"][label] for h in rows], label=f"voltage RMS {label} [mV]")
    else:
        ax[1].semilogy(steps, [h["data_V"] ** 0.5 * cfg.get("data_scale_mV", 1.0) for h in rows],
                       label="voltage data RMS [mV]")
    ax[1].axhline(cfg.get("data_noise_mV", 0) or 1e-9, color="k", lw=0.8, ls="--", label="noise level")
    ax[1].set_xlabel("Adam step"); ax[1].legend(); ax[1].set_title("Data misfit")
    fig.suptitle(run.name); fig.tight_layout()
    out = run / "inverse_parameters.png"
    fig.savefig(out, dpi=90)
    last = rows[-1]["parameters"]
    print(out)
    for n in names:
        print(f"{n}: estimate {last[n]:.4g}, true {true[n]:.4g}, error {100 * (last[n] / true[n] - 1):+.1f} %")


if __name__ == "__main__":
    main(sys.argv[1])
