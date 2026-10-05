"""Diagnostic figure for a v2 run: voltage and field profiles vs PyBaMM."""
import sys, glob, json
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from dfn_pinn.v2.params import CellParams, Protocol, Scales
from dfn_pinn.v2.model import DFNPINN
from dfn_pinn.v2.reference import load
from dfn_pinn.v2.evaluate import predict_on_reference

def main(run, ref_path, ckpt="final.pt"):
    run = Path(run); ck = torch.load(run / ckpt, weights_only=False)
    cfg = ck["train"]; cell = CellParams(**ck["cell"]); prot = Protocol(**ck["protocol"])
    m = DFNPINN(Scales(cell, prot), cfg["width"], cfg["depth"], cfg["act"], cfg["projection"], ic_tau_s=cfg["ic_tau_s"],
                inventory=cfg.get("inventory", "soft"), fourier_t=cfg.get("fourier_t", 0),
                width_scalar=cfg.get("width_scalar", 0) or None, short_t=tuple(cfg.get("short_t", [])),
                collector_bc=cfg.get("collector_bc", "soft"), fourier_period=cfg.get("fourier_period", 1.0))
    m.load_state_dict(ck["model"], strict=False); m = m.to(torch.float64)
    ref = load(ref_path); p = predict_on_reference(m, ref, kinetics=cfg.get("kinetics", "inverse")); t = ref["t"]
    times = [10, 60, 300, 1500, 3000]; cols = plt.cm.viridis(np.linspace(0, 0.9, len(times)))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    a = ax[0, 0]; a.plot(t, ref["V"], "k-", label="PyBaMM"); a.plot(t, p["V"], "r--", label="PINN v2")
    a.set_xlabel("t [s]"); a.set_ylabel("V [V]"); a.legend(); a2 = a.twinx(); a2.plot(t, 1e3*(p["V"]-ref["V"]), "b:", lw=1); a2.set_ylabel("error [mV]", color="b")
    x = ref["x"]*1e6
    for tt, c in zip(times, cols):
        i = np.argmin(abs(t - tt))
        ax[0,1].plot(x, ref["c_e"][:, i], "-", color=c, label=f"{t[i]:.0f} s"); ax[0,1].plot(x, p["c_e"][:, i], "--", color=c)
        ax[0,2].plot(x, ref["phi_e"][:, i], "-", color=c); ax[0,2].plot(x, p["phi_e"][:, i], "--", color=c)
        ax[1,0].plot(ref["x_n"]*1e6, ref["j_n"][:, i], "-", color=c); ax[1,0].plot(ref["x_n"]*1e6, p["j_n"][:, i], "--", color=c)
        ax[1,0].plot(ref["x_p"]*1e6, ref["j_p"][:, i], "-", color=c); ax[1,0].plot(ref["x_p"]*1e6, p["j_p"][:, i], "--", color=c)
        ax[1,1].plot(ref["x_n"]*1e6, ref["theta_surf_n"][:, i], "-", color=c); ax[1,1].plot(ref["x_n"]*1e6, p["theta_surf_n"][:, i], "--", color=c)
        ax[1,2].plot(ref["x_p"]*1e6, ref["theta_surf_p"][:, i], "-", color=c); ax[1,2].plot(ref["x_p"]*1e6, p["theta_surf_p"][:, i], "--", color=c)
    ax[0,1].set_title("c_e [mol/m3] (solid PyBaMM, dashed PINN)"); ax[0,1].legend(fontsize=8)
    ax[0,2].set_title("phi_e [V]"); ax[1,0].set_title("j [A/m2]"); ax[1,1].set_title("theta_surf negative"); ax[1,2].set_title("theta_surf positive")
    for a in ax.ravel()[1:]: a.set_xlabel("x [um]")
    fig.suptitle(str(run.name)); fig.tight_layout(); out = run / "diagnostic.png"; fig.savefig(out, dpi=90); print(out)

if __name__ == "__main__":
    main(*sys.argv[1:])
