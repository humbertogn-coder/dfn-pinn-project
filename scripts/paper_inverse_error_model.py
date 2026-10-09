"""Does the PINN's forward error explain the difference between its 8-parameter estimate and the ideal estimator?

Linear error model (as for the DFN, V2_RESULTS 3): if the PINN voltage is V_FV(theta) + e(t), least squares on the
same data returns theta_PINN ~ theta_ideal - (J^T J)^-1 J^T e, with J the finite-volume Jacobian of the voltage at
the data points (absolute FD steps, dfn_pinn.lispan.fdjac convention: 1 % in log, 1 mV for U0_1).  e is the PINN
voltage at its final parameters minus the finite-volume voltage at the same parameters (40 / 20 volumes).

    python scripts/paper_inverse_error_model.py   ->  results/paper/inverse_error_model.json
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dfn_pinn.lispan import runs  # noqa: E402
from dfn_pinn.lispan.multirate import load_data  # noqa: E402
from lispan_fit_experiment import NAMES, U0_SCALE, values, model_voltage, GRID  # noqa: E402
from dataclasses import replace  # noqa: E402
from dfn_pinn.lispan.params import LiSPANParams  # noqa: E402

RUN = ROOT / "results" / "lispan_runs" / "lispan_inv8p_multirate_lm_20261007T083347Z"
FV = ROOT / "results" / "lispan" / "fit_experiment" / "fit_0.1+1C_nominal.json"


def main():
    torch.set_num_threads(1)
    GRID[:] = [40, 20]
    base = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)
    models, ck = runs.load_multirate(RUN / "final.pt")
    spec = json.loads((RUN / "spec.json").read_text())
    est_p = models[0].parameter_values()
    x_p = np.array([est_p[n] / U0_SCALE if n.startswith("U0") else np.log(est_p[n]) for n in NAMES])
    fv = json.loads(FV.read_text())
    x_i = np.array([fv["estimates"][n] / U0_SCALE if n.startswith("U0") else np.log(fv["estimates"][n]) for n in NAMES])
    e, t_all, rates = [], [], []
    for k, (m, r) in enumerate(zip(models, spec["rates"])):
        (T, Vd), _ = load_data(ROOT / r["data_path"], m.t_end, spec["train"]["data_noise_mV"], spec["train"]["seed"] + 100 * k,
                               torch.float32)
        with torch.no_grad():
            Vp = m.voltage(T).view(-1).double().numpy()
        t = T.view(-1).double().numpy() * m.t_end
        crate = m.prot.crate
        Vf = model_voltage(x_p, crate, t, base)
        e.append(1e3 * (Vp - Vf)); t_all.append(t); rates.append(crate)
        print(f"{crate:g} C: PINN - FV at the PINN parameters: {np.sqrt(np.mean(e[-1] ** 2)):.3f} mV rms", flush=True)
    e = np.concatenate(e)
    # FV Jacobian at the PINN estimate (mV per unit of x), forward differences with absolute steps
    def V(x):
        return np.concatenate([1e3 * model_voltage(x, cr, t, base) for cr, t in zip(rates, t_all)])
    V0 = V(x_p)
    J = np.empty((V0.size, len(NAMES)))
    for i in range(len(NAMES)):
        xp = x_p.copy(); xp[i] += 0.01
        J[:, i] = (V(xp) - V0) / 0.01
        print(f"  J column {NAMES[i]} done", flush=True)
    dx = -np.linalg.solve(J.T @ J, J.T @ e)
    to_u = lambda n, v: 1e3 * U0_SCALE * v if n.startswith("U0") else 100.0 * v      # % or mV
    out = {"run": RUN.name, "ideal": FV.name,
           "forward_error_rms_mV": {f"{r:g}C": float(np.sqrt(np.mean(ee ** 2))) for r, ee in
                                    zip(rates, np.split(e, np.cumsum([len(t) for t in t_all])[:-1]))},
           "predicted_bias_pct_or_mV": {n: float(to_u(n, d)) for n, d in zip(NAMES, dx)},
           "actual_pinn_minus_ideal_pct_or_mV": {n: float(to_u(n, a - b)) for n, a, b in zip(NAMES, x_p, x_i)}}
    d = ROOT / "results" / "paper"
    d.mkdir(parents=True, exist_ok=True)
    (d / "inverse_error_model.json").write_text(json.dumps(out, indent=1))
    for n in NAMES:
        print(f"{n:6s} predicted {out['predicted_bias_pct_or_mV'][n]:+.2f}  actual {out['actual_pinn_minus_ideal_pct_or_mV'][n]:+.2f}")


if __name__ == "__main__":
    sys.exit(main())
