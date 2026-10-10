"""What limits the Simanjuntak Li-SPAN cell at higher rate? Uses the model calibrated by the PINN inverse on the
measured Fig. 4b points (run lispan_inv8p_experiment_tr_*, 0.1 C + 1 C) to answer, inside the model:

  1. Rate dependence: FV discharges at 0.05, 0.1, 0.2, 1 C at the PINN estimates (capacity, mean voltage, what is
     left unreacted at the cut-off, salt depletion, uniformity of Li2S across the electrode).
  2. Voltage-loss budget at 1 C: the same discharge with one loss mechanism removed at a time
       contact resistance Z_CC = 0;  ideal electrolyte (D_salt, D_S, kappa x 100);  fast kinetics (k0 x 100).
  3. The PINN's own fields at 0.1 C and 1 C (cathode-averaged species, Li2S profile) against the FV solution at the
     same parameters (the PINN fields are physics-only, trust-region run).

    python scripts/lispan_rate_analysis.py   ->  results/lispan/span_chem/rate_analysis.{json,npz}
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dfn_pinn.lispan.params import LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.model import solve  # noqa: E402
from dfn_pinn.lispan.runs import load_multirate  # noqa: E402
from lispan_make_inverse_data import apply_truth  # noqa: E402

OUT = ROOT / "results" / "lispan" / "span_chem"
RUN = sorted((ROOT / "results" / "lispan_runs").glob("lispan_inv8p_experiment_tr_*"))[-1]
RATES = (0.05, 0.1, 0.2, 1.0)


def summarize(p, res):
    Q, V = np.asarray(res["Q_mAh_gS"]), np.asarray(res["V"])
    c = {k: np.asarray(res[k]) for k in ("c_S4", "c_S3", "c_S2", "c_S1")}
    c40 = p.c_init[0]
    # sulfur bookkeeping at the end (cathode average): fraction of chains whose step m is still undone
    avg = {k: float(v[:, -1].mean()) for k, v in c.items()}
    epsL = np.asarray(res["eps_L"])[:, -1]
    ce = np.asarray(res["c_Li"])
    return {"Q_end_mAh_gS": float(Q[-1]), "V_mean_V": float(np.trapezoid(V, Q) / (Q[-1] - Q[0])),
            "end_species_mol_m3": avg,
            "end_fraction_S4_S3_S2": [avg["c_S4"] / c40, avg["c_S3"] / c40, avg["c_S2"] / c40],
            "Li2S_end_min_max": [float(epsL.min()), float(epsL.max())],
            "salt_min_max_mol_m3": [float(ce.min()), float(ce.max())]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    est = json.loads((RUN / "inverse_result.json").read_text())["estimates"]
    spec = json.loads((RUN / "spec.json").read_text())
    base = replace(LiSPANParams(), c_DL=1e-6, Z_CC=spec["params"]["Z_CC"], reversible=tuple(spec["params"]["reversible"]))
    p = apply_truth(base, est)
    rep = {"run": RUN.name, "estimates": est, "Z_CC_Ohm_m2": p.Z_CC, "k0": list(p.k0), "U0": list(p.U0), "b": list(p.b),
           "rates": {}, "variants_1C": {}, "variants_0.1C": {}}
    curves = {}
    for cr in RATES:
        _, res = solve(p, LiSPANProtocol.from_crate(cr, ramp_s=30.0), N_c=20, N_s=10, n_out=800)
        rep["rates"][f"{cr:g}"] = summarize(p, res)
        curves[f"rate_{cr:g}_Q"] = res["Q_mAh_gS"]; curves[f"rate_{cr:g}_V"] = res["V"]
        for k in ("c_S4", "c_S3", "c_S2", "c_S1"):
            curves[f"rate_{cr:g}_{k}"] = np.asarray(res[k]).mean(0)
        curves[f"rate_{cr:g}_epsL_end"] = np.asarray(res["eps_L"])[:, -1]
        curves[f"rate_{cr:g}_y_c"] = res["y_c"]
        curves[f"rate_{cr:g}_t"] = res["t"]
        print(cr, rep["rates"][f"{cr:g}"], flush=True)
    variants = {
        "calibrated": p,
        "no contact resistance": replace(p, Z_CC=0.0),
        "ideal electrolyte (x100 transport)": replace(p, D_salt=p.D_salt * 100, D_S=p.D_S * 100, kappa0=p.kappa0 * 100),
        "fast kinetics (k0 x100)": replace(p, kin_scale=100.0),
        "all three removed": replace(p, Z_CC=0.0, D_salt=p.D_salt * 100, D_S=p.D_S * 100, kappa0=p.kappa0 * 100,
                                     kin_scale=100.0),
    }
    for cr, key in ((1.0, "variants_1C"), (0.1, "variants_0.1C")):
        for name, pv in variants.items():
            _, res = solve(pv, LiSPANProtocol.from_crate(cr, ramp_s=30.0), N_c=20, N_s=10, n_out=800)
            rep[key][name] = summarize(pv, res)
            curves[f"{key}_{name}_Q"] = res["Q_mAh_gS"]; curves[f"{key}_{name}_V"] = res["V"]
            print(cr, name, rep[key][name]["Q_end_mAh_gS"], rep[key][name]["V_mean_V"], flush=True)
    # voltage-loss budget at 1 C, at equal capacity (mean over 50-800 mAh/g_S) relative to the 0.05 C curve
    qg = np.linspace(50, 800, 300)
    v_ref = np.interp(qg, curves["rate_0.05_Q"], curves["rate_0.05_V"])
    for key in ("variants_1C", "variants_0.1C"):
        for name in variants:
            v = np.interp(qg, curves[f"{key}_{name}_Q"], curves[f"{key}_{name}_V"], right=np.nan)
            rep[key][name]["loss_vs_0.05C_mV"] = float(1e3 * np.nanmean(v_ref - v))
    # PINN fields (physics-only networks of the inverse run) vs FV at the same parameters
    models, ck = load_multirate(RUN / "final.pt")
    rep["pinn_vs_fv"] = {}
    for m in models:
        cr = m.prot.current / 10.0
        _, res = solve(p, LiSPANProtocol(current=m.prot.current, ramp_s=m.prot.ramp_s), N_c=20, N_s=10, n_out=800,
                       t_end=m.t_end)
        t = np.asarray(res["t"])
        yc = np.asarray(res["y_c"]); ny = yc.size
        T = torch.as_tensor(t / m.t_end, dtype=torch.float32).view(-1, 1)
        Yq = torch.as_tensor(yc / m.p.L_cat, dtype=torch.float32).view(1, ny, 1).expand(len(t), ny, 1).reshape(-1, 1)
        Tq = T.view(-1, 1, 1).expand(len(t), ny, 1).reshape(-1, 1)
        with torch.no_grad():
            sp = m.species(Yq, Tq)
            Vp = m.voltage(T).view(-1).numpy()
        tag = f"pinn_{cr:g}"
        curves[f"{tag}_t"] = t; curves[f"{tag}_Q"] = res["Q_mAh_gS"]; curves[f"{tag}_V_pinn"] = Vp
        curves[f"{tag}_V_fv"] = res["V"]
        errs = {}
        for k in ("c_S4", "c_S3", "c_S2", "c_S1", "eps_L"):
            pk = sp[k].view(len(t), ny).numpy().T
            curves[f"{tag}_{k}_pinn"] = pk.mean(0); curves[f"{tag}_{k}_fv"] = np.asarray(res[k]).mean(0)
            errs[k] = float(np.sqrt(np.mean((pk - np.asarray(res[k])) ** 2)))
        curves[f"{tag}_epsL_end_pinn"] = sp["eps_L"].view(len(t), ny).numpy()[-1]
        curves[f"{tag}_epsL_end_fv"] = np.asarray(res["eps_L"])[:, -1]
        curves[f"{tag}_y_c"] = yc
        mask = t > 100
        rep["pinn_vs_fv"][f"{cr:g}C"] = {"V_rms_mV": float(1e3 * np.sqrt(np.mean((Vp - res["V"])[mask] ** 2))),
                                        "field_rms": errs, "t_end_s": float(m.t_end),
                                        "Q_window_mAh_gS": float(np.asarray(res["Q_mAh_gS"])[-1])}
        print(tag, rep["pinn_vs_fv"][f"{cr:g}C"], flush=True)
    (OUT / "rate_analysis.json").write_text(json.dumps(rep, indent=1))
    np.savez(OUT / "rate_analysis.npz", **{k: np.asarray(v) for k, v in curves.items()})


if __name__ == "__main__":
    sys.exit(main())
