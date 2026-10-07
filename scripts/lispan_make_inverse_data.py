"""Synthetic voltage data for the Li-SPAN inverse PINN: finite-volume discharge (same physics as the PINN:
no double layer, PINN-benchmark reversibility, 30 s ramp) sampled every --every seconds.

    python scripts/lispan_make_inverse_data.py --crate 0.1 --every 300
    python scripts/lispan_make_inverse_data.py --crate 0.1 --truth k0_2=1.3 Z_CC=0.8   # multipliers on the nominal values

The file holds t [s], V [V] (noise-free; the trainer adds data_noise_mV) and the true parameter values
as the multipliers / offsets the PINN learns (LiSPANPINN.PARAMETERS).
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.lispan.params import LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.model import solve  # noqa: E402
from dfn_pinn.lispan.pinn import LiSPANPINN  # noqa: E402

FIELDS = {"k0_1": ("k0", 0), "k0_2": ("k0", 1), "k0_3": ("k0", 2), "b_1": ("b", 0), "b_2": ("b", 1), "b_3": ("b", 2),
          "U0_1": ("U0", 0), "U0_2": ("U0", 1), "U0_3": ("U0", 2), "D_salt": ("D_salt", None),
          "kappa0": ("kappa0", None), "Z_CC": ("Z_CC", None)}


def apply_truth(p: LiSPANParams, truth: dict) -> LiSPANParams:
    for name, val in truth.items():
        fld, idx = FIELDS[name]
        add = name.startswith("U0")
        if idx is None:
            p = replace(p, **{fld: getattr(p, fld) + val if add else getattr(p, fld) * val})
        else:
            vals = list(getattr(p, fld))
            vals[idx] = vals[idx] + val if add else vals[idx] * val
            p = replace(p, **{fld: tuple(vals)})
    return p


def time_of_charge(Q, current, ramp_s):
    """Invert Q(t) = I (t + tau (ln(1 + e^{-2t/tau}) - ln 2)) for t (monotone; Newton from t = Q/I + tau ln 2)."""
    t = Q / current + ramp_s * np.log(2.0)
    for _ in range(50):
        f = current * (t + ramp_s * (np.log1p(np.exp(-2.0 * t / ramp_s)) - np.log(2.0))) - Q
        t = np.maximum(t - f / (current * np.tanh(t / ramp_s) + 1e-30), 1e-6)
    return t


def from_experiment(args):
    p = replace(LiSPANParams(), c_DL=1e-6, Z_CC=args.Z_CC)
    prot = LiSPANProtocol.from_crate(args.crate, ramp_s=30.0)
    a = np.loadtxt(args.experiment, delimiter=",", skiprows=1)
    Q_C_m2 = a[:, 0] * 3.6 * p.m_S_model * 1e3          # mAh/g_S * g_S/m2 * 3.6 C/mAh
    t = time_of_charge(Q_C_m2, prot.current, prot.ramp_s)
    out = Path(args.out) if args.out else ROOT / "results" / "lispan" / "inverse_data" / f"V_{args.crate:g}C_experiment.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t=t, V=a[:, 1], I=np.full_like(t, prot.current), Q_mAh_gS=a[:, 0], truth_json=json.dumps({}),
             model_json=json.dumps({"source": str(args.experiment), "crate": args.crate,
                                    "normalization": "Q per model sulfur m_S_model, 1 C = 10 A/m2 (paper convention)"}))
    print(f"{out}: {len(t)} measured points, t {t.min():.0f}-{t.max():.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crate", type=float, default=0.1)
    ap.add_argument("--every", type=float, default=300.0, help="sampling interval [s]")
    ap.add_argument("--t-min", type=float, default=100.0)
    ap.add_argument("--Z-CC", type=float, default=0.025)
    ap.add_argument("--reversible", default="TFF")
    ap.add_argument("--truth", nargs="*", default=[], help="name=value: multiplier (offset in V for U0_m)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--experiment", default=None,
                    help="CSV (Q_mAh_gS, V) of measured points (e.g. results/lispan/paper_digitized/fig4b_exp_0.1C.csv): "
                         "converted to time with the model's capacity normalization (paper convention: 1 C = 10 A/m2, "
                         "specific capacity per model sulfur m_S_model)")
    args = ap.parse_args()
    if args.experiment:
        return from_experiment(args)
    truth = {k: float(v) for k, v in (s.split("=", 1) for s in args.truth)}
    rev = tuple(c.upper() == "T" for c in args.reversible)
    base = replace(LiSPANParams(), c_DL=1e-6, reversible=rev, Z_CC=args.Z_CC)
    p = apply_truth(base, truth)
    prot = LiSPANProtocol.from_crate(args.crate, ramp_s=30.0)
    _, res = solve(p, prot, N_c=40, N_s=20, n_out=2000)
    t_end = float(res["t"][-1])
    t = np.arange(args.t_min, t_end, args.every)
    V = np.interp(t, res["t"], res["V"])
    tag = "nominal" if not truth else "_".join(f"{k}{v:g}" for k, v in truth.items())
    out = Path(args.out) if args.out else ROOT / "results" / "lispan" / "inverse_data" / f"V_{args.crate:g}C_{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    full = {n: (0.0 if n.startswith("U0") else 1.0) for n in LiSPANPINN.PARAMETERS}
    full.update(truth)
    np.savez(out, t=t, V=V, I=np.full_like(t, prot.current), t_end_ref=t_end, truth_json=json.dumps(full),
             model_json=json.dumps({"crate": args.crate, "Z_CC": args.Z_CC, "reversible": list(rev), "c_DL": 1e-6,
                                    "ramp_s": 30.0, "N_c": 40, "N_s": 20}))
    print(f"{out}: {len(t)} samples, t_end {t_end:.0f} s, V {V.min():.3f}-{V.max():.3f} V, truth {truth or 'nominal'}")


if __name__ == "__main__":
    sys.exit(main())
