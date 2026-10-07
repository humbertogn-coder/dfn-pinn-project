"""Local identifiability of Li-SPAN parameters from discharge voltage curves (finite-volume model, no PINN).

Same method as scripts/v2_identifiability.py for the DFN: sensitivities S_ij = dV(t_j)/dp_i by central
finite differences of the finite-volume reference (src/dfn_pinn/lispan/model.py), then for Gaussian voltage
noise sigma

* the Fisher information F = S S^T / sigma^2,
* Cramer-Rao lower bounds sqrt((F^-1)_ii) (relative for log-parameters, mV for the U0 offsets),
* the correlation matrix and the singular values of S / sigma,

per C-rate and for combinations of rates.  p_i is log(parameter) for the rate constants, OCV slopes and
transport/resistance parameters, and the offset in volts for U0_m.  Local statements around the nominal
parameters: they flag weak and correlated directions, they do not prove global uniqueness.

The default model is the PINN benchmark (reaction 1 reversible, 2 and 3 irreversible, Z_CC 0.025, no
double layer, 30 s current ramp), observed on a uniform time grid from t_min to 0.95 of the nominal
discharge time (the steep end of discharge would dominate otherwise and depends on the cutoff handling).

    python scripts/lispan_identifiability.py --rates 0.05 0.1 0.2 1 --noise-mV 1 --n-obs 100
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.lispan.params import LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.model import solve  # noqa: E402

# name -> (kind, field, index): kind "log" perturbs log(value), "add" perturbs the value in volts
PARAMS = {
    "k0_1": ("log", "k0", 0), "k0_2": ("log", "k0", 1), "k0_3": ("log", "k0", 2),
    "b_1": ("log", "b", 0), "b_2": ("log", "b", 1), "b_3": ("log", "b", 2),
    "U0_1": ("add", "U0", 0), "U0_2": ("add", "U0", 1), "U0_3": ("add", "U0", 2),
    "D_salt": ("log", "D_salt", None), "kappa0": ("log", "kappa0", None), "Z_CC": ("log", "Z_CC", None),
    "K_sp": ("log", "K_sp", None), "k0_L": ("log", "k0_L", None), "kappa_SPAN": ("log", "kappa_SPAN", None),
    "D_S": ("log", "D_S", None), "t_plus": ("log", "t_plus", None),
}


def perturbed(p: LiSPANParams, name: str, delta: float) -> LiSPANParams:
    kind, fld, idx = PARAMS[name]
    val = getattr(p, fld)
    if idx is None:
        new = val * np.exp(delta) if kind == "log" else val + delta
        return replace(p, **{fld: float(new)})
    vals = list(val)
    vals[idx] = vals[idx] * np.exp(delta) if kind == "log" else vals[idx] + delta
    return replace(p, **{fld: tuple(float(v) for v in vals)})


def voltage_on(p, prot, t_obs, N_c, N_s):
    _, res = solve(p, prot, N_c=N_c, N_s=N_s, n_out=600, t_end=float(t_obs[-1]) * 1.02)
    t, V = res["t"], res["V"]
    if t[-1] < t_obs[-1]:
        raise RuntimeError(f"solution ended at {t[-1]:.0f} s before the observation window ({t_obs[-1]:.0f} s)")
    return np.interp(t_obs, t, V), res


def fisher_report(S, names, sigma, units):
    """S: (n_par, n_obs) sensitivities in V per unit parameter.  Returns CRLB, correlations, singular values."""
    Sn = S / sigma
    Fm = Sn @ Sn.T
    sv = np.linalg.svd(Sn, compute_uv=False)
    try:
        cov = np.linalg.inv(Fm)
        if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0):
            raise np.linalg.LinAlgError
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(Fm)
        sd = np.full(len(names), np.inf)
        corr = np.full((len(names), len(names)), np.nan)
    crlb = {n: (float(s * 100.0) if units[n] == "rel" else float(s * 1e3)) for n, s in zip(names, sd)}
    return {"crlb": crlb, "units": {n: ("%" if units[n] == "rel" else "mV") for n in names},
            "corr": corr.tolist(), "singular_values": sv.tolist(),
            "cond": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")}


def strongest_pairs(corr, names, k=8):
    c = np.array(corr, dtype=float)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(c[i, j]):
                out.append((abs(c[i, j]), names[i], names[j], c[i, j]))
    return [(a, b, round(float(v), 4)) for _, a, b, v in sorted(out, reverse=True)[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", nargs="+", type=float, default=[0.05, 0.1, 0.2, 1.0])
    ap.add_argument("--params", nargs="+", default=list(PARAMS))
    ap.add_argument("--noise-mV", type=float, default=1.0)
    ap.add_argument("--n-obs", type=int, default=100, help="voltage samples per discharge curve")
    ap.add_argument("--t-min", type=float, default=100.0)
    ap.add_argument("--window", type=float, default=0.95, help="fraction of the nominal discharge time observed")
    ap.add_argument("--h-log", type=float, default=0.05)
    ap.add_argument("--h-volt", type=float, default=0.005)
    ap.add_argument("--Z-CC", type=float, default=0.025)
    ap.add_argument("--reversible", default="TFF", help="e.g. TFF (PINN benchmark) or TTF (paper)")
    ap.add_argument("--N-c", type=int, default=20)
    ap.add_argument("--N-s", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results" / "lispan" / "identifiability"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rev = tuple(c.upper() == "T" for c in args.reversible)
    base = replace(LiSPANParams(), c_DL=1e-6, reversible=rev, Z_CC=args.Z_CC)
    names = list(args.params)
    units = {n: ("volt" if PARAMS[n][0] == "add" else "rel") for n in names}
    sigma = args.noise_mV * 1e-3
    t0 = time.time()
    per_rate, sens = {}, {}
    for cr in args.rates:
        prot = LiSPANProtocol.from_crate(cr, ramp_s=30.0)
        _, ref = solve(base, prot, N_c=args.N_c, N_s=args.N_s, n_out=600)
        t_end0 = float(ref["t"][-1])
        t_obs = np.linspace(args.t_min, args.window * t_end0, args.n_obs)
        V0 = np.interp(t_obs, ref["t"], ref["V"])
        Q_obs = np.interp(t_obs, ref["t"], ref["Q_mAh_gS"])
        S = np.zeros((len(names), len(t_obs)))
        for i, n in enumerate(names):
            h = args.h_volt if units[n] == "volt" else args.h_log
            Vp, _ = voltage_on(perturbed(base, n, +h), prot, t_obs, args.N_c, args.N_s)
            Vm, _ = voltage_on(perturbed(base, n, -h), prot, t_obs, args.N_c, args.N_s)
            S[i] = (Vp - Vm) / (2.0 * h)
            print(f"{cr:5.2f} C  {n:10s} rms dV/dp {np.sqrt(np.mean(S[i] ** 2)) * 1e3:9.3f} mV per unit  "
                  f"({time.time() - t0:6.0f} s)", flush=True)
        sens[cr] = {"t": t_obs, "Q": Q_obs, "V": V0, "S": S, "t_end": t_end0}
        per_rate[cr] = fisher_report(S, names, sigma, units)
        np.savez(out / f"sensitivities_{cr:g}C.npz", t=t_obs, Q_mAh_gS=Q_obs, V=V0, S=S, names=np.array(names))
    combos = {}
    rates = list(args.rates)
    sets = [[r] for r in rates] + ([[min(rates), max(rates)]] if len(rates) > 1 else []) + ([rates] if len(rates) > 2 else [])
    for rs in sets:
        S = np.concatenate([sens[r]["S"] for r in rs], axis=1)
        combos["+".join(f"{r:g}C" for r in rs)] = fisher_report(S, names, sigma, units)

    # reduced sets: U0_2, U0_3 are exactly confounded with k0_2, k0_3 when reactions 2, 3 are irreversible
    reduced = [n for n in names if not (n in ("U0_2", "U0_3") and not rev[int(n[-1]) - 1])]
    red_report = {}
    if reduced != names:
        idx = [names.index(n) for n in reduced]
        ured = {n: units[n] for n in reduced}
        for rs in sets:
            S = np.concatenate([sens[r]["S"][idx] for r in rs], axis=1)
            red_report["+".join(f"{r:g}C" for r in rs)] = fisher_report(S, reduced, sigma, ured)

    summary = {"model": {"reversible": list(rev), "Z_CC": args.Z_CC, "c_DL": 1e-6, "ramp_s": 30.0,
                         "N_c": args.N_c, "N_s": args.N_s},
               "observation": {"noise_mV": args.noise_mV, "n_obs_per_curve": args.n_obs, "t_min": args.t_min,
                               "window": args.window},
               "params": names, "units": {n: ("%" if units[n] == "rel" else "mV") for n in names},
               "rms_sensitivity_mV": {f"{r:g}C": {n: float(np.sqrt(np.mean(sens[r]["S"][i] ** 2)) * 1e3)
                                                  for i, n in enumerate(names)} for r in rates},
               "combined": combos, "reduced_params": reduced, "reduced": red_report,
               "wall_s": time.time() - t0}
    (out / "identifiability.json").write_text(json.dumps(summary, indent=1))

    # text report
    lines = [f"Li-SPAN local identifiability: noise {args.noise_mV} mV, {args.n_obs} samples per curve, "
             f"window t in [{args.t_min:.0f} s, {args.window} t_end]; reversible {args.reversible}, Z_CC {args.Z_CC}"]
    lines.append("rms sensitivity [mV per unit: per e-fold for log-parameters, per V for U0]:")
    lines.append("param      " + "".join(f"{r:>10g}C" for r in rates))
    for i, n in enumerate(names):
        lines.append(f"{n:10s} " + "".join(f"{np.sqrt(np.mean(sens[r]['S'][i] ** 2)) * 1e3:11.2f}" for r in rates))
    for title, rep, nm in (("all parameters", combos, names), ("reduced (U0 of irreversible reactions fixed)", red_report, reduced)):
        if not rep:
            continue
        lines.append(f"\nCRLB, {title} (% for log-parameters, mV for U0):")
        keys = list(rep)
        lines.append("param      " + "".join(f"{k:>16s}" for k in keys))
        for n in nm:
            lines.append(f"{n:10s} " + "".join(f"{rep[k]['crlb'][n]:16.4g}" for k in keys))
        lines.append("condition number: " + ", ".join(f"{k} {rep[k]['cond']:.3g}" for k in keys))
        k_all = keys[-1]
        lines.append(f"strongest correlations ({k_all}): " +
                     "; ".join(f"{a}-{b} {v:+.3f}" for a, b, v in strongest_pairs(rep[k_all]["corr"], nm)))
    txt = "\n".join(lines)
    (out / "identifiability.txt").write_text(txt + "\n")
    print(txt)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(rates), figsize=(4.2 * len(rates), 4.2), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, r in zip(axes, rates):
            for i, n in enumerate(names):
                ax.plot(sens[r]["Q"], sens[r]["S"][i] * (1e3 if units[n] == "rel" else 10.0), label=n, lw=1)
            ax.set_title(f"{r:g} C")
            ax.set_xlabel("Q [mAh/g S]")
            ax.axhline(0, color="k", lw=0.5)
        axes[0].set_ylabel("dV/dp [mV per e-fold; mV per 10 mV for U0]")
        axes[-1].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "sensitivities.png", dpi=130)
    except Exception as exc:  # plotting is optional
        print("plot skipped:", exc)


if __name__ == "__main__":
    sys.exit(main())
