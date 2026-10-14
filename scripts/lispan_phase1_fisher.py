"""Phase 1 (identifiability proposal): which Li-SPAN parameters does each set of discharge rates identify,
and how uncertain are the predictions at rates that were not measured?

Local (linearized) Bayesian analysis around a realistic truth, the calibrated Simanjuntak cell (the classical
fit to their measured 0.1 C + 1 C points, results/lispan/fit_experiment/fit_0.1+1C_fdabs.json):

* sensitivities S_ij = dV(t_j)/dp_i by central finite differences of the finite-volume model, per C-rate
  (0.05, 0.1, 0.2, 0.5, 1 C), 100 samples per curve on [0.02, 0.95] of the truth's discharge time;
* for a measurement set M (a list of rates), Gaussian voltage noise sigma and a Gaussian prior on p
  (sd ln 2 for log-parameters = "known within a factor of 2", 50 mV for step potentials), the linearized
  posterior covariance  Sigma_M = (sum_{r in M} S_r S_r^T / sigma^2 + P^-1)^-1;
* posterior sd / prior sd per parameter (1 = nothing learned), also for the combinations
  ln k0_m + (F / 2RT) U0_m of the irreversible steps (the only kinetic quantity a galvanostatic discharge sees);
* prediction sd of the voltage at every rate, sd_V(t) = sqrt(diag(S_r^T Sigma_M S_r)), summarized as rms over t.

Step 1 is reversible and steps 2-3 irreversible (PINN-benchmark model).  For an irreversible step the rate is
k0 exp(-F(phi - U0 + b xi) / 2RT), so (k0, U0) enter the rate only through k0 exp(F U0 / 2RT), at EVERY current
(U0_2, U0_3 also set the initial S2- equilibrium, a negligible effect).  The analysis includes U0_2, U0_3 to show
this explicitly.

    python scripts/lispan_phase1_fisher.py --jobs 2          # sensitivities (about 12 min on 2 cores), cached
    python scripts/lispan_phase1_fisher.py --analyze-only    # sets, priors, figures from the cached files
"""

import argparse
from dataclasses import replace
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dfn_pinn.lispan.params import LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.model import solve  # noqa: E402
from lispan_make_inverse_data import apply_truth  # noqa: E402
from lispan_identifiability import PARAMS, perturbed  # noqa: E402

F_2RT = 96485.0 / (2.0 * 8.314 * 298.15)          # 1/V, for the k0-U0 combination of an irreversible step
NAMES = ["U0_1", "b_1", "k0_1", "U0_2", "b_2", "k0_2", "U0_3", "b_3", "k0_3",
         "Z_CC", "kappa0", "D_salt", "t_plus", "kappa_SPAN"]
# 2 C is left out: for some perturbed parameter sets (b_2 +-5 %) the finite-volume solver stalls at 2 C, where this
# cell reaches the 1.0 V cut-off at about a quarter of its capacity.
RATES = [0.05, 0.1, 0.2, 0.5, 1.0]
SETS = {                                            # label -> rates measured
    "C/10": [0.1],
    "C/10 + C/5": [0.1, 0.2],
    "1C": [1.0],
    "C/10 + C/2": [0.1, 0.5],
    "C/10 + 1C": [0.1, 1.0],
    "all five rates": RATES,
}
OUT = ROOT / "results" / "lispan" / "phase1"
FIT_JSON = ROOT / "results" / "lispan" / "fit_experiment" / "fit_0.1+1C_fdabs.json"


def truth_params():
    est = json.loads(FIT_JSON.read_text())["estimates"]
    base = replace(LiSPANParams(), c_DL=1e-6, reversible=(True, False, False), Z_CC=0.025)
    return apply_truth(base, est), est


def v_on(p, crate, t_obs, N_c, N_s):
    prot = LiSPANProtocol.from_crate(crate, ramp_s=30.0)
    _, res = solve(p, prot, N_c=N_c, N_s=N_s, n_out=1500, t_end=float(t_obs[-1]) * 1.05 + 100.0)
    t, V = res["t"], res["V"]
    return np.where(t_obs <= t[-1], np.interp(t_obs, t, V), prot.V_min)


def sens_one(args):
    crate, h_log, h_volt, n_obs, N_c, N_s = args
    p, _ = truth_params()
    prot = LiSPANProtocol.from_crate(crate, ramp_s=30.0)
    _, res = solve(p, prot, N_c=N_c, N_s=N_s, n_out=1500)
    t_end = float(res["t"][-1])
    t_obs = np.linspace(0.02 * t_end, 0.95 * t_end, n_obs)
    V0 = v_on(p, crate, t_obs, N_c, N_s)
    S = np.zeros((len(NAMES), n_obs))
    t0 = time.time()
    for i, n in enumerate(NAMES):
        h = h_volt if PARAMS[n][0] == "add" else h_log
        Vp = v_on(perturbed(p, n, +h), crate, t_obs, N_c, N_s)
        Vm = v_on(perturbed(p, n, -h), crate, t_obs, N_c, N_s)
        S[i] = (Vp - Vm) / (2.0 * h)
    Q = float(prot.current) * t_obs / 3.6 / (1e3 * p.m_S_model)        # mAh per g S (ramp neglected)
    np.savez(OUT / f"sens_{crate:g}C.npz", S=S, t=t_obs, V=V0, t_end=t_end, names=np.array(NAMES), Q=Q)
    return crate, time.time() - t0


def prior_sd(log_sd=np.log(2.0), volt_sd=0.05):
    return np.array([volt_sd if PARAMS[n][0] == "add" else log_sd for n in NAMES])


def analyze(sigma_mV, log_sd, volt_sd):
    S = {cr: np.load(OUT / f"sens_{cr:g}C.npz") for cr in RATES}
    sig = sigma_mV * 1e-3
    psd = prior_sd(log_sd, volt_sd)
    Pinv = np.diag(1.0 / psd ** 2)
    i_k = {m: NAMES.index(f"k0_{m}") for m in (2, 3)}
    i_u = {m: NAMES.index(f"U0_{m}") for m in (2, 3)}
    combo = {}
    for m in (2, 3):                                 # c_m = ln k0_m + (F/2RT) U0_m
        a = np.zeros(len(NAMES)); a[i_k[m]] = 1.0; a[i_u[m]] = F_2RT
        combo[f"k0_{m}*exp(F U0_{m}/2RT)"] = a
    report = {"truth": "calibrated Simanjuntak cell (fit_0.1+1C_fdabs.json)", "sigma_mV": sigma_mV,
              "prior": {"log_sd": log_sd, "volt_sd_V": volt_sd}, "names": NAMES, "rates": RATES, "sets": {}}
    rms_sens = {f"{cr:g}C": {n: float(np.sqrt(np.mean(S[cr]["S"][i] ** 2)) * 1e3) for i, n in enumerate(NAMES)}
                for cr in RATES}
    report["rms_sensitivity_mV_per_unit"] = rms_sens
    for label, rates in SETS.items():
        Fm = sum(S[cr]["S"] @ S[cr]["S"].T for cr in rates) / sig ** 2
        cov = np.linalg.inv(Fm + Pinv)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        comb = {}
        for k, a in combo.items():
            prior_c = float(np.sqrt(a @ np.diag(psd ** 2) @ a))
            comb[k] = {"sd": float(np.sqrt(a @ cov @ a)), "prior_sd": prior_c, "ratio": float(np.sqrt(a @ cov @ a) / prior_c)}
        pred = {}
        for cr in RATES:
            Sr = S[cr]["S"]
            sdV = np.sqrt(np.einsum("it,ij,jt->t", Sr, cov, Sr))
            sdV0 = np.sqrt(np.einsum("it,ij,jt->t", Sr, np.diag(psd ** 2), Sr))
            pred[f"{cr:g}C"] = {"rms_mV": float(np.sqrt(np.mean(sdV ** 2)) * 1e3), "max_mV": float(sdV.max() * 1e3),
                                "prior_rms_mV": float(np.sqrt(np.mean(sdV0 ** 2)) * 1e3), "measured": cr in rates}
        iu, ik = NAMES.index("U0_2"), NAMES.index("k0_2")
        report["sets"][label] = {
            "rates": rates,
            "posterior_sd": {n: float(s) for n, s in zip(NAMES, sd)},
            "ratio_to_prior": {n: float(s / p) for n, s, p in zip(NAMES, sd, psd)},
            "combinations": comb,
            "prediction": pred,
            "corr_k0_2_U0_2": float(corr[iu, ik]),
            "corr_k0_3_U0_3": float(corr[NAMES.index("U0_3"), NAMES.index("k0_3")]),
            "corr_k0_1_U0_1": float(corr[NAMES.index("U0_1"), NAMES.index("k0_1")]),
            "corr_Z_CC_kappa0": float(corr[NAMES.index("Z_CC"), NAMES.index("kappa0")]),
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--h-log", type=float, default=0.05)
    ap.add_argument("--h-volt", type=float, default=0.005)
    ap.add_argument("--grid", nargs=2, type=int, default=[20, 10])
    ap.add_argument("--sigma-mV", type=float, default=5.0, help="voltage noise (digitized-data level)")
    ap.add_argument("--prior-log-sd", type=float, default=float(np.log(2.0)))
    ap.add_argument("--prior-volt-sd", type=float, default=0.05)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        todo = [(cr, args.h_log, args.h_volt, args.n_obs, *args.grid) for cr in RATES
                if not (OUT / f"sens_{cr:g}C.npz").exists()]
        with Pool(args.jobs) as pool:
            for cr, dt in pool.imap_unordered(sens_one, todo):
                print(f"sensitivities {cr:g} C done in {dt:.0f} s", flush=True)
    rep = analyze(args.sigma_mV, args.prior_log_sd, args.prior_volt_sd)
    tag = f"sigma{args.sigma_mV:g}mV"
    (OUT / f"fisher_{tag}.json").write_text(json.dumps(rep, indent=1))
    lines = [f"posterior sd / prior sd (sigma {args.sigma_mV} mV, prior factor {np.exp(args.prior_log_sd):.2g} / "
             f"{1000 * args.prior_volt_sd:.0f} mV); 1 = not identified"]
    lines.append("param".ljust(34) + "".join(s.rjust(18) for s in SETS))
    for n in NAMES:
        lines.append(n.ljust(34) + "".join(f"{rep['sets'][s]['ratio_to_prior'][n]:18.3f}" for s in SETS))
    for k in rep["sets"]["C/10"]["combinations"]:
        lines.append(k.ljust(34) + "".join(f"{rep['sets'][s]['combinations'][k]['ratio']:18.3f}" for s in SETS))
    lines.append("prediction rms sd [mV] (* = measured rate)")
    for cr in RATES:
        key = f"{cr:g}C"
        lines.append(key.ljust(34) + "".join(
            f"{rep['sets'][s]['prediction'][key]['rms_mV']:17.1f}{'*' if rep['sets'][s]['prediction'][key]['measured'] else ' '}"
            for s in SETS))
    lines.append("corr(k0_2, U0_2): " + ", ".join(f"{s} {rep['sets'][s]['corr_k0_2_U0_2']:+.3f}" for s in SETS))
    lines.append("corr(k0_3, U0_3): " + ", ".join(f"{s} {rep['sets'][s]['corr_k0_3_U0_3']:+.3f}" for s in SETS))
    lines.append("corr(k0_1, U0_1): " + ", ".join(f"{s} {rep['sets'][s]['corr_k0_1_U0_1']:+.3f}" for s in SETS))
    txt = "\n".join(lines)
    (OUT / f"fisher_{tag}.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
