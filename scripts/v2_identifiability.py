"""Local identifiability of DFN parameters from voltage data (PyBaMM, no PINN).

For a set of parameters p_i the script computes the sensitivities
S_ij = dV(t_j)/d log p_i by central finite differences of the PyBaMM DFN
(same cell and ramp protocol as the v2 benchmark), then

* the Fisher information F = S S^T / sigma^2 for Gaussian voltage noise sigma,
* the Cramer-Rao lower bound on the relative standard deviation of each
  parameter, sqrt((F^-1)_ii),
* the parameter correlation matrix,
* the singular values of the (noise-normalized) sensitivity matrix,

for one C-rate and for a set of C-rates combined. These are LOCAL,
linearized statements around the true parameters: they identify weak and
correlated directions, they do not prove global uniqueness.

    python scripts/v2_identifiability.py --rates 0.5 1 2 --noise-mV 1 --out results/v2_identifiability
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.v2.params import CellParams, Protocol  # noqa: E402
from dfn_pinn.v2 import reference  # noqa: E402

# parameter -> how to apply a multiplier m to the PyBaMM parameter set
SCALARS = {
    "D_n": "Negative particle diffusivity [m2.s-1]",
    "D_p": "Positive particle diffusivity [m2.s-1]",
    "sigma_p": "Positive electrode conductivity [S.m-1]",
    "sigma_n": "Negative electrode conductivity [S.m-1]",
    "R_p": "Positive particle radius [m]",
    "R_n": "Negative particle radius [m]",
    "t_plus": "Cation transference number",
    # aging parameters (step C): initial stoichiometries (LLI) and active-material fractions (LAM)
    "theta_n0": "Initial concentration in negative electrode [mol.m-3]",
    "theta_p0": "Initial concentration in positive electrode [mol.m-3]",
    "eps_am_n": "Negative electrode active material volume fraction",
    "eps_am_p": "Positive electrode active material volume fraction",
}
FUNCTIONS = {  # multiplier applied to a parameter function
    "k_n": "Negative electrode exchange-current density [A.m-2]",
    "k_p": "Positive electrode exchange-current density [A.m-2]",
    "D_e": "Electrolyte diffusivity [m2.s-1]",
    "kappa_e": "Electrolyte conductivity [S.m-1]",
}


def scaled_parameters(cell, protocol, multipliers):
    import pybamm

    p = reference.parameter_values(cell, protocol)
    for name, m in multipliers.items():
        if abs(m - 1.0) < 1e-15:
            continue
        if name in SCALARS:
            p[SCALARS[name]] = p[SCALARS[name]] * m
        elif name in FUNCTIONS:
            key = FUNCTIONS[name]
            f = p[key]

            def scaled(*args, f=f, m=m):
                return m * f(*args)
            p[key] = scaled
        else:
            raise KeyError(name)
    return p


def voltage(cell, protocol, multipliers, times, mesh):
    import pybamm

    model = pybamm.lithium_ion.DFN()
    sim = pybamm.Simulation(model, parameter_values=scaled_parameters(cell, protocol, multipliers),
                            var_pts=mesh, solver=pybamm.CasadiSolver(mode="safe", rtol=1e-8, atol=1e-10))
    sol = sim.solve(times)
    V = np.asarray(sol["Voltage [V]"].entries)
    if len(V) < len(times):          # hit a voltage cut-off: pad with the last value
        V = np.concatenate([V, np.full(len(times) - len(V), V[-1])])
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", nargs="*", default=["D_n", "D_p", "k_n", "k_p", "sigma_p", "D_e", "kappa_e"])
    ap.add_argument("--rates", nargs="*", type=float, default=[1.0])
    ap.add_argument("--noise-mV", type=float, default=1.0)
    ap.add_argument("--dlog", type=float, default=0.05, help="half step in log p for central differences")
    ap.add_argument("--dt", type=float, default=10.0, help="sampling interval of the voltage data [s]")
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--nr", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "results" / "v2_identifiability"))
    ap.add_argument("--add-R0", action="store_true", help="append a lumped series resistance: dV per 1 mOhm = -I(t)")
    ap.add_argument("--t-end", type=float, default=None, help="protocol duration [s] (default 3000/rate, max 7000)")
    ap.add_argument("--t-min", type=float, default=0.0, help="first sampled time [s] (data window)")
    args = ap.parse_args()

    cell = CellParams()
    mesh = {"x_n": args.nx, "x_s": max(args.nx // 2, 10), "x_p": args.nx, "r_n": args.nr, "r_p": args.nr}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sigma = args.noise_mV * 1e-3
    t0 = time.perf_counter()
    S_all, V0_all, report = [], {}, {"params": list(args.params) + (["R0"] if args.add_R0 else []), "noise_mV": args.noise_mV, "dlog": args.dlog,
                                     "dt_s": args.dt, "rates": {}}
    for rate in args.rates:
        protocol = Protocol(current_A=5.0 * rate, ramp_s=30.0, t_end_s=args.t_end or min(3000.0 / rate, 7000.0))
        times = np.arange(args.t_min, protocol.t_end_s + 1e-9, args.dt)
        V0 = voltage(cell, protocol, {}, times, mesh)
        names = list(args.params) + (["R0"] if args.add_R0 else [])
        S = np.zeros((len(names), len(times)))
        if args.add_R0:   # linear parameter: sensitivity per 1 mOhm (the CRLB below is then in mOhm)
            S[-1] = -1e-3 * protocol.current(times)
        for i, name in enumerate(args.params):
            Vp = voltage(cell, protocol, {name: np.exp(args.dlog)}, times, mesh)
            Vm = voltage(cell, protocol, {name: np.exp(-args.dlog)}, times, mesh)
            S[i] = (Vp - Vm) / (2 * args.dlog)
            print(f"rate {rate:g}C  {name:8s}  max |dV/dlog p| = {1e3 * np.abs(S[i]).max():7.2f} mV   "
                  f"rms = {1e3 * np.sqrt((S[i] ** 2).mean()):6.2f} mV   ({time.perf_counter() - t0:.0f} s)", flush=True)
        S_all.append(S)
        V0_all[rate] = V0
        F = S @ S.T / sigma ** 2
        report["rates"][str(rate)] = analyse(F, S, names, sigma, label=f"{rate:g}C alone")
        np.savez(out / f"sensitivities_{rate:g}C.npz", t=times, V=V0, S=S, params=np.array(names))
    if len(args.rates) > 1:
        S = np.concatenate(S_all, axis=1)
        F = S @ S.T / sigma ** 2
        report["combined"] = analyse(F, S, names, sigma, label="all rates combined")
    (out / "identifiability.json").write_text(json.dumps(report, indent=1))
    print(f"\nSaved {out / 'identifiability.json'}")


def analyse(F, S, params, sigma, label):
    n = len(params)
    cov = np.linalg.pinv(F)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))                 # std of log p  (= relative std of p)
    corr = cov / np.outer(std, std)
    sv = np.linalg.svd(S / sigma, compute_uv=False)
    cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
    eigval, eigvec = np.linalg.eigh(F)
    weakest = eigvec[:, 0]
    print(f"\n=== {label}: {S.shape[1]} voltage samples, noise {1e3 * sigma:g} mV ===")
    print(f"{'parameter':10s} {'CRLB rel. std':>14s}   {'max |dV/dlog p| [mV]':>22s}")
    for i, name in enumerate(params):
        print(f"{name:10s} {100 * std[i]:13.2f} %   {1e3 * np.abs(S[i]).max():22.2f}")
    print("correlation matrix:")
    print("           " + " ".join(f"{p:>8s}" for p in params))
    for i, name in enumerate(params):
        print(f"{name:10s} " + " ".join(f"{corr[i, j]:8.3f}" for j in range(n)))
    print(f"singular values of S/sigma: {np.array2string(sv, precision=3, max_line_width=120)}  (condition {cond:.3g})")
    print("weakest direction (eigenvector of smallest FIM eigenvalue): "
          + ", ".join(f"{p}:{w:+.2f}" for p, w in zip(params, weakest)))
    return {"crlb_rel_std": dict(zip(params, std.tolist())), "correlation": corr.tolist(),
            "singular_values": sv.tolist(), "condition": float(cond), "weakest_direction": dict(zip(params, weakest.tolist())),
            "max_abs_sensitivity_mV": dict(zip(params, (1e3 * np.abs(S).max(axis=1)).tolist())),
            "n_samples": int(S.shape[1])}


if __name__ == "__main__":
    main()
