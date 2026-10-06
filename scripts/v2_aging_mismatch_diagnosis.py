"""Which degradation mechanism makes the aged 1C discharge unrepresentable by the plain DFN with
the aged (theta0, eps_am, porosity) state?  (V2_RESULTS.md section 8)

For each variant of the O'Kane 2022 degradation model (mechanisms switched off one at a time), cycle
N times with the LG M50 protocol, take the last CC discharge and its true aged state, run the plain
DFN (dfn_pinn.v2.reference) from that state and report the voltage mismatch (rms / max, best lumped
R0, residual after R0).

    python scripts/v2_aging_mismatch_diagnosis.py --cycles 60 --coarse
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.v2.params import CellParams, Protocol  # noqa: E402
from dfn_pinn.v2.reference import solve  # noqa: E402

BASE = {"SEI": "solvent-diffusion limited", "SEI porosity change": "true",
        "lithium plating": "partially reversible", "lithium plating porosity change": "true",
        "particle mechanics": "swelling only", "SEI on cracks": "false",
        "loss of active material": "stress-driven"}
VARIANTS = {
    "all (no cracking)": {},
    "no lithium plating": {"lithium plating": "none", "lithium plating porosity change": "false"},
    "SEI without porosity change": {"SEI porosity change": "false"},
    "no SEI": {"SEI": "none", "SEI porosity change": "false"},
    "no LAM": {"loss of active material": "none"},
    "no plating, no SEI porosity change": {"lithium plating": "none", "lithium plating porosity change": "false",
                                           "SEI porosity change": "false"},
    "all, stress-induced diffusion off": {"stress-induced diffusion": "false"},
    "no mechanics (no swelling, no LAM)": {"particle mechanics": "none", "loss of active material": "none"},
    "SEI only": {"particle mechanics": "none", "loss of active material": "none", "lithium plating": "none",
                 "lithium plating porosity change": "false"},
    "SEI only, no porosity change": {"particle mechanics": "none", "loss of active material": "none", "lithium plating": "none",
                                     "lithium plating porosity change": "false", "SEI porosity change": "false"},
    "none (plain DFN, OKane2022)": {"particle mechanics": "none", "loss of active material": "none", "lithium plating": "none",
                                    "lithium plating porosity change": "false", "SEI": "none", "SEI porosity change": "false"},
}


def run_variant(options, args, param):
    import pybamm
    model = pybamm.lithium_ion.DFN(options)
    var_pts = {"x_n": 10, "x_s": 5, "x_p": 10, "r_n": 15, "r_p": 15} if args.coarse else \
              {"x_n": 20, "x_s": 10, "x_p": 20, "r_n": 30, "r_p": 30}
    cycle = ["Charge at 1C until 4.2 V", "Hold at 4.2 V until C/100", "Rest for 5 minutes",
             "Discharge at 1C until 2.5 V", "Hold at 2.5 V until C/100", "Rest for 5 minutes"]
    exp = pybamm.Experiment([tuple(cycle)] * args.cycles)
    sim = pybamm.Simulation(model, parameter_values=param, experiment=exp,
                            solver=pybamm.CasadiSolver(mode="safe", dt_max=60), var_pts=var_pts)
    sol = sim.solve()
    cyc = sol.cycles[-1]
    cand = []
    for st in cyc.steps:
        try:
            Ist = np.asarray(st["Current [A]"].entries); tst = np.asarray(st["Time [s]"].entries)
        except TypeError:
            continue
        if len(tst) > 2 and Ist.mean() > 0.5 and Ist.std() < 0.05 * abs(Ist.mean()):
            cand.append((abs(np.trapezoid(Ist, tst)), st))
    dis = max(cand, key=lambda c: c[0])[1]
    t = np.asarray(dis["Time [s]"].entries); t = t - t[0]
    V = np.asarray(dis["Voltage [V]"].entries)
    cmax_n = param["Maximum concentration in negative electrode [mol.m-3]"]
    cmax_p = param["Maximum concentration in positive electrode [mol.m-3]"]
    cn = np.asarray(dis["X-averaged negative particle concentration [mol.m-3]"].entries)
    cp = np.asarray(dis["X-averaged positive particle concentration [mol.m-3]"].entries)
    r = np.linspace(0, 1, cn.shape[0]); w = r ** 2; w = w / w.sum()
    state = {"theta_n0": float((cn[:, 0] * w).sum() / cmax_n), "theta_p0": float((cp[:, 0] * w).sum() / cmax_p),
             "eps_am_n": float(np.asarray(dis["X-averaged negative electrode active material volume fraction"].entries)[0]),
             "eps_am_p": float(np.asarray(dis["X-averaged positive electrode active material volume fraction"].entries)[0]),
             "porosity_n": float(np.asarray(dis["X-averaged negative electrode porosity"].entries)[0])}
    try:
        state["LLI_pct"] = float(np.asarray(sol.summary_variables["Loss of lithium inventory [%]"])[-1])
    except Exception:
        pass
    return t, V, state


def mismatch(t_cc, V_cc, cell, ramp_s=30.0):
    prot = Protocol(current_A=5.0, ramp_s=ramp_s, t_end_s=float(t_cc[-1]))
    times = np.linspace(0, t_cc[-1], 300)
    _, sol, _ = solve(cell, prot, mesh={"x_n": 20, "x_s": 10, "x_p": 20, "r_n": 30, "r_p": 30}, times=times, rtol=1e-6, atol=1e-8)
    V = np.asarray(sol["Voltage [V]"].entries); tt = np.asarray(sol["Time [s]"].entries)
    Vd = np.interp(tt, t_cc, V_cc); m = tt >= 100
    e = (V - Vd)[m]; I_hat = 5.0 * np.tanh(tt[m] / 30.0)
    R0 = float(np.sum(e * I_hat) / np.sum(I_hat ** 2)); e2 = e - R0 * I_hat
    return 1e3 * np.sqrt(np.mean(e ** 2)), 1e3 * np.abs(e).max(), 1e3 * R0, 1e3 * np.sqrt(np.mean(e2 ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=60)
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--scale-dead-li", type=float, default=5.0)
    ap.add_argument("--scale-lam", type=float, default=20.0)
    ap.add_argument("--scale-sei", type=float, default=20.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--ramp", type=float, default=30.0, help="tanh current ramp of the plain-DFN reference [s]; the "
                    "data have a current step, so 30 s lags the data by 20.8 s of charge (section 8 of V2_RESULTS.md)")
    ap.add_argument("--save", default=None, help="save the extracted CC discharges and states to this npz")
    args = ap.parse_args()
    import pybamm
    fresh = CellParams()
    saved = {}
    for name, delta in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        param = pybamm.ParameterValues("OKane2022")
        for key, fac in {"Dead lithium decay rate [s-1]": args.scale_dead_li,
                         "Negative electrode LAM constant proportional term [s-1]": args.scale_lam,
                         "Positive electrode LAM constant proportional term [s-1]": args.scale_lam,
                         "SEI solvent diffusivity [m2.s-1]": args.scale_sei}.items():
            try:
                base = param[key]
            except KeyError:
                continue
            if callable(base):      # e.g. the SEI solvent diffusivity is a function of temperature in OKane2022
                param.update({key: (lambda f, b: (lambda *a, **k: f * b(*a, **k)))(fac, base)})
            else:
                param.update({key: fac * base})
        options = dict(BASE); options.update(delta)
        t0 = time.perf_counter()
        try:
            t_cc, V_cc, st = run_variant(options, args, param)
        except Exception as exc:   # an option combination PyBaMM rejects
            print(f"{name:36s}: FAILED ({str(exc)[:80]})", flush=True); continue
        aged = CellParams(c_n0=st["theta_n0"] * fresh.cmax_n, c_p0=st["theta_p0"] * fresh.cmax_p,
                          eps_am_n=st["eps_am_n"], eps_am_p=st["eps_am_p"])
        aged_por = CellParams(c_n0=st["theta_n0"] * fresh.cmax_n, c_p0=st["theta_p0"] * fresh.cmax_p,
                              eps_am_n=st["eps_am_n"], eps_am_p=st["eps_am_p"], eps_n=st["porosity_n"])
        r1 = mismatch(t_cc, V_cc, aged, args.ramp); r2 = mismatch(t_cc, V_cc, aged_por, args.ramp)
        if args.save:
            saved[name] = {"t": t_cc, "V": V_cc, "state": st}
        print(f"{name:36s}: cycle {args.cycles}, CC {t_cc[-1]:.0f} s, LLI {st.get('LLI_pct', float('nan')):.2f} %, "
              f"theta0 {st['theta_n0']:.4f}/{st['theta_p0']:.4f}, eps_am rel {st['eps_am_n'] / fresh.eps_am_n:.4f}/{st['eps_am_p'] / fresh.eps_am_p:.4f}, "
              f"porosity_n {st['porosity_n']:.4f} | plain DFN(theta0, eps_am): {r1[0]:.1f} mV rms, {r1[1]:.1f} max, best R0 {r1[2]:+.2f} mOhm -> {r1[3]:.1f} | "
              f"+porosity: {r2[0]:.1f} rms, R0 {r2[2]:+.2f} -> {r2[3]:.1f}   ({time.perf_counter() - t0:.0f} s)", flush=True)
    if args.save and saved:
        import json
        np.savez(args.save, **{f"{k}__t": v["t"] for k, v in saved.items()}, **{f"{k}__V": v["V"] for k, v in saved.items()},
                 states=json.dumps({k: v["state"] for k, v in saved.items()}))


if __name__ == "__main__":
    main()
