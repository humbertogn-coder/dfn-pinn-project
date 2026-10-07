"""Synthetic aging data set for the "commercial cell" step (stand-in for the LG M50 degradation data).

PyBaMM DFN (Chen2020 / OKane2022 parameters) coupled with the four degradation mechanisms of
O'Kane et al. 2022 (SEI growth, particle cracking with SEI on cracks, lithium plating, stress-driven
loss of active material), cycled with the protocol of the LG M50 data set description: 1C CC-CV charge
to 4.2 V (C/100 cut-off), 5 min rest, 1C CC-CV discharge to 2.5 V (C/100 cut-off), 5 min rest, 25 C.

    python scripts/v2_make_aging_dataset.py --cycles 10 --save-every 3 --out results/aging_synthetic/cellA

For every saved cycle the CC part of the discharge step is written as cycles/cycle_XXXX.npz with
t [s] (from the start of the step), I [A] (> 0 discharge), V [V], plus the ground truth of the aged
state that the inverse PINN estimates: theta_n0 / theta_p0 (x- and r-averaged stoichiometries at the
start of the discharge), eps_am_n / eps_am_p (active-material volume fractions), and the summary
variables (LLI, LAM, capacity, SEI thickness, plated lithium). summary.csv collects them per cycle.
The user's real data set is read through the same npz layout (src/dfn_pinn/v2/aging.py).
"""

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=1, help="save the discharge of every k-th cycle (and the last)")
    ap.add_argument("--out", default=str(ROOT / "results" / "aging_synthetic" / "cellA"))
    ap.add_argument("--coarse", action="store_true", help="coarser mesh (faster, for pipeline tests)")
    ap.add_argument("--scale-cracking", type=float, default=1.0, help="x negative electrode cracking rate (Paris' law)")
    ap.add_argument("--scale-dead-li", type=float, default=1.0, help="x dead lithium decay rate")
    ap.add_argument("--scale-lam", type=float, default=1.0, help="x LAM constant proportional terms (both electrodes)")
    ap.add_argument("--scale-sei", type=float, default=1.0, help="x SEI solvent diffusivity (solvent-diffusion limited SEI)")
    ap.add_argument("--no-cracking", action="store_true",
                    help="swelling only, no cracks / SEI on cracks")
    ap.add_argument("--charge-crate", type=float, default=1.0,
                    help="CC charge rate of the cycling protocol (the LG M50 data set uses 1C; without stress-induced "
                         "diffusion the 1C charge saturates the graphite surface and the DAE solver fails -> use 0.5)")
    ap.add_argument("--no-stress-diffusion", action="store_true",
                    help="switch off PyBaMM's stress-induced diffusion (on by default with particle mechanics; it "
                         "multiplies the NMC diffusivity by 100-400 and is what made the aged cells unrepresentable by "
                         "the 5-parameter aging DFN, V2_RESULTS.md section 8)")
    args = ap.parse_args()
    import pybamm

    out = Path(args.out)
    (out / "cycles").mkdir(parents=True, exist_ok=True)
    options = {"SEI": "solvent-diffusion limited", "SEI porosity change": "true",
               "lithium plating": "partially reversible", "lithium plating porosity change": "true",
               "particle mechanics": ("swelling and cracking", "swelling only"), "SEI on cracks": "true",
               "loss of active material": "stress-driven", "calculate discharge energy": "true"}
    if args.no_cracking:
        options["particle mechanics"] = "swelling only"
        options["SEI on cracks"] = "false"
    if args.no_stress_diffusion:
        options["stress-induced diffusion"] = "false"
    model = pybamm.lithium_ion.DFN(options)
    param = pybamm.ParameterValues("OKane2022")
    # the LG M50 data set varied the cracking rate, the dead-lithium decay rate and the LAM proportional term
    scaled = {"Negative electrode cracking rate": args.scale_cracking,
              "Dead lithium decay rate [s-1]": args.scale_dead_li,
              "Negative electrode LAM constant proportional term [s-1]": args.scale_lam,
              "Positive electrode LAM constant proportional term [s-1]": args.scale_lam,
              "SEI solvent diffusivity [m2.s-1]": args.scale_sei}
    for key, fac in scaled.items():
        if fac != 1.0:
            try:
                base = param[key]
                param.update({key: (lambda f, b: (lambda *a, **k: f * b(*a, **k)))(fac, base) if callable(base) else fac * base})
                print(f"{key}: x{fac}")
            except KeyError:
                print(f"WARNING: parameter {key!r} not in OKane2022, not scaled")
    var_pts = {"x_n": 10, "x_s": 5, "x_p": 10, "r_n": 15, "r_p": 15} if args.coarse else \
              {"x_n": 20, "x_s": 10, "x_p": 20, "r_n": 30, "r_p": 30}
    cycle = [f"Charge at {args.charge_crate:g}C until 4.2 V", "Hold at 4.2 V until C/100", "Rest for 5 minutes",
             "Discharge at 1C until 2.5 V", "Hold at 2.5 V until C/100", "Rest for 5 minutes"]
    exp = pybamm.Experiment([tuple(cycle)] * args.cycles)     # one tuple = one cycle (6 steps)
    solver = pybamm.CasadiSolver(mode="safe", dt_max=60)
    sim = pybamm.Simulation(model, parameter_values=param, experiment=exp, solver=solver, var_pts=var_pts)
    t0 = time.perf_counter()
    sol = sim.solve()
    print(f"solved {args.cycles} cycles in {time.perf_counter() - t0:.0f} s")

    cmax_n = param["Maximum concentration in negative electrode [mol.m-3]"]
    cmax_p = param["Maximum concentration in positive electrode [mol.m-3]"]
    eps_n0 = param["Negative electrode active material volume fraction"]
    eps_p0 = param["Positive electrode active material volume fraction"]
    rows = []
    for i, cyc in enumerate(sol.cycles, start=1):
        # summary variables of this cycle (PyBaMM >= 24: dict-like per cycle)
        sv = {}
        try:
            for key in ["Capacity [A.h]", "Loss of lithium inventory [%]",
                        "Loss of active material in negative electrode [%]",
                        "Loss of active material in positive electrode [%]",
                        "Total lithium lost [mol]", "Loss of capacity to negative SEI [A.h]",
                        "Loss of capacity to negative lithium plating [A.h]"]:
                try:
                    v = sol.summary_variables[key]
                    sv[key] = float(np.asarray(v)[i - 1])
                except Exception:
                    pass
        except Exception:
            pass
        # CC discharge step = the step with discharge current (> 0 in PyBaMM) and the largest charge throughput
        # (the first 'Charge' step is skipped by PyBaMM when the cell starts full, so positions are not fixed)
        cand = []
        for st in cyc.steps:
            if getattr(st, "t", None) is None or len(getattr(st, "t", [])) == 0:
                continue                                             # skipped (infeasible) step
            try:
                Ist = np.asarray(st["Current [A]"].entries); tst = np.asarray(st["Time [s]"].entries)
            except TypeError:
                continue
            if len(tst) > 2 and Ist.mean() > 0.5 and Ist.std() < 0.05 * abs(Ist.mean()):
                cand.append((abs(np.trapezoid(Ist, tst)), st))
        if not cand:
            print(f"cycle {i}: no CC discharge step found, skipped"); continue
        dis = max(cand, key=lambda c: c[0])[1]
        t = np.asarray(dis["Time [s]"].entries); t = t - t[0]
        I = np.asarray(dis["Current [A]"].entries)
        V = np.asarray(dis["Voltage [V]"].entries)
        cn = np.asarray(dis["X-averaged negative particle concentration [mol.m-3]"].entries)   # (r, t)
        cp = np.asarray(dis["X-averaged positive particle concentration [mol.m-3]"].entries)
        # r-average with volume weights on the (uniform) r mesh: approximate by mean of r^2-weighted values
        r = np.linspace(0, 1, cn.shape[0]); w = r ** 2; w = w / w.sum()
        th_n0 = float((cn[:, 0] * w).sum() / cmax_n); th_p0 = float((cp[:, 0] * w).sum() / cmax_p)
        eps_n = float(np.asarray(dis["X-averaged negative electrode active material volume fraction"].entries)[0])
        eps_p = float(np.asarray(dis["X-averaged positive electrode active material volume fraction"].entries)[0])
        sei = float("nan")
        for name in ["X-averaged negative total SEI thickness [m]", "X-averaged negative SEI thickness [m]",
                     "X-averaged total SEI thickness [m]", "X-averaged SEI thickness [m]"]:
            try:
                sei = float(np.asarray(dis[name].entries)[0]); break
            except Exception:
                continue
        extra = {}   # state variables the 5-parameter aging DFN does not contain (model-form mismatch diagnostics)
        for key, name in [("a_n_ratio", "X-averaged negative electrode surface area to volume ratio [m-1]"),
                          ("a_p_ratio", "X-averaged positive electrode surface area to volume ratio [m-1]"),
                          ("roughness_n", "X-averaged negative electrode roughness ratio"),
                          ("porosity_n", "X-averaged negative electrode porosity"),
                          ("porosity_p", "X-averaged positive electrode porosity")]:
            try:
                val = float(np.asarray(dis[name].entries)[0])
                if key.startswith("a_"):
                    base = 3 * (eps_n0 if key == "a_n_ratio" else eps_p0) / param["Negative particle radius [m]" if key == "a_n_ratio" else "Positive particle radius [m]"]
                    val = val / base
                extra[key] = val
            except Exception:
                pass
        row = {"cycle": i, "t_cc_s": float(t[-1]), "Q_cc_Ah": float(np.trapezoid(I, t) / 3600),
               "theta_n0": th_n0, "theta_p0": th_p0, "eps_am_n": eps_n, "eps_am_p": eps_p,
               "eps_am_n_rel": eps_n / eps_n0, "eps_am_p_rel": eps_p / eps_p0, "SEI_thickness_m": sei, **extra, **sv}
        rows.append(row)
        if i % args.save_every == 0 or i == args.cycles or i == 1:
            np.savez(out / "cycles" / f"cycle_{i:04d}.npz", t=t, I=I, V=V, cycle=i, theta_n0=th_n0, theta_p0=th_p0,
                     eps_am_n=eps_n, eps_am_p=eps_p, SEI_thickness_m=sei, **extra,
                     **{k.replace(" ", "_"): v for k, v in sv.items()})
        print(f"cycle {i:4d}: CC discharge {t[-1]:.0f} s, {row['Q_cc_Ah']:.3f} Ah, theta0 n/p {th_n0:.4f}/{th_p0:.4f}, "
              f"eps_am rel n/p {row['eps_am_n_rel']:.4f}/{row['eps_am_p_rel']:.4f}, LLI {sv.get('Loss of lithium inventory [%]', float('nan')):.3f} %")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "cycle", k))
    with open(out / "summary.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=keys); wri.writeheader(); wri.writerows(rows)
    print("written", out)


if __name__ == "__main__":
    main()
