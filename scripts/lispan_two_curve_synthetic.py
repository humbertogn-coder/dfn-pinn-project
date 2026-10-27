"""Known-truth test of the two-curve tool with w_S free (docs/TWO_CURVE_TOOL.md, "Synthetic test").

FV discharges of the evaluation chemistries of the v2 design are generated at two interior model rates (default 0.2 +
0.6 C, so that the network's rate range does not pin w_S) with w_S, the common U0 shift d and Z_CC away from their
nominal values, and fitted exactly like measured data (scripts/lispan_two_curve_fit.py: capacity basis, 80 points up to
0.90 of the end capacity, 8 starts).  The fitted parameters are compared with the truth, noise-free (error caused by
the network alone) and with Gaussian voltage noise (error against the tool's own linearized sd).

    python scripts/lispan_two_curve_synthetic.py --workers 2
    python scripts/lispan_two_curve_synthetic.py --cases p0 p5 --noise-mV 0 --workers 1

Truth: chemistry inputs of the design's evaluation cases; w_S alternates 0.8 / 1.25 between cases; d = +20 mV;
Z_CC = 0.002 Ohm m2 (20 Ohm cm2).  FV grid 20 / 10 volumes (<= 0.08 mV from 40 / 20, docs/PAPER_ASSUMPTIONS.md).
Errors in the units of the tool's sd: w_S, b, k0 in % (log difference x 100), dU0 and d in mV, Z_CC in Ohm cm2.
Output: results/lispan/two_curve/synthetic_<rates>.json.
"""

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dfn_pinn.lispan.chem_pinn import chem_vector, load_chem_model  # noqa: E402
from dfn_pinn.lispan.rate_pinn import ONE_C  # noqa: E402
from dfn_pinn.lispan.two_curve import Curve, TwoCurveFit, fv_curve, m_S_g  # noqa: E402

NETWORK = ROOT / "results" / "lispan_chem_runs" / "chem_v2_norar_s0" / "final.pt"
OUT = ROOT / "results" / "lispan" / "two_curve"
_MODEL = {}


def _init(network):
    torch.set_num_threads(1)
    _MODEL["m"] = load_chem_model(network)[0]


def scale(name):
    if name == "d" or name.startswith("dU0_"):
        return 1e3
    if name == "Z_CC":
        return 1e4
    return 100.0


def run_case(job):
    a, cid, chem, w_true = job
    model = _MODEL["m"]
    spec = model.chem_inputs
    x_true = np.array([math.log(w_true)] + chem_vector(spec, chem) + [a["d"], a["Z"]])
    t0 = time.time()
    clean = []
    for r in a["rates"]:
        i_sp = r * ONE_C * w_true / m_S_g(model.p)
        _, Q_end = fv_curve(model.p, spec, x_true, i_sp, [1.0], N=tuple(a["grid"]), ramp_s=model.ramp_s)
        if Q_end is None:
            return {"case": cid, "error": f"FV failed at {r} C"}
        q = np.linspace(max(15.0, i_sp * 100.0 / 3.6), a["frac"] * Q_end, a["n_points"])
        V, _ = fv_curve(model.p, spec, x_true, i_sp, q, N=tuple(a["grid"]), ramp_s=model.ramp_s)
        clean.append(Curve(i_sp, q, V, f"{r:g} C"))
    t_fv = time.time() - t0
    rows = []
    names = None
    for noise in a["noise_mV"]:
        for draw in range(a["draws"] if noise > 0 else 1):
            rng = np.random.default_rng([a["seed"], zlib.crc32(cid.encode()), draw, int(10 * noise)])
            data = [Curve(c.i_sp, c.q, c.V + (rng.normal(0, 1e-3 * noise, len(c.V)) if noise > 0 else 0.0), c.label)
                    for c in clean]
            t1 = time.time()
            fit = TwoCurveFit(model, data)
            names = fit.names
            rep = fit.report(fit.fit(a["starts"], a["seed"]))
            x = np.asarray(rep["x"])
            err = {n: scale(n) * (x[j] - x_true[j]) for j, n in enumerate(names)}
            z = {n: (err[n] / rep["sd"][n]) if rep["sd"].get(n) else None for n in names}
            rows.append({"noise_mV": noise, "draw": draw, "params": rep["params"], "err": err, "sd": rep["sd"],
                         "z": z, "rms_mV": rep["rms_mV"], "T_max": rep["T_max"], "model_rates": rep["model_rates"],
                         "flags": [f["flag"] + (f" {f['param']}" if "param" in f else "") for f in rep["flags"]
                                   if f["level"] != "info"],
                         "spread": rep["spread"], "fit_s": time.time() - t1})
    truth = {n: (math.exp(v) if scale(n) == 100.0 else v) for n, v in zip(names, x_true)}
    return {"case": cid, "truth": truth, "rows": rows, "fv_s": t_fv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default=str(NETWORK))
    ap.add_argument("--design", default=str(ROOT / "configs" / "lispan_chem_design_v2.json"))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--rates", nargs=2, type=float, default=[0.2, 0.6], help="model C-rates at the true w_S")
    ap.add_argument("--w", nargs="+", type=float, default=[0.8, 1.25], help="true w_S, cycled over the cases")
    ap.add_argument("--d", type=float, default=0.02)
    ap.add_argument("--Z", type=float, default=0.002)
    ap.add_argument("--noise-mV", nargs="+", type=float, default=[0.0, 5.0])
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--frac", type=float, default=0.90)
    ap.add_argument("--n-points", type=int, default=80)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--grid", nargs=2, type=int, default=[20, 10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    design = json.loads(Path(a.design).read_text())
    chems = {}
    for c in design["eval_cases"]:
        chems.setdefault(c["id"].split("_")[0], c["chem"])
    ids = a.cases or list(chems)
    cfg = {"rates": a.rates, "d": a.d, "Z": a.Z, "noise_mV": a.noise_mV, "draws": a.draws, "frac": a.frac,
           "n_points": a.n_points, "starts": a.starts, "grid": a.grid, "seed": a.seed}
    jobs = [(cfg, cid, chems[cid], a.w[k % len(a.w)]) for k, cid in enumerate(ids)]
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.network,)) as pool:
        res = []
        for r in pool.imap(run_case, jobs):
            res.append(r)
            if "error" in r:
                print(f"{r['case']}: {r['error']}", flush=True)
                continue
            for row in r["rows"]:
                print(f"{r['case']:10s} noise {row['noise_mV']:g}: rms {row['rms_mV']:.2f} mV | "
                      + ", ".join(f"{n} {row['err'][n]:+.2f}" + (f" ({row['z'][n]:+.1f} sd)" if row["noise_mV"] > 0 and row["z"][n] is not None else "")
                                  for n in row["err"])
                      + f" | T_max {max(row['T_max']):.3f} | flags {row['flags'] or '-'}", flush=True)
    ok = [r for r in res if "error" not in r]
    names = list(ok[0]["rows"][0]["err"])
    summary = {}
    print(f"\nsummary over {len(ok)} chemistries (units: w_S, b, k0 %; dU0, d mV; Z_CC Ohm cm2)")
    for noise in a.noise_mV:
        rows = [row for r in ok for row in r["rows"] if row["noise_mV"] == noise]
        s = {}
        for n in names:
            e = np.abs([row["err"][n] for row in rows])
            sd = [row["sd"][n] for row in rows if row["sd"].get(n)]
            zz = np.abs([row["z"][n] for row in rows if row["z"][n] is not None])
            s[n] = {"median_abs_err": float(np.median(e)), "max_abs_err": float(np.max(e)),
                    "median_sd": float(np.median(sd)) if sd else None,
                    "median_abs_z": float(np.median(zz)) if len(zz) else None,
                    "frac_abs_z_lt2": float(np.mean(zz < 2)) if len(zz) else None}
        summary[f"{noise:g}"] = s
        print(f"noise {noise:g} mV: " + "; ".join(
            f"{n} |err| med {v['median_abs_err']:.2f} max {v['max_abs_err']:.2f}"
            + (f", sd {v['median_sd']:.2f}, |z| med {v['median_abs_z']:.2f}, <2: {100 * v['frac_abs_z_lt2']:.0f} %"
               if noise > 0 and v["median_abs_z"] is not None else "") for n, v in s.items()))
    out = Path(a.json) if a.json else OUT / f"synthetic_{'+'.join(f'{r:g}' for r in a.rates)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "cases": res, "summary": summary, "wall_s": time.time() - t0}, indent=1,
                              default=float))
    print("written", out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
