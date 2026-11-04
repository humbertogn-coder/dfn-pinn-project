"""How much does the network's fixed cell geometry bias the two-curve parameters? (geometry step 0)

The chemistry network (and so the two-curve tool) is trained for the Simanjuntak test cell: 100 um cathode with a low
SPAN fraction (porosity 0.87, 0.77 mg S/cm2) and a 618 um glass-fibre separator.  Here FV discharges are generated for
other cell geometries with a KNOWN chemistry, converted to specific capacity / specific current like literature data,
and fitted with the network exactly as SPAN-DB cells are.  The fitted parameters minus the truth are the bias that the
geometry mismatch alone causes; the misfit and the blind prediction of a middle current show whether it is visible.

Geometry transform (geometry(), all per-volume SPAN quantities scale together so that the chemistry per chain is
unchanged):
  - cathode: thickness L_cat; SPAN volume fraction eps_SPAN = f x 0.0945 with the carbon fraction, the chain
    concentrations (c_init, c_ref) and the specific surface a_SPAN0 scaled by the same f;
  - separator: thickness L_sep, porosity, Bruggeman exponent beta_sep (effective transport = porosity^beta).
Same specific current = same C-rate in every cell (1 C = ONE_C / m_S of the reference cell, per g S).

    python scripts/lispan_geometry_bias.py --workers 2
    python scripts/lispan_geometry_bias.py --geoms ref celgard --cases p0 --workers 1

Truth: w_S = 1 (data per g S of the model's sulfur), chemistry of the design's evaluation cases, d = 0, Z_CC = 20 Ohm cm2.
Output: results/lispan/geometry/bias_<rates>.json; errors in the tool's sd units (w_S, b, k0 in %, dU0 and d in mV,
Z_CC in Ohm cm2).
"""

import argparse
from dataclasses import replace
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dfn_pinn.lispan.chem_pinn import chem_vector, load_chem_model  # noqa: E402
from dfn_pinn.lispan.model import solve  # noqa: E402
from dfn_pinn.lispan.params import LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.rate_pinn import ONE_C  # noqa: E402
from dfn_pinn.lispan.two_curve import Curve, TwoCurveFit, fv_params, m_S_g, network_sd_for  # noqa: E402

NETWORK = ROOT / "results" / "lispan_chem_runs" / "chem_v4_warm_salt_s0" / "final.pt"
OUT = ROOT / "results" / "lispan" / "geometry"

# name: (L_cat [um], f = eps_SPAN / 0.0945, L_sep [um], separator porosity, beta_sep)
# Celgard-type separator: 25 um, porosity 0.41, MacMullin number ~8 -> porosity^beta = 1/8 -> beta = 2.33
GEOMS = {
    "ref": (100.0, 1.0, 618.0, 0.89, 1.0),
    "celgard": (100.0, 1.0, 25.0, 0.41, 2.33),
    "thick2x": (200.0, 1.0, 618.0, 0.89, 1.0),
    "dense3": (67.0, 3.0, 618.0, 0.89, 1.0),
    "dense4": (75.0, 4.0, 618.0, 0.89, 1.0),
    "coin_mid": (67.0, 3.0, 25.0, 0.41, 2.33),
    "coin_high": (75.0, 4.0, 25.0, 0.41, 2.33),
}
_W = {}


def geometry(p, L_cat_um, f, L_sep_um, sep_porosity, beta_sep):
    return replace(p, L_cat=L_cat_um * 1e-6, eps_SPAN=f * p.eps_SPAN, eps_CB=f * p.eps_CB,
                   c_init=(f * p.c_init[0],) + tuple(p.c_init[1:]), c_ref=tuple(f * c for c in p.c_ref),
                   a_SPAN0=f * p.a_SPAN0, L_sep=L_sep_um * 1e-6, eps_sep=1.0 - sep_porosity, beta_sep=beta_sep)


def _init(network):
    torch.set_num_threads(1)
    _W["m"] = load_chem_model(network)[0]
    _W["sd"] = network_sd_for(network)


def fv_data(p_geo, i_sp, N=(20, 10), ramp_s=30.0):
    """FV discharge of the geometry cell at specific current i_sp [A/g S]: (Q [mAh/g S], V) or None."""
    prot = LiSPANProtocol(current=i_sp * m_S_g(p_geo), ramp_s=ramp_s)
    for tol in (1e-6, 1e-5):
        try:
            _, res = solve(p_geo, prot, N_c=N[0], N_s=N[1], n_out=1200, rtol=tol)
            return np.asarray(res["Q_mAh_gS"]), np.asarray(res["V"])
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    return None


def scale(name):
    if name == "d" or name.startswith("dU0_"):
        return 1e3
    if name == "Z_CC":
        return 1e4
    return 100.0


def run(job):
    a, gname, cid, chem = job
    model = _W["m"]
    spec = model.chem_inputs
    x_true = np.array([0.0] + chem_vector(spec, chem) + [0.0, a["Z"]])
    p_geo = geometry(fv_params(model.p, spec, x_true), *GEOMS[gname])
    ref_mS = m_S_g(model.p)
    out = {"geometry": gname, "case": cid, "loading_mg_cm2": m_S_g(p_geo) / 10.0,
           "porosity_cat": p_geo.eps_e0, "curves": {}}
    t0 = time.time()
    data = {}
    for r in sorted(set(a["rates"]) | set(a["predict"])):
        i_sp = r * ONE_C / ref_mS
        res = fv_data(p_geo, i_sp)
        if res is None:
            out["curves"][f"{r:g}"] = {"failed": True}
            continue
        Q, V = res
        q = np.linspace(max(15.0, i_sp * 100.0 / 3.6), a["frac"] * Q[-1], a["n_points"])
        data[r] = Curve(i_sp, q, np.interp(q, Q, V), f"{r:g} C")
        out["curves"][f"{r:g}"] = {"Q_end": float(Q[-1]), "V_mid": float(np.interp(0.5 * Q[-1], Q, V))}
    out["fv_s"] = time.time() - t0
    if any(r not in data for r in a["rates"]):
        out["error"] = "FV failed at a calibration rate"
        return out
    fit = TwoCurveFit(model, [data[r] for r in a["rates"]], network_sd=_W["sd"])
    rep = fit.report(fit.fit(a["starts"], 0))
    x = np.asarray(rep["x"])
    out.update(params=rep["params"], sd_total=rep["sd_total"], rms_mV=rep["rms_mV"], T_max=rep["T_max"],
               flags=[f["flag"] + (f" {f['param']}" if "param" in f else "") for f in rep["flags"] if f["level"] != "info"],
               err={n: scale(n) * (x[j] - x_true[j]) for j, n in enumerate(fit.names)})
    preds = {}
    for r in a["predict"]:
        if r not in data:
            continue
        c = data[r]
        V, T_max = fit.predict(x, c.i_sp, c.q)
        ok = np.isfinite(V)
        if ok.sum() >= 5:
            preds[f"{r:g}"] = {"rms_mV": 1e3 * float(np.sqrt(np.mean((V[ok] - c.V[ok]) ** 2))), "T_max": float(T_max)}
    out["predictions"] = preds
    out["wall_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default=str(NETWORK))
    ap.add_argument("--design", default=str(ROOT / "configs" / "lispan_chem_design_v3.json"))
    ap.add_argument("--geoms", nargs="*", default=list(GEOMS))
    ap.add_argument("--cases", nargs="*", default=["p0", "litHao", "litChen", "litSpan300"])
    ap.add_argument("--rates", nargs=2, type=float, default=[0.2, 2.0])
    ap.add_argument("--predict", nargs="*", type=float, default=[0.5, 1.0])
    ap.add_argument("--Z", type=float, default=0.002)
    ap.add_argument("--frac", type=float, default=0.90)
    ap.add_argument("--n-points", type=int, default=80)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    design = json.loads(Path(a.design).read_text())
    chems = {}
    for c in design["eval_cases"]:
        chems.setdefault(c["id"].split("_")[0], c["chem"])
    cfg = {"rates": a.rates, "predict": a.predict, "Z": a.Z, "frac": a.frac, "n_points": a.n_points, "starts": a.starts}
    jobs = [(cfg, g, cid, chems[cid]) for g in a.geoms for cid in a.cases]
    t0 = time.time()
    res = []
    with Pool(a.workers, initializer=_init, initargs=(a.network,)) as pool:
        for r in pool.imap(run, jobs):
            res.append(r)
            if "error" in r:
                print(f"{r['geometry']:10s} {r['case']:10s} {r['error']}", flush=True)
                continue
            e = r["err"]
            print(f"{r['geometry']:10s} {r['case']:10s} load {r['loading_mg_cm2']:.2f} rms {r['rms_mV']:5.1f} | w_S {e['w_S']:+6.1f}% "
                  f"dU0_2 {e['dU0_2']:+6.1f} dU0_3rel2 {e['dU0_3rel2']:+6.1f} b_12 {e['b_12']:+6.1f}% b_3 {e['b_3']:+6.1f}% "
                  f"k0_1 {e['k0_1']:+6.0f}% d {e['d']:+6.1f} Z {e['Z_CC']:+6.1f} | pred "
                  + ", ".join(f"{k} C {v['rms_mV']:.0f}" for k, v in r["predictions"].items())
                  + f" | {','.join(r['flags']) or '-'} ({r['wall_s']:.0f} s)", flush=True)
    out = Path(a.json) if a.json else OUT / f"bias_{'+'.join(f'{r:g}' for r in a.rates)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "geometries": {g: GEOMS[g] for g in a.geoms}, "results": res,
                               "wall_s": time.time() - t0}, indent=1, default=float))
    print("written", out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
