"""Which combinations of the cell geometry does the discharge voltage depend on? (geometry v5, step 1)

FV discharges of one chemistry in groups of geometries that share a candidate reduced variable:
  - "loading": same sulfur loading m_S = f x L_cat (f = SPAN scale factor, see lispan_geometry_bias.geometry) with
    different splits between thickness and density;
  - "thickness": same f, different thickness (loading changes with it);
  - "separator": same effective separator length L_sep / porosity^beta with different thickness / porosity / beta;
  - "sep_length": same separator type, different effective length.
Inside a group every member is compared with the first one at equal specific capacity (mAh/g S) and specific current:
  rms_mV      plain rms voltage difference over the common capacity range (both calibration rates);
  rms_dZ_mV   rms left after the best common offset d and series resistance Z (V + d - Z i_cell fitted over the rates
              together) - the part the two-curve tool cannot absorb without biasing the chemistry parameters;
  dZ_ohm_cm2  that series resistance; dQ_pct  end-capacity difference at each rate.

    python scripts/lispan_geometry_scaling.py --workers 2
Output: results/lispan/geometry/scaling_<case>.json
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from dfn_pinn.lispan.chem_pinn import CHEM_SPECS, chem_vector  # noqa: E402
from dfn_pinn.lispan.params import LiSPANParams  # noqa: E402
from dfn_pinn.lispan.rate_pinn import ONE_C  # noqa: E402
from dfn_pinn.lispan.two_curve import fv_params, m_S_g  # noqa: E402
from lispan_geometry_bias import fv_data, geometry  # noqa: E402

CEL = (25.0, 0.41, 2.33)        # Celgard-type: 25 um, porosity 0.41, MacMullin ~8
# member: (L_cat um, f, L_sep um, sep porosity, beta)
GROUPS = {
    "loading_x2": [(200, 1, *CEL), (100, 2, *CEL), (67, 3, *CEL), (50, 4, *CEL), (40, 5, *CEL)],
    "loading_x4": [(200, 2, *CEL), (133, 3, *CEL), (100, 4, *CEL), (80, 5, *CEL)],
    "thickness_f3": [(33, 3, *CEL), (67, 3, *CEL), (100, 3, *CEL), (133, 3, *CEL)],
    "separator_200": [(67, 3, 25, 0.41, 2.33), (67, 3, 180, 0.90, 1.0), (67, 3, 100, 0.50, 1.0), (67, 3, 50, 0.25, 1.0),
                      (67, 3, 20, 0.32, 2.0)],
    "sep_length": [(67, 3, 25, 0.41, 1.0), (67, 3, 25, 0.41, 2.33), (67, 3, 25, 0.41, 3.0), (67, 3, 260, 0.90, 1.0),
                   (67, 3, 618, 0.89, 1.0)],
}


def eff_len(m):
    return m[2] / m[3] ** m[4]


def _job(job):
    case, x, gname, k, member, rates = job
    p = geometry(fv_params(LiSPANParams(), CHEM_SPECS["v2"], np.asarray(x)), *member)
    out = {"group": gname, "k": k, "member": member, "loading": m_S_g(p) / 10.0, "porosity_cat": p.eps_e0,
           "L_sep_eff_um": eff_len(member), "curves": {}}
    t0 = time.time()
    for r in rates:
        i_sp = r * ONE_C / m_S_g(LiSPANParams())          # same C-rate per g S as the reference cell
        res = fv_data(p, i_sp)
        out["curves"][f"{r:g}"] = None if res is None else {"Q": res[0].tolist(), "V": res[1].tolist(),
                                                           "i_cell_A_m2": i_sp * m_S_g(p)}
    out["wall_s"] = time.time() - t0
    return out


def compare(a, b, rates, frac=0.9):
    """b against a: rms, rms after a common d + Z i_cell, end-capacity differences."""
    rows, cols, dq = [], [], {}
    for r in rates:
        ca, cb = a["curves"].get(f"{r:g}"), b["curves"].get(f"{r:g}")
        if not ca or not cb:
            return None
        qa, va, qb, vb = map(np.asarray, (ca["Q"], ca["V"], cb["Q"], cb["V"]))
        q = np.linspace(30.0, frac * min(qa[-1], qb[-1]), 120)
        dv = np.interp(q, qb, vb) - np.interp(q, qa, va)
        rows.append(dv)
        # b ~ a + d - dZ i_cell,b (a common offset and an extra series resistance carried by b's own current)
        cols.append(np.column_stack([np.ones_like(q), -np.full_like(q, cb["i_cell_A_m2"])]))
        dq[f"{r:g}"] = 100.0 * (qb[-1] / qa[-1] - 1.0)
    dv, A = np.concatenate(rows), np.vstack(cols)
    coef, *_ = np.linalg.lstsq(A, dv, rcond=None)
    res = dv - A @ coef
    return {"rms_mV": 1e3 * float(np.sqrt(np.mean(dv ** 2))), "rms_dZ_mV": 1e3 * float(np.sqrt(np.mean(res ** 2))),
            "d_mV": 1e3 * float(coef[0]), "dZ_ohm_cm2": 1e4 * float(coef[1]),
            "dQ_pct": dq, "max_mV": 1e3 * float(np.max(np.abs(dv)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default=str(ROOT / "configs" / "lispan_chem_design_v3.json"))
    ap.add_argument("--case", default="p0")
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--rates", nargs="*", type=float, default=[0.2, 1.0, 2.0])
    ap.add_argument("--Z", type=float, default=0.002)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    design = json.loads(Path(a.design).read_text())
    chem = next(c["chem"] for c in design["eval_cases"] if c["id"].split("_")[0] == a.case)
    x = [0.0] + chem_vector(CHEM_SPECS["v2"], chem) + [0.0, a.Z]
    jobs = [(a.case, x, g, k, m, a.rates) for g in a.groups for k, m in enumerate(GROUPS[g])]
    t0 = time.time()
    with Pool(a.workers) as pool:
        sims = list(pool.imap(_job, jobs))
    out = {"case": a.case, "rates": a.rates, "Z_ohm_cm2": 1e4 * a.Z, "groups": {}, "sims": sims}
    for g in a.groups:
        mem = [s for s in sims if s["group"] == g]
        rows = []
        for s in mem:
            c = compare(mem[0], s, a.rates)
            rows.append({"member": s["member"], "loading": s["loading"], "porosity_cat": s["porosity_cat"],
                         "L_sep_eff_um": s["L_sep_eff_um"], "wall_s": s["wall_s"],
                         "Q_end": {r: (s["curves"][r]["Q"][-1] if s["curves"][r] else None) for r in s["curves"]},
                         "vs_first": c})
            cs = "FV failed" if c is None else (f"rms {c['rms_mV']:6.1f}  after d+Z {c['rms_dZ_mV']:5.1f} mV  "
                                                f"dZ {c['dZ_ohm_cm2']:+6.1f}  d {c['d_mV']:+6.1f}  dQ "
                                                + " ".join(f"{v:+5.1f}" for v in c["dQ_pct"].values()) + " %")
            print(f"{g:14s} L {s['member'][0]:4.0f} f {s['member'][1]:.0f} sep {s['member'][2]:4.0f}/{s['member'][3]:.2f}/"
                  f"{s['member'][4]:.2f} (eff {s['L_sep_eff_um']:5.0f})  load {s['loading']:.2f} por {s['porosity_cat']:.2f} | {cs}",
                  flush=True)
        out["groups"][g] = rows
    path = ROOT / "results" / "lispan" / "geometry" / f"scaling_{a.case}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print("written", path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
