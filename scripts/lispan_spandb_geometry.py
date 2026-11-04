"""Cell geometry in SPAN-DB: what the database reports for the fitted cells, mapped to the model's geometry (geometry v5).

For every cell in a SPAN-DB fit run (summary.csv of scripts/lispan_spandb_fit.py) this reads the electrode and cell
rows and derives:
  - sulfur loading m_S [mg S/cm2]: loading_s_mg_cm2, else loading_span_mg_cm2 x s_content (first number of a range);
  - cathode thickness [um] and porosity when reported;
  - the model's SPAN scale factor f (eps_SPAN = f x 0.0945, all SPAN quantities per volume scaled by f) from loading and
    thickness: m_S = m_S,ref x f x L_cat / L_ref (m_S,ref = 0.767 mg S/cm2 at L_ref = 100 um);
  - separator class with TYPICAL catalogue thickness and porosity (to be checked against data sheets; values below are
    nominal numbers, not measurements) and an assumed MacMullin number -> effective length L_sep x N_M.
Output: results/lispan/geometry/spandb_geometry_<tag>.json (per cell + coverage tables) and a printed summary.

    python scripts/lispan_spandb_geometry.py --spandb <span_db folder> --results results/lispan/spandb_v4ws_120 --tag v4_120
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
M_S_REF = 0.767      # mg S/cm2 of the model cell (params: L_cat 100 um, eps_SPAN 0.0945, c_init 598)
L_REF = 100.0        # um
SOLID_REF = 0.13     # solid volume fraction of the model cathode (porosity 0.87) at f = 1

# class: (thickness um, porosity, MacMullin number) - nominal catalogue values, assumed MacMullin numbers
SEPARATORS = {
    "celgard_2400": (25.0, 0.41, 8.0),
    "celgard_2500": (25.0, 0.55, 5.0),
    "celgard_2325": (25.0, 0.39, 9.0),
    "celgard_2320": (20.0, 0.39, 9.0),
    "celgard_2300": (25.0, 0.40, 9.0),
    "celgard_other": (25.0, 0.40, 8.0),
    "pe_20um": (20.0, 0.40, 8.0),
    "pp_25um": (25.0, 0.40, 8.0),
    "glass_fibre": (260.0, 0.90, 1.4),
    "gel_or_membrane": (None, None, None),
    "solid": (None, None, None),
    "NR": (None, None, None),
}


def num(v):
    """First number in a free-text field (ranges -> lower end); None when not reported."""
    s = str(v or "").strip()
    if not s or s.upper().startswith(("NR", "NA", "N/A")):
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s.replace(",", "."))
    if not m:
        return None
    x = float(m.group(0))
    return x if math.isfinite(x) else None


def separator_class(text):
    t = (text or "").lower()
    if not t.strip() or t.strip().startswith("nr"):
        return "NR"
    if t.startswith("na") or "solid" in t or "pellet" in t or "llzo" in t:
        return "solid"
    if "glass" in t or "whatman" in t or "gf/" in t:
        return "glass_fibre"
    if "gpe" in t or "gel" in t or "mof" in t or "cellulose" in t or "membrane (mof" in t:
        return "gel_or_membrane"
    for k in ("2400", "2500", "2325", "2320", "2300"):
        if k in t:
            return "celgard_" + k
    if "celgard" in t or "clegard" in t or "cellgard" in t:
        return "celgard_other"
    if "entek" in t or re.search(r"\bpe\b", t) or "polyethylene" in t:
        return "pe_20um"
    if re.search(r"\bpp\b", t) or "polypropylene" in t:
        return "pp_25um"
    return "gel_or_membrane"


def quantiles(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]        # noqa: E731
    return {"n": len(v), "min": v[0], "p10": q(0.1), "median": st.median(v), "p90": q(0.9), "max": v[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spandb", required=True)
    ap.add_argument("--results", default=str(ROOT / "results" / "lispan" / "spandb_v4ws_120"))
    ap.add_argument("--tag", default="v4_120")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    db = Path(a.spandb) / "db"
    rd = lambda f: {r[next(iter(r))]: r for r in csv.DictReader(open(db / f, encoding="utf-8"))}      # noqa: E731
    cells, elec, mats = rd("cells.csv"), rd("electrodes.csv"), rd("materials.csv")
    summ = list(csv.DictReader(open(Path(a.results) / "summary.csv", encoding="utf-8")))
    rows = []
    for s in summ:
        if s["status"] != "fitted":
            continue
        c = cells[s["cell"]]
        e = elec.get(c["electrode_id"], {})
        m = mats.get(e.get("material_id", ""), {})
        s_frac = num(m.get("s_content_wt"))
        mS = num(e.get("loading_s_mg_cm2"))
        mS_src = "reported"
        if mS is None and num(e.get("loading_span_mg_cm2")) is not None and s_frac is not None:
            mS, mS_src = num(e["loading_span_mg_cm2"]) * s_frac / 100.0, "span x s_content"
        if mS is None:
            mS_src = None
        L = num(e.get("thickness_um"))
        por = num(e.get("porosity"))
        if por is not None and por > 1.0:
            por /= 100.0
        f = mS / (M_S_REF * L / L_REF) if (mS is not None and L) else None
        sep = separator_class(c.get("separator"))
        L_sep, por_sep, N_M = SEPARATORS[sep]
        rms = num(s.get("rms_mV"))
        rows.append({
            "cell": s["cell"], "family": s["electrolyte_family"], "rms_mV": rms, "good": rms is not None and rms <= 25.0,
            "loading_S_mg_cm2": mS, "loading_source": mS_src, "thickness_um": L, "porosity": por, "f_model": f,
            "porosity_model_from_f": (1.0 - SOLID_REF * f) if f is not None else None,
            "active_wt": num(e.get("active_wt")), "separator_text": c.get("separator"), "separator_class": sep,
            "L_sep_um": L_sep, "porosity_sep": por_sep, "MacMullin": N_M,
            "L_sep_eff_um": (L_sep * N_M) if L_sep else None, "cell_format": c.get("cell_format"),
            "es_ratio_uL_mg": num(c.get("es_ratio_uL_mg")),
        })
    n = len(rows)
    cover = lambda k: sum(r[k] is not None for r in rows)       # noqa: E731
    sep_counts = {}
    for r in rows:
        sep_counts[r["separator_class"]] = sep_counts.get(r["separator_class"], 0) + 1
    liquid = [r for r in rows if r["separator_class"] != "solid"]
    out = {
        "results": a.results, "n_fitted": n,
        "coverage": {"loading_S": cover("loading_S_mg_cm2"), "thickness": cover("thickness_um"),
                     "porosity": cover("porosity"), "f_model": cover("f_model"), "es_ratio": cover("es_ratio_uL_mg"),
                     "separator_known": sum(r["separator_class"] not in ("NR",) for r in rows),
                     "separator_catalogue": sum(r["L_sep_um"] is not None for r in rows)},
        "separator_classes": dict(sorted(sep_counts.items(), key=lambda kv: -kv[1])),
        "loading_S": quantiles([r["loading_S_mg_cm2"] for r in liquid]),
        "thickness_um": quantiles([r["thickness_um"] for r in liquid]),
        "porosity": quantiles([r["porosity"] for r in liquid]),
        "f_model": quantiles([r["f_model"] for r in liquid]),
        "L_sep_eff_um": quantiles([r["L_sep_eff_um"] for r in liquid]),
        "loading_S_good": quantiles([r["loading_S_mg_cm2"] for r in liquid if r["good"]]),
        "loading_S_bad": quantiles([r["loading_S_mg_cm2"] for r in liquid if not r["good"]]),
        "model_reference": {"loading_S": M_S_REF, "thickness_um": L_REF, "porosity": 1 - SOLID_REF, "f": 1.0,
                            "L_sep_eff_um": 618.0 / 0.89},
        "separator_table": SEPARATORS,
        "cells": rows,
    }
    path = Path(a.out) if a.out else ROOT / "results" / "lispan" / "geometry" / f"spandb_geometry_{a.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"fitted cells {n}; coverage", out["coverage"])
    print("separators", out["separator_classes"])
    for k in ("loading_S", "loading_S_good", "loading_S_bad", "thickness_um", "porosity", "f_model", "L_sep_eff_um"):
        print(f"{k:16s}", out[k])
    print("written", path)


if __name__ == "__main__":
    main()
