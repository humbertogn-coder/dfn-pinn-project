"""Convert one cell of the LG M50 synthetic degradation data set into the per-cycle npz layout of
dfn_pinn.v2.aging (t, I, V of the 1C discharge + ground-truth scalars when present).

    python scripts/v2_import_lgm50.py --cell-dir <folder of cell p1c1> --out results/aging_lgm50/p1c1 --every 50

The real file layout is not known yet (only LG_M50_cells_metadata.csv has been seen), so this reader
is deliberately generic: it accepts one CSV/parquet per cycle, or one long table with a cycle column,
and recognizes the usual column names (time/current/voltage/cycle, case-insensitive, with or without
units). Discharge current is made positive. Adjust COLUMN_ALIASES once the files are available; the
rest of the pipeline (scripts/v2_inverse_aging.py) then works unchanged.
"""

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

COLUMN_ALIASES = {
    "t": ["time [s]", "time_s", "time", "test_time", "t", "time (s)", "time[s]"],
    "I": ["current [a]", "current_a", "current", "i", "current (a)", "current[a]"],
    "V": ["voltage [v]", "voltage_v", "voltage", "v", "terminal voltage [v]", "voltage (v)", "voltage[v]"],
    "cycle": ["cycle", "cycle_number", "cycle number", "cycle_index", "cyc"],
    "step": ["step", "step_number", "step type", "step_type", "mode"],
}
TRUTH_ALIASES = {   # optional per-cycle ground truth (degradation pathways); extend when the files are known
    "LLI_pct": ["loss of lithium inventory [%]", "lli [%]", "lli"],
    "LAM_ne_pct": ["loss of active material in negative electrode [%]", "lam_ne [%]", "lam_ne"],
    "LAM_pe_pct": ["loss of active material in positive electrode [%]", "lam_pe [%]", "lam_pe"],
    "SEI_thickness_m": ["x-averaged negative total sei thickness [m]", "sei thickness [m]", "sei_thickness"],
    "Capacity_Ah": ["capacity [a.h]", "discharge capacity [a.h]", "capacity_ah", "capacity"],
}


def find_col(df, aliases):
    cols = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a in cols:
            return cols[a]
    for c in df.columns:   # fuzzy: alias contained in the column name
        lc = c.lower()
        if any(re.sub(r"[^a-z]", "", a) in re.sub(r"[^a-z]", "", lc) for a in aliases):
            return c
    return None


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() in (".h5", ".hdf5", ".hdf"):
        return pd.read_hdf(path)
    return pd.read_csv(path)


def discharge_segment(df, tcol, icol, vcol):
    """Rows of the main discharge: the contiguous block of |I| > 0.5 A with the largest voltage DROP
    (sign conventions differ between cyclers and PyBaMM, so the voltage decides). Current is returned
    positive (discharge)."""
    t = df[tcol].to_numpy(float); I = df[icol].to_numpy(float); V = df[vcol].to_numpy(float)
    on = np.abs(I) > 0.5
    blocks, cur = [], None
    for k, flag in enumerate(on):
        if flag and cur is None:
            cur = k
        if cur is not None and (not flag or k == len(on) - 1):
            end = k if not flag else k + 1
            blocks.append((cur, end)); cur = None
    if not blocks:
        raise ValueError("no block with |I| > 0.5 A")
    a, b = max(blocks, key=lambda ab: V[ab[0]] - V[ab[1] - 1])   # largest voltage drop = discharge
    return t[a:b] - t[a], np.abs(I[a:b]), V[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=50, help="keep every k-th cycle (and the first and last)")
    ap.add_argument("--pattern", default="*", help="glob for the per-cycle files inside --cell-dir")
    args = ap.parse_args()
    cell_dir, out = Path(args.cell_dir), Path(args.out)
    (out / "cycles").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in cell_dir.glob(args.pattern) if p.suffix.lower() in (".csv", ".parquet", ".pq", ".h5", ".hdf5"))
    if not files:
        sys.exit(f"no data files in {cell_dir}")
    rows = []
    if len(files) == 1:   # one long table with a cycle column
        df = read_table(files[0])
        cyc_col = find_col(df, COLUMN_ALIASES["cycle"])
        groups = df.groupby(cyc_col) if cyc_col else [(1, df)]
        items = [(int(k), g) for k, g in groups]
    else:                 # one file per cycle: cycle number from the file name
        items = []
        for p in files:
            m = re.findall(r"\d+", p.stem)
            items.append((int(m[-1]) if m else len(items) + 1, read_table(p)))
    items.sort(key=lambda kv: kv[0])
    last = items[-1][0]
    for k, df in items:
        if not (k == items[0][0] or k == last or k % args.every == 0):
            continue
        tcol, icol, vcol = (find_col(df, COLUMN_ALIASES[x]) for x in ("t", "I", "V"))
        if None in (tcol, icol, vcol):
            sys.exit(f"cycle {k}: could not identify time/current/voltage columns in {list(df.columns)[:12]}")
        t, I, V = discharge_segment(df, tcol, icol, vcol)
        truth = {}
        for key, aliases in TRUTH_ALIASES.items():
            c = find_col(df, aliases)
            if c is not None:
                truth[key] = float(np.nanmean(df[c].to_numpy(float)))
        np.savez(out / "cycles" / f"cycle_{k:04d}.npz", t=t, I=I, V=V, cycle=k, **truth)
        rows.append({"cycle": k, "t_cc_s": float(t[-1]), "Q_Ah": float(np.trapezoid(I, t) / 3600), **truth})
        print(f"cycle {k:5d}: discharge {t[-1]:.0f} s, {rows[-1]['Q_Ah']:.3f} Ah, {len(t)} samples")
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print("written", out)


if __name__ == "__main__":
    main()
