"""Generate the PyBaMM reference used to evaluate the v2 PINN.

Example (Anaconda Prompt, from the repository root):
    python scripts/v2_make_reference.py --current 5 --t-end 3000 --ramp 30

Writes results/v2_reference/<name>.npz (ignored by Git; regenerate as needed).
"""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.v2.params import CellParams, Protocol  # noqa: E402
from dfn_pinn.v2 import reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", type=float, default=5.0, help="plateau current [A], >0 discharge")
    ap.add_argument("--t-end", type=float, default=3000.0)
    ap.add_argument("--ramp", type=float, default=30.0)
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--nr", type=int, default=60)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cell, protocol = CellParams(), Protocol(args.current, args.ramp, args.t_end)
    mesh = {"x_n": args.nx, "x_s": max(args.nx // 2, 10), "x_p": args.nx, "r_n": args.nr, "r_p": args.nr}
    start = time.perf_counter()
    _, solution, mesh = reference.solve(cell, protocol, mesh=mesh)
    name = args.out or f"ref_I{args.current:g}A_t{args.t_end:g}s_ramp{args.ramp:g}s_x{args.nx}_r{args.nr}"
    path = ROOT / "results" / "v2_reference" / f"{name}.npz"
    data = reference.export(solution, path, cell, protocol, mesh)
    print(f"Solved in {time.perf_counter() - start:.1f} s; t_final = {data['t'][-1]:.1f} s; "
          f"V: {data['V'][0]:.4f} -> {data['V'][-1]:.4f} V")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
