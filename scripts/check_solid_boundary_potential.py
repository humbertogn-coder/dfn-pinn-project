"""Run only the fixed-current solid-potential boundary checks."""

from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    print("Checking joint solid-current boundaries, gauge, curvature and gradients.", flush=True)
    print("Analytic fixtures only; no training or full DFN validation.", flush=True)
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest",
        "tests/test_solid_boundary_potential.py", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).resolve().parents[1]))
