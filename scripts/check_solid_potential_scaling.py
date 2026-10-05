"""Run the bounded solid-potential output-scale checks."""

from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print("Auditing solid potential amplitude: analytic currents, charge signs and paired gradients.", flush=True)
    print("No training, checkpoint modification or full DFN validation.", flush=True)
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest",
                                     "tests/test_solid_potential_scaling.py", "-q",
                                     "-p", "no:cacheprovider"], cwd=root))
