"""Audit the region-local electrolyte mass operator without training."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Auditing electrolyte mass: nonlinear diffusive flux, SI scales, reaction signs and gradients.", flush=True)
    print("Region-local analytic checks only; no training or full DFN validation.", flush=True)
    path = Path(__file__).resolve().parents[1]/"tests/test_electrolyte_mass.py"
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
