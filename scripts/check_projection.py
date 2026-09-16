"""Run integral-current projection tests without simulation or training."""

from pathlib import Path
import sys

import pytest


if __name__ == "__main__":
    print("Auditing current projection: physical weights, signs, balances and gradients.", flush=True)
    print("Independent quadrature checks include a deliberate underresolution case.", flush=True)
    tests = Path(__file__).resolve().parents[1] / "tests" / "test_projection.py"
    sys.exit(pytest.main([str(tests), "-q", "-p", "no:cacheprovider"]))
