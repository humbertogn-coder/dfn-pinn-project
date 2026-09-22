"""Audit time integration of interfacial current without training."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Auditing current integral: SI charge, moving time limit, signs and gradients.", flush=True)
    print("Constant/variable currents, inventory coupling and deliberate underresolution; no training.", flush=True)
    path = Path(__file__).resolve().parents[1]/"tests/test_current_integral.py"
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
