"""Run the constitutive value/derivative audit without a DFN simulation."""

from pathlib import Path
import sys

import pytest


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "tests" / "test_constitutive.py"
    print("Auditing Chen2020 values and first/second derivatives in float64.", flush=True)
    print("Scope: raw fits and processed PyBaMM expressions on interior samples.", flush=True)
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
