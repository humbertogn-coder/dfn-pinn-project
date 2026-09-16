"""Audit normalized spherical diffusion in PyTorch; no training or DFN solve."""

from pathlib import Path
import sys

import pytest


if __name__ == "__main__":
    print("Auditing spherical residual: chain rule, center, surface flux and lithium balance.", flush=True)
    print("Analytic polynomial and no-flux eigenmode; both Chen2020 particle scales.", flush=True)
    tests = Path(__file__).resolve().parents[1] / "tests" / "test_spherical_residual.py"
    sys.exit(pytest.main([str(tests), "-q", "-p", "no:cacheprovider"]))
