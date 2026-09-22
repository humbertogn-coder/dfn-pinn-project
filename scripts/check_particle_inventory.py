"""Audit synthetic hard inventory correction before using it in training."""

from pathlib import Path
import sys

import pytest


if __name__ == "__main__":
    print("Auditing inventory projection: mean, radial flux, time derivatives and parameter gradients.", flush=True)
    print("Synthetic unit outward flux only. Includes deliberate quadrature underresolution; no training.", flush=True)
    tests = Path(__file__).resolve().parents[1]/"tests/test_particle_inventory.py"
    sys.exit(pytest.main([str(tests), "-q", "-p", "no:cacheprovider"]))
