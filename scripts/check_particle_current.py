"""Audit shared-current particle constraints without training."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Auditing shared particle current: inventory, surface flux and gradients.", flush=True)
    print("Both particle scales and signs; manufactured fields only, no training.", flush=True)
    path = Path(__file__).resolve().parents[1]/"tests/test_particle_current.py"
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
