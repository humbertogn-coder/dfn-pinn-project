"""Audit shared-current Butler-Volmer coupling without training."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Auditing particle kinetics: shared current, SI fields, signs and gradients.", flush=True)
    print("Manufactured states for both electrodes and explicit fit modes; no training.", flush=True)
    path = Path(__file__).resolve().parents[1]/"tests/test_particle_kinetics.py"
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
