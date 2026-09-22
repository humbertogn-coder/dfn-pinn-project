"""Check the first coupled experiment specification without training."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Checking coupled particle specification and compatible analytic reference.", flush=True)
    print("Specification check only; no training or DFN solve is run by this command.", flush=True)
    path = Path(__file__).resolve().parents[1]/"tests/test_coupled_particle_spec.py"
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
