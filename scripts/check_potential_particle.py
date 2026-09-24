"""Check potential-driven PINN setup; no optimization or reference solves."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Checking potential PINN: initial state, fixed protocol, raw kinetics and gradients.", flush=True)
    print("No training, reference simulation or full test suite is run.", flush=True)
    path = Path(__file__).resolve().parents[1]/"tests/test_potential_particle.py"
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
