"""Audit the physical local-inventory contract before electrochemical coupling."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Auditing physical particle inventory: current signs, charge units and autograd.", flush=True)
    print("Both Chen2020 particle scales; no training or coupled DFN solve.", flush=True)
    tests = Path(__file__).resolve().parents[1]/"tests/test_particle_balance.py"
    sys.exit(pytest.main([str(tests), "-q", "-p", "no:cacheprovider"]))
