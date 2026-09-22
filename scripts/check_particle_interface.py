"""Audit assembled surface feedback on manufactured particle states."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Auditing particle interface: surface feedback, flux, kinetics and inventory.", flush=True)
    print("Manufactured diffusion states and perturbed fields; no training or full DFN solve.", flush=True)
    path = Path(__file__).resolve().parents[1]/"tests/test_particle_interface.py"
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
