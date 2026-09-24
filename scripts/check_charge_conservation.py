"""Audit local DFN charge operators without training."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Auditing charge: solid/electrolyte currents, source signs, scales, gauge and gradients.", flush=True)
    print("Region-local analytic checks only; no interfaces, training or full DFN validation.", flush=True)
    path = Path(__file__).resolve().parents[1]/"tests/test_charge_conservation.py"
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
