"""Run symmetric Butler-Volmer unit tests without simulation or training."""

from pathlib import Path
import sys

import pytest


if __name__ == "__main__":
    print("Auditing symmetric Butler-Volmer: signs, round trips, derivatives and overflow.", flush=True)
    print("Scope: one electron, unit utilization, positive j0; no film resistance.", flush=True)
    tests = Path(__file__).resolve().parents[1] / "tests" / "test_kinetics.py"
    sys.exit(pytest.main([str(tests), "-q", "-p", "no:cacheprovider"]))
