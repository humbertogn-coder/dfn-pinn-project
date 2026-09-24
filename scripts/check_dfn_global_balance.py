"""Audit three-region current and salt integral identities without training."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Auditing three-region balances: physical dx, charge signs, interfaces and collector gauge.", flush=True)
    print("Includes deliberate source/flux errors; global salt cancellation is not a local PDE solution.", flush=True)
    path = Path(__file__).resolve().parents[1]/"tests/test_dfn_global_balance.py"
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
