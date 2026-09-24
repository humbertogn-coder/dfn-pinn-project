"""Audit DFN boundary/interface contracts without training."""

from pathlib import Path
import sys
import pytest


if __name__ == "__main__":
    print("Auditing DFN boundaries: one-sided traces, flux continuity, collector signs and single gauge.", flush=True)
    print("Analytic interface checks only; no training or full DFN validation.", flush=True)
    path = Path(__file__).resolve().parents[1]/"tests/test_dfn_boundaries.py"
    sys.exit(pytest.main([str(path), "-q", "-p", "no:cacheprovider"]))
