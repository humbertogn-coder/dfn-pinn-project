"""Curve-selection rules of scripts/lispan_spandb_fit.py (no network needed)."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lispan_spandb_fit as F  # noqa: E402


def _row(label, i, v0, slope=1e-3, q_start=0.0, q_end=600.0, cycle=None):
    q = np.linspace(q_start, q_end, 60)
    return {"label": label, "i_sp": i, "q_raw": q, "v_raw": v0 - slope * q, "cycle": cycle}


def test_basis_key():
    assert F.basis_key("NR") == "NR" and F.basis_key("") == "NR"
    assert F.basis_key("SPAN composite") == "SPAN" and F.basis_key("S") == "S"
    assert F.basis_key("electrode") == "electrode" and F.basis_key("electrode (tap)") is None
    assert F.basis_key("cell") is None


def test_rate_order_drops_an_activation_like_slow_curve():
    rows = [_row("0.1", 0.1, 1.70), _row("0.2", 0.2, 2.10), _row("0.5", 0.5, 2.05), _row("1", 1.0, 1.98)]
    assert F.rate_order_check(rows) == ["0.1"]
    ok = [_row("0.1", 0.1, 2.15), _row("0.2", 0.2, 2.10), _row("0.5", 0.5, 2.05)]
    assert F.rate_order_check(ok) == []
    # a single inversion between two neighbours: the slower one goes (ties)
    tie = [_row("0.1", 0.1, 2.00), _row("0.5", 0.5, 2.10), _row("1", 1.0, 1.90)]
    assert F.rate_order_check(tie) == ["0.1"]
    # within the tolerance nothing is dropped; fewer than three curves are never checked
    near = [_row("0.1", 0.1, 2.08), _row("0.2", 0.2, 2.10), _row("0.5", 0.5, 2.0)]
    assert F.rate_order_check(near) == []
    assert F.rate_order_check(rows[:2]) == []


def test_rate_order_uses_the_common_window():
    # a partial slow curve (only the tail) must not be compared outside the range it covers
    rows = [_row("0.05", 0.05, 2.2, q_start=500.0, q_end=650.0), _row("0.1", 0.1, 2.15), _row("0.5", 0.5, 2.05)]
    assert F.rate_order_check(rows) == []
