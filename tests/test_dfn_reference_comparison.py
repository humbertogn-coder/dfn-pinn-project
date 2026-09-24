"""Focused checks for native-grid comparison helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "dfn_comparison", Path(__file__).resolve().parents[1] / "scripts/compare_dfn_reference.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_scaled_metrics_and_invalid_data():
    result = comparison.metrics(np.array([1., 3.]), np.array([1., 2.]), 2., .5)
    assert result["max_scaled_error"] == .5
    assert result["rms_error"] == pytest.approx(np.sqrt(.5))
    assert not result["pass"]
    with pytest.raises(ValueError):
        comparison.metrics(np.array([np.nan]), np.array([0.]), 1., 1.)


def test_chunked_coordinate_order_and_scaling():
    points = np.arange(27000, dtype=float).reshape(9000, 3) / 27000
    def field(p):
        assert p.dtype == torch.float64
        return p[:, :1] + 2*p[:, 1:2] + 3*p[:, 2:3]
    result = comparison.evaluate(field, points, 4.)
    np.testing.assert_allclose(result, 4*(points[:, 0]+2*points[:, 1]+3*points[:, 2]))
