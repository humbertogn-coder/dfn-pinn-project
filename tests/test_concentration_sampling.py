"""Check spatial interpolation without mixing time and spatial axes."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_concentration_convergence import sample_spatial


def test_linear_field_keeps_radial_spatial_and_time_axes():
    r = np.array([0., 1., 2.])
    x = np.array([0., 2., 4., 6.])
    t = np.array([0., 10.])
    values = 2 * r[:, None, None] + 3 * x[None, :, None] + t[None, None, :]
    targets = [np.array([0.5, 1.5]), np.array([1., 3., 5.])]
    actual = sample_spatial(values, [r, x], targets)
    expected = 2 * targets[0][:, None, None] + 3 * targets[1][None, :, None] + t[None, None, :]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_rejects_extrapolation():
    with pytest.raises(ValueError):
        sample_spatial(np.ones((2, 3)), [np.array([0., 1.])], [np.array([1.1])])


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        sample_spatial(np.ones((2, 3)), [np.array([0., 1., 2.])], [np.array([1.])])
