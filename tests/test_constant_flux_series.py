"""Check initial data and analytic mass balance independently of PyBaMM."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_constant_flux_sphere import eigenvalues, series_solution


def test_initial_concentration_is_uniform_including_boundaries():
    np.testing.assert_array_equal(series_solution([0, 0.5, 1], [0], eigenvalues(32)),
                                  np.ones((3, 1)))


def test_series_has_correct_integrated_mass_loss():
    nodes, weights = np.polynomial.legendre.leggauss(160)
    radius = (nodes + 1) / 2
    time = np.array([0.001, 0.005, 0.02])
    concentration = series_solution(radius, time, eigenvalues(128))
    mean = (1.5 * weights * radius**2) @ concentration
    np.testing.assert_allclose(mean, 1 - 3 * time, rtol=0, atol=1e-11)
