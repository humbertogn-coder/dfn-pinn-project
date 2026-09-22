from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from check_ramp_flux_sphere import ramp_solution, SLOPE
from check_constant_flux_sphere import eigenvalues, series_solution


def test_initial_and_zero_flux():
    roots = eigenvalues(128)
    r = np.linspace(0, 1, 21)
    assert np.array_equal(ramp_solution(r, [0], roots), np.ones((21, 1)))
    assert np.array_equal(ramp_solution(r, [0, .02], roots, slope=0), np.ones((21, 2)))


def test_time_derivative_is_step_response():
    r = np.array([0., .4, 1.])
    roots = eigenvalues(512)
    t, h = .01, 1e-7
    derivative = (ramp_solution(r, [t+h], roots)-ramp_solution(r, [t-h], roots))/(2*h)
    np.testing.assert_allclose(derivative, SLOPE*(series_solution(r, [t], roots)-1), rtol=1e-6, atol=1e-8)


def test_independent_mean_and_sign():
    r, w = np.polynomial.legendre.leggauss(512)
    r, w = (r+1)/2, w/2
    t = np.array([0., 1e-5, .001, .02])
    c = ramp_solution(r, t, eigenvalues(128))
    np.testing.assert_allclose((3*r*r*w) @ c, 1-1.5*SLOPE*t*t, atol=1e-12, rtol=0)
    assert ramp_solution([1.], [.02], eigenvalues(128))[0, 0] < 1


def test_center_and_surface_series_refinement():
    roots = eigenvalues(4096)
    t = np.r_[0., np.geomspace(1e-5, .02, 30)]
    fine = ramp_solution([0., .1, 1.], t, roots)
    coarse = ramp_solution([0., .1, 1.], t, roots[:2048])
    assert np.max(np.abs(fine-coarse)) < 1e-9


def test_surface_gradient():
    roots = eigenvalues(1024)
    h, t = 1e-5, .01
    c = ramp_solution([1-2*h, 1-h, 1.], [t], roots)[:, 0]
    derivative = (3*c[2]-4*c[1]+c[0])/(2*h)
    np.testing.assert_allclose(derivative, -SLOPE*t, atol=2e-7, rtol=0)


@pytest.mark.parametrize("r,t", [([-1], [0]), ([1], [-1]), ([np.nan], [0]), ([], [0])])
def test_invalid_domain(r, t):
    with pytest.raises(ValueError):
        ramp_solution(r, t, eigenvalues(4))
