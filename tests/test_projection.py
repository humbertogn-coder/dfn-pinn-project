"""Physical Jacobians, signs, quadrature limits and differentiability."""

import pytest
import torch

from dfn_pinn.projection import (
    current_correction, gauss_legendre, integrated_current, project_current,
)


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


@pytest.mark.parametrize("target", [48.6855, -48.6855, 0.])
@pytest.mark.parametrize("variable_area", [False, True])
def test_balance_idempotence_and_shape(target, variable_area):
    x, w = gauss_legendre(8, 97.2e-6, 172.8e-6)
    a = 3e5 * (1+x/x[-1]) if variable_area else t(3e5)
    raw = torch.stack([torch.sin(x*1e4), 2+x*1e4])
    projected = project_current(raw, a, w, t(target))
    assert projected.shape == raw.shape
    torch.testing.assert_close(integrated_current(projected, a, w), t([target, target]),
                               atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(project_current(projected, a, w, t(target)), projected)


def test_physical_jacobian_and_signs():
    length = 85.2e-6
    x, w = gauss_legendre(4, 0., length)
    torch.testing.assert_close(w.sum(), t(length))
    torch.testing.assert_close((w*x).sum(), t(length**2/2), atol=1e-20, rtol=1e-13)
    a = t(3*.75/5.86e-6)
    target = t([5/.1027, -5/.1027, 0.])
    actual = project_current(torch.zeros(3, 4, dtype=torch.float64), a, w, target)
    torch.testing.assert_close(actual, (target/(a*length))[:, None].expand_as(actual))


def test_first_second_derivatives_all_inputs():
    raw = t([[.1, .4, -.2], [.5, -.3, .1]], True)
    # Scaled example avoids poorly resolved finite differences on micron weights.
    area, weights, target = t([1., 2., 3.], True), t([.2, .3, .5], True), t([.1, -.2], True)
    args = (raw, area, weights, target)
    assert torch.autograd.gradcheck(project_current, args)
    assert torch.autograd.gradgradcheck(project_current, args)


def test_projection_jacobian_and_time_gradient():
    w, area = t([.2, .3, .5]), t([1., 2., 3.])
    raw = t([.4, -.2, .7], True)
    jac = torch.autograd.functional.jacobian(lambda j: project_current(j, area, w, t(.8)), raw)
    expected = torch.eye(3, dtype=torch.float64) - (area*w/(area*w).sum())[None, :]
    torch.testing.assert_close(jac, expected)
    time = t(.4, True)
    out = project_current(time*raw, area, w, time**2)
    derivative = torch.autograd.grad(integrated_current(out, area, w), time)[0]
    torch.testing.assert_close(derivative, 2*time)


def test_independent_quadrature_exact_polynomial():
    x, w = gauss_legendre(4, 0., 1.)
    xx, ww = gauss_legendre(9, 0., 1.)
    correction = current_correction(x**3, 1+x, w, t(2.))
    # Reuse the same correction; recomputing it would hide quadrature error.
    checked = integrated_current(xx**3+correction, 1+xx, ww)
    torch.testing.assert_close(checked, t(2.), atol=1e-13, rtol=1e-13)


def test_independent_quadrature_exposes_underresolution():
    x, w = gauss_legendre(1, 0., 1.)
    xx, ww = gauss_legendre(8, 0., 1.)
    correction = current_correction(x**2, t(1.), w, t(0.))
    checked = integrated_current(xx**2+correction, t(1.), ww)
    torch.testing.assert_close(checked, t(1/12), atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf")])
def test_invalid_area_or_weights(bad):
    with pytest.raises(ValueError):
        project_current(t([1., 2.]), t(bad), t([.5, .5]), t(0.))
    with pytest.raises(ValueError):
        project_current(t([1., 2.]), t(1.), t([.5, bad]), t(0.))


def test_invalid_shapes_and_dtype():
    for area, weights, target in [(t([1., 2., 3.]), t([.5, .5]), t(0.)),
                                  (t(1.), t([1.]), t(0.)),
                                  (t(1.), t([.5, .5]), t([0., 0.]))]:
        with pytest.raises(ValueError):
            project_current(t([1., 2.]), area, weights, target)
    with pytest.raises(ValueError):
        project_current(t([1., 2.]), t(1.), t([.5, .5]), torch.tensor(0., dtype=torch.float32))


@pytest.mark.parametrize("order,left,right", [(0, 0., 1.), (True, 0., 1.),
                                             (2, 1., 1.), (2, 1., 0.), (2, 0., float("inf"))])
def test_invalid_quadrature(order, left, right):
    with pytest.raises(ValueError):
        gauss_legendre(order, left, right)


def test_float32_smoke():
    x, w = gauss_legendre(8, 0., 1., dtype=torch.float32)
    result = project_current(x, x*0+1, w, torch.tensor(2., dtype=torch.float32))
    assert result.dtype == torch.float32
    torch.testing.assert_close((result*w).sum(), torch.tensor(2.))
