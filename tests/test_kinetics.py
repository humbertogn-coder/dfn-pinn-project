"""Independent formula, derivative, sign and numerical-domain checks."""

import numpy as np
import pybamm
import pytest
import torch

from dfn_pinn.constitutive import FARADAY_CONSTANT as F, GAS_CONSTANT as R
from dfn_pinn.kinetics import direct_bv, inverse_bv


def tensor(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


@pytest.mark.parametrize("temperature", [283.15, 298.15, 313.15])
def test_roundtrips_and_signs(temperature):
    eta = torch.linspace(-.3, .3, 61, dtype=torch.float64)[:, None]
    j0 = tensor([1e-6, .1, 1., 10.])[None, :]
    temp = tensor(temperature)
    current = direct_bv(eta, j0, temp)
    torch.testing.assert_close(inverse_bv(current, j0, temp), eta.expand_as(current), rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(direct_bv(-eta, j0, temp), -current)
    assert torch.equal(torch.sign(current), torch.sign(eta).expand_as(current))
    imposed = tensor([-100., -1., 0., 1., 100.])[:, None]
    recovered = direct_bv(inverse_bv(imposed, j0, temp), j0, temp)
    torch.testing.assert_close(recovered, imposed.expand_as(recovered), rtol=1e-12, atol=1e-12)


def test_pybamm_expression():
    assert F == float(pybamm.constants.F.value)
    assert R == float(pybamm.constants.R.value)
    eta = np.linspace(-.2, .2, 31)
    expected = np.array([float((2 * .7 * pybamm.sinh(
        pybamm.constants.F * e / (2 * pybamm.constants.R * 298.15))).evaluate()) for e in eta])
    np.testing.assert_allclose(direct_bv(tensor(eta), tensor(.7), tensor(298.15)).numpy(),
                               expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("function", [direct_bv, inverse_bv])
def test_first_second_and_mixed_derivatives(function):
    signal = tensor([-.03, 0., .06], True)
    j0 = tensor([.2, .7, 3.], True)
    temp = tensor(298.15, True)
    assert torch.autograd.gradcheck(function, (signal, j0, temp))
    assert torch.autograd.gradgradcheck(function, (signal, j0, temp))


def test_analytic_slopes_and_zero():
    eta = tensor([-.1, 0., .1], True)
    j0, temp = tensor(.8), tensor(298.15)
    beta = F / (2*R*temp)
    current = direct_bv(eta, j0, temp)
    slope = torch.autograd.grad(current.sum(), eta)[0]
    torch.testing.assert_close(slope, 2*j0*beta*torch.cosh(beta*eta))
    j = tensor([-2., 0., 2.], True)
    overpotential = inverse_bv(j, j0, temp)
    inverse_slope = torch.autograd.grad(overpotential.sum(), j)[0]
    torch.testing.assert_close(inverse_slope, 1/(beta*torch.sqrt(j*j+4*j0*j0)))
    assert direct_bv(tensor(0.), j0, temp).item() == 0
    assert inverse_bv(tensor(0.), j0, temp).item() == 0


@pytest.mark.parametrize("function", [direct_bv, inverse_bv])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_positive_inputs(function, value):
    with pytest.raises(ValueError):
        function(tensor(0.), tensor(value), tensor(298.15))
    with pytest.raises(ValueError):
        function(tensor(0.), tensor(1.), tensor(value))


@pytest.mark.parametrize("function", [direct_bv, inverse_bv])
def test_invalid_signal_and_types(function):
    for value in [float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            function(tensor(value), tensor(1.), tensor(298.15))
    with pytest.raises(TypeError):
        function(0., tensor(1.), tensor(298.15))
    with pytest.raises(ValueError):
        function(tensor(0.), torch.tensor(1., dtype=torch.float32), tensor(298.15))


def test_explicit_overflow_failures():
    with pytest.raises(FloatingPointError):
        direct_bv(tensor(100.), tensor(1.), tensor(298.15))
    with pytest.raises(FloatingPointError):
        inverse_bv(tensor(1e300), tensor(1e-300), tensor(298.15))


def test_small_exchange_current_and_large_ratio():
    j0, temp = tensor(1e-12), tensor(298.15)
    eta = inverse_bv(tensor(1.), j0, temp)
    assert torch.isfinite(eta)
    torch.testing.assert_close(direct_bv(eta, j0, temp), tensor(1.), rtol=1e-12, atol=1e-12)
    # Inversion does not remove sensitivity to depletion.
    j = tensor(0., True)
    slope = torch.autograd.grad(inverse_bv(j, j0, temp), j)[0]
    assert slope > 1e10


@pytest.mark.parametrize("function", [direct_bv, inverse_bv])
def test_float32_shape_and_gradients(function):
    signal = torch.tensor([[-.01], [0.], [.01]], dtype=torch.float32, requires_grad=True)
    result = function(signal, torch.tensor([.2, 1.]), torch.tensor(298.15))
    assert result.shape == (3, 2) and result.dtype == signal.dtype and result.device == signal.device
    result.sum().backward()
    assert torch.isfinite(signal.grad).all()
