import math

import pytest
import torch
from scipy.optimize import brentq

from dfn_pinn.constitutive import FARADAY_CONSTANT as F
from dfn_pinn.projection import gauss_legendre
from dfn_pinn.spherical_diffusion import (
    ParticleScales, center_residual, diffusion_residual, surface_residual, volume_average,
)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def points(rho, tau=.02):
    r = tensor(rho).reshape(-1)
    return torch.stack((r, r*0+.3, r*0+tau), dim=1).requires_grad_()


@pytest.fixture(params=[(5.86e-6, 3.3e-14, 33133.), (5.22e-6, 4e-15, 63104.)])
def scales(request):
    return ParticleScales(*request.param)


def test_manufactured_polynomial_and_center(scales):
    lam = scales.diffusion_number
    model = lambda p: .6+.02*(p[:, :1]**2+6*lam*p[:, 2:3])
    result = diffusion_residual(model, points([0., 1e-8, .2, .7, 1.]), scales)
    torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-14, rtol=0)
    assert center_residual(model, points([0.])).item() == 0
    # A non-radially-smooth field must fail symmetry, even with a finite center PDE.
    assert center_residual(lambda p: p[:, :1], points([0.])).item() == 1


def test_no_flux_eigenmode(scales):
    alpha = brentq(lambda a: math.tan(a)-a, math.pi+1e-5, 1.5*math.pi-1e-5)
    def model(p):
        r = p[:, :1]
        return .5+.1*torch.sin(alpha*r)/(alpha*r)*torch.exp(-alpha**2*scales.diffusion_number*p[:, 2:3])
    # Use interior points: this expression's removable zero is not AD-safe at r=0.
    residual = diffusion_residual(model, points([.01, .1, .4, .9]), scales)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-11, rtol=0)
    boundary = surface_residual(model, points([1.]), tensor(0.), scales, 1.)
    torch.testing.assert_close(boundary, torch.zeros_like(boundary), atol=1e-10, rtol=0)


@pytest.mark.parametrize("q", [-.1, 0., .1])
def test_flux_sign_and_lithium_balance(scales, q):
    lam = scales.diffusion_number
    model = lambda p: .5-q*(p[:, :1]**2/2+3*lam*p[:, 2:3])
    j = F*scales.diffusivity_m2_s*scales.concentration_mol_m3/scales.radius_m*q
    residual = surface_residual(model, points([1.]), tensor(j), scales, 2.)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-14, rtol=0)
    r, w = gauss_legendre(8, 0., 1.)
    tau = tensor(.02).requires_grad_()
    c = .5-q*(r**2/2+3*lam*tau)
    mean = volume_average(c, r, w)
    torch.testing.assert_close(mean, .5-q*(.3+3*lam*tau), atol=1e-14, rtol=1e-14)
    dmean_dt = torch.autograd.grad(mean, tau)[0]*scales.concentration_mol_m3/scales.time_s
    torch.testing.assert_close(dmean_dt, tensor(-3*j/(F*scales.radius_m)), atol=1e-12, rtol=1e-12)


def test_parameter_gradient_and_wrong_time_factor(scales):
    coefficient = tensor(.2).requires_grad_()
    p = points([0., .2, .8])
    model = lambda x: coefficient*x[:, :1]**2+x[:, 2:3]
    residual = diffusion_residual(model, p, scales)
    torch.testing.assert_close(residual, torch.ones_like(residual)*(1-6*scales.diffusion_number*coefficient))
    gradient = torch.autograd.grad(residual.sum(), coefficient, create_graph=True)[0]
    torch.testing.assert_close(gradient, tensor(-18*scales.diffusion_number))
    loss = residual.square().mean()
    first = torch.autograd.grad(loss, coefficient, create_graph=True)[0]
    second = torch.autograd.grad(first, coefficient)[0]
    torch.testing.assert_close(second, tensor(72*scales.diffusion_number**2))


def test_constant_and_linear_time(scales):
    model = lambda p: p[:, :1]*0+.5
    assert diffusion_residual(model, points([0., .5]), scales).abs().max() == 0
    model = lambda p: p[:, 2:3]+.5
    torch.testing.assert_close(diffusion_residual(model, points([0., .5]), scales), tensor([[1.], [1.]]))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        ParticleScales(0., 1., 1.)
    s = ParticleScales(1., 1., 1.)
    model = lambda p: p[:, :1]**2
    with pytest.raises(ValueError):
        diffusion_residual(model, points([.5]).detach(), s)
    with pytest.raises(ValueError):
        diffusion_residual(lambda p: model(p).detach(), points([.5]), s)
    with pytest.raises(ValueError):
        center_residual(model, points([.1]))
    with pytest.raises(ValueError):
        surface_residual(model, points([1.], tau=0.), tensor(0.), s, 1.)
    with pytest.raises(ValueError):
        surface_residual(model, points([1.]), tensor([0.]), s, 1.)
    with pytest.raises(ValueError):
        diffusion_residual(model, points([-0.1]), s)
