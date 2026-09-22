import pytest
import torch
from torch import nn

from dfn_pinn.constitutive import FARADAY_CONSTANT
from dfn_pinn.particle_balance import particle_mean_rate
from dfn_pinn.particle_current import ParticleCurrent
from dfn_pinn.spherical_diffusion import ParticleScales, diffusion_residual


@pytest.fixture(params=[(5.86e-6, 3.3e-14, 33133.), (5.22e-6, 4e-15, 63104.)])
def scales(request):
    return ParticleScales(*request.param)


class Current(nn.Module):
    def __init__(self, sign, ramp):
        super().__init__()
        self.amplitude = nn.Parameter(torch.tensor(sign, dtype=torch.float64))
        self.ramp = ramp

    def forward(self, p):
        return self.amplitude*(1+p[:, :1])*(1+self.ramp*p[:, 1:2])


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("ramp", [0., 2.])
def test_shared_current_inventory_flux_and_gradients(scales, sign, ramp):
    current = Current(sign, ramp)
    coupled = ParticleCurrent(current, scales, order=4)
    initial = torch.tensor(.8, dtype=torch.float64)
    p = torch.tensor([[1., .2, .001], [1., .7, .002]], dtype=torch.float64, requires_grad=True)
    xt = p[:, 1:]
    mean = coupled.mean_target(xt, initial)
    expected_charge = scales.time_s*current.amplitude*(1+xt[:, :1])*(xt[:, 1:2]+ramp*xt[:, 1:2]**2/2)
    factor = 3/(FARADAY_CONSTANT*scales.radius_m*scales.concentration_mol_m3)
    torch.testing.assert_close(mean, initial-factor*expected_charge)
    grad = torch.autograd.grad(mean.sum(), p, create_graph=True)[0]
    torch.testing.assert_close(grad[:, 2:3], particle_mean_rate(current(xt), scales))

    def concentration(z):
        j = current(z[:, 1:])
        b = -j*scales.radius_m/(2*FARADAY_CONSTANT*scales.diffusivity_m2_s*scales.concentration_mol_m3)
        # rho^2 - 3/5 has zero spherical volume mean.
        return coupled.mean_target(z[:, 1:], initial)+b*(z[:, :1]**2-.6)

    residual = coupled.flux_residual(concentration, p, 2.)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)
    if ramp == 0:
        torch.testing.assert_close(diffusion_residual(concentration, p, scales),
                                   torch.zeros_like(residual), atol=1e-12, rtol=0)
    # Independent midpoint-free Gauss integration of the manufactured field.
    from dfn_pinn.projection import gauss_legendre
    r, w = gauss_legendre(12, 0., 1.)
    grid = torch.stack((r, p[0, 1].expand_as(r), p[0, 2].expand_as(r)), dim=1)
    actual_mean = (3*r**2*w*concentration(grid).flatten()).sum()
    torch.testing.assert_close(actual_mean, mean[0, 0])
    sensitivity = torch.autograd.grad(mean.sum(), current.amplitude)[0]
    torch.testing.assert_close(sensitivity, (-factor*expected_charge/current.amplitude).sum())
    assert list(coupled.parameters()) == [current.amplitude]
    assert "integral.current_model.amplitude" in coupled.state_dict()


def test_exact_inventory_does_not_enforce_flux(scales):
    current = Current(1., 2.)
    coupled = ParticleCurrent(current, scales)
    initial = torch.tensor(.8, dtype=torch.float64)
    uniform = lambda p: coupled.mean_target(p[:, 1:], initial)
    p = torch.tensor([[1., .2, .001]], dtype=torch.float64, requires_grad=True)
    residual = coupled.flux_residual(uniform, p, 1.)
    torch.testing.assert_close(residual, -current(p[:, 1:]))
    derivative = torch.autograd.grad(residual.sum(), current.amplitude)[0]
    assert derivative.abs() > 0


def test_zero_time_inventory_and_invalid_surface(scales):
    coupled = ParticleCurrent(Current(1., 0.), scales)
    initial = torch.tensor(.8, dtype=torch.float64)
    p = torch.tensor([[.2, 0.]], dtype=torch.float64)
    torch.testing.assert_close(coupled.mean_target(p, initial), initial.expand(1, 1))
    for point in ([[1., .2, 0.]], [[.5, .2, .1]], [[1., .2]]):
        with pytest.raises(ValueError):
            coupled.flux_residual(lambda z: z[:, :1], torch.tensor(point, dtype=torch.float64), 1.)
