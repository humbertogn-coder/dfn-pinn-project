import pytest
import torch

from dfn_pinn.constitutive import FARADAY_CONSTANT
from dfn_pinn.particle_balance import particle_mean_from_charge, particle_mean_rate
from dfn_pinn.spherical_diffusion import ParticleScales, surface_residual


@pytest.fixture(params=[(5.86e-6, 3.3e-14, 33133.), (5.22e-6, 4e-15, 63104.)])
def scales(request):
    return ParticleScales(*request.param, time_s=3600.)


def test_current_sign_and_zero(scales):
    j = torch.tensor([-2., 0., 2.], dtype=torch.float64)
    rate = particle_mean_rate(j, scales)
    assert rate[0] > 0 and rate[1] == 0 and rate[2] < 0
    torch.testing.assert_close(rate[0], -rate[2])


def test_charge_chain_rule_and_gradients(scales):
    tau = torch.tensor([.01, .1], dtype=torch.float64, requires_grad=True)
    amplitude = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    # j(tau)=amplitude*tau; integral j dt = t_ref*amplitude*tau^2/2.
    charge = scales.time_s*amplitude*tau**2/2
    mean = particle_mean_from_charge(torch.tensor(.8, dtype=torch.float64), charge, scales)
    derivative = torch.autograd.grad(mean.sum(), tau, create_graph=True)[0]
    torch.testing.assert_close(derivative, particle_mean_rate(amplitude*tau, scales))
    sensitivity = torch.autograd.grad(mean.sum(), amplitude)[0]
    assert torch.isfinite(sensitivity) and sensitivity < 0


def test_physical_charge_does_not_depend_on_time_scale(scales):
    other = ParticleScales(scales.radius_m, scales.diffusivity_m2_s, scales.concentration_mol_m3, 1.)
    initial, charge = torch.tensor(.8, dtype=torch.float64), torch.tensor(10., dtype=torch.float64)
    torch.testing.assert_close(particle_mean_from_charge(initial, charge, scales),
                               particle_mean_from_charge(initial, charge, other))


def test_surface_sign_matches_existing_residual(scales):
    j = torch.tensor(2., dtype=torch.float64)
    b = -j*scales.radius_m/(2*FARADAY_CONSTANT*scales.diffusivity_m2_s*scales.concentration_mol_m3)
    rate = particle_mean_rate(j, scales)
    model = lambda p: .8+b*p[:, :1]**2+rate*p[:, 2:3]
    p = torch.tensor([[1., .2, .01], [1., .4, .02]], dtype=torch.float64, requires_grad=True)
    residual = surface_residual(model, p, j, scales, 2.)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)
    torch.testing.assert_close(rate, 6*b*scales.diffusion_number)


def test_invalid_charge_shape(scales):
    with pytest.raises(ValueError, match="scalar or match"):
        particle_mean_from_charge(torch.ones(2, dtype=torch.float64), torch.ones(2, 1, dtype=torch.float64), scales)


def test_nonfinite_current_rejected(scales):
    with pytest.raises(ValueError):
        particle_mean_rate(torch.tensor(float("nan"), dtype=torch.float64), scales)
