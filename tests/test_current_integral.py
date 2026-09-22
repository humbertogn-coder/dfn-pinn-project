import pytest
import torch
from torch import nn

from dfn_pinn.current_integral import CurrentIntegral
from dfn_pinn.particle_balance import particle_mean_from_charge, particle_mean_rate
from dfn_pinn.spherical_diffusion import ParticleScales


def points():
    return torch.tensor([[.2, .1], [.7, .4]], dtype=torch.float64, requires_grad=True)


@pytest.mark.parametrize("current", [-2., 0., 2.])
def test_constant_current_units_and_sign(current):
    p = points()
    model = CurrentIntegral(lambda z: z[:, :1]*0+current, 3600.)
    torch.testing.assert_close(model(p), 3600*current*p[:, 1:2])


def test_variable_current_and_fundamental_theorem():
    p = points()
    current = lambda z: (1+z[:, :1])*(2-3*z[:, 1:2]+4*z[:, 1:2]**2)
    integral = CurrentIntegral(current, 7., order=3)
    x, t = p[:, :1], p[:, 1:2]
    expected = 7*(1+x)*(2*t-1.5*t*t+4*t**3/3)
    torch.testing.assert_close(integral(p), expected)
    gradient = torch.autograd.grad(integral(p).sum(), p, create_graph=True)[0]
    torch.testing.assert_close(gradient[:, 1:2], 7*current(p))
    torch.testing.assert_close(gradient[:, :1], expected/(1+x))
    second = torch.autograd.grad(gradient[:, 1].sum(), p)[0][:, 1:2]
    torch.testing.assert_close(second, 7*(1+x)*(-3+8*t))


def test_zero_time_value_and_right_derivative():
    p = torch.tensor([[.3, 0.]], dtype=torch.float64, requires_grad=True)
    integral = CurrentIntegral(lambda z: 2+z[:, :1]+z[:, 1:2], 10.)
    charge = integral(p)
    assert charge.item() == 0
    derivative = torch.autograd.grad(charge.sum(), p)[0][:, 1:2]
    torch.testing.assert_close(derivative, torch.tensor([[23.]], dtype=torch.float64))


@pytest.mark.parametrize("radius,diffusivity,scale", [(5.86e-6, 3.3e-14, 33133.), (5.22e-6, 4e-15, 63104.)])
def test_inventory_chain_and_learned_parameter(radius, diffusivity, scale):
    class Current(nn.Module):
        def __init__(self):
            super().__init__()
            self.amplitude = nn.Parameter(torch.tensor(2., dtype=torch.float64))

        def forward(self, z):
            return self.amplitude*(1+z[:, :1])*z[:, 1:2]
    current = Current()
    scales = ParticleScales(radius, diffusivity, scale, 3600.)
    p = points()
    integral = CurrentIntegral(current, scales.time_s)
    mean = particle_mean_from_charge(torch.tensor(.8, dtype=torch.float64), integral(p), scales)
    rate = torch.autograd.grad(mean.sum(), p, create_graph=True)[0][:, 1:2]
    torch.testing.assert_close(rate, particle_mean_rate(current(p), scales))
    derivative = torch.autograd.grad(mean.sum(), current.amplitude)[0]
    assert torch.isfinite(derivative) and derivative < 0


def test_gradcheck_batch_independence_and_smooth_nonpolynomial():
    current = lambda z: torch.exp(z[:, 1:2])*(1+z[:, :1]**2)
    integral = CurrentIntegral(current, 3., order=16)
    p = points()
    torch.testing.assert_close(integral(p), 3*torch.expm1(p[:, 1:2])*(1+p[:, :1]**2))
    torch.testing.assert_close(integral(p), torch.cat([integral(row[None, :]) for row in p]))
    assert torch.autograd.gradcheck(integral, (p,))
    assert torch.autograd.gradgradcheck(integral, (p,))


def test_underresolution_is_visible_in_charge_and_derivative():
    current = lambda z: z[:, 1:2]**4
    p = points()
    coarse = CurrentIntegral(current, 1., order=1)(p)
    fine = CurrentIntegral(current, 1., order=3)(p)
    exact = p[:, 1:2]**5/5
    torch.testing.assert_close(fine, exact)
    assert (coarse-exact).abs().max() > 1e-3
    derivative = torch.autograd.grad(coarse.sum(), p)[0][:, 1:2]
    assert (derivative-current(p)).abs().max() > .01


@pytest.mark.parametrize("reference", [0., -1., float("nan"), float("inf"), True])
def test_invalid_time_reference(reference):
    with pytest.raises(ValueError):
        CurrentIntegral(lambda z: z[:, :1], reference)


def test_invalid_inputs_and_model_shape():
    integral = CurrentIntegral(lambda z: z[:, :1], 1.)
    with pytest.raises(ValueError):
        integral(torch.zeros(2, 3, dtype=torch.float64))
    with pytest.raises(ValueError):
        integral(torch.tensor([[.2, -1.]], dtype=torch.float64))
    with pytest.raises(ValueError):
        integral(points().float())
    with pytest.raises(ValueError):
        CurrentIntegral(lambda z: z[:, 0], 1.)(points())
