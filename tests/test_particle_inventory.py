import pytest
import torch
from torch import nn

from dfn_pinn.particle_inventory import UnitFluxInventoryProjection
from dfn_pinn.projection import gauss_legendre
from dfn_pinn.spherical_diffusion import _derivatives, diffusion_residual, ParticleScales


class Polynomial(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(.7, dtype=torch.float64))

    def forward(self, p):
        return 1+p[:, 1:2]+p[:, 2:3]+self.a*p[:, 2:3]**2*p[:, :1]**2


def points():
    return torch.tensor([[0., .2, .001], [.4, .3, .01], [1., .8, .02]],
                        dtype=torch.float64, requires_grad=True)


def integrated(model, order, time=.01):
    r, w = gauss_legendre(order, 0., 1.)
    p = torch.stack((r, r*0+.3, r*0+time), dim=1)
    return (3*r*r*w*model(p).ravel()).sum()


def test_independent_mean_and_closed_form():
    raw = Polynomial()
    model = UnitFluxInventoryProjection(raw, 4)
    p = points()
    expected = 1-3*p[:, 2:3]+raw.a*p[:, 2:3]**2*(p[:, :1]**2-.6)
    torch.testing.assert_close(model(p), expected, atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(integrated(model, 32), torch.tensor(.97, dtype=torch.float64))


def test_radial_derivatives_and_time_chain_rule():
    raw = Polynomial()
    model = UnitFluxInventoryProjection(raw, 4)
    p = points()
    dr, drr, dt = _derivatives(model, p)
    old_dr, old_drr, _ = _derivatives(raw, p)
    torch.testing.assert_close(dr, old_dr)
    torch.testing.assert_close(drr, old_drr)
    torch.testing.assert_close(dt, -3+2*raw.a*p[:, 2:3]*(p[:, :1]**2-.6))
    assert dr[0].item() == 0


def test_second_time_derivative_and_parameter_graph():
    raw = Polynomial()
    model = UnitFluxInventoryProjection(raw, 4)
    p = points()
    _, _, dt = _derivatives(model, p)
    dtt = torch.autograd.grad(dt.sum(), p, create_graph=True)[0][:, 2:3]
    torch.testing.assert_close(dtt, 2*raw.a*(p[:, :1]**2-.6))
    loss = diffusion_residual(model, p, ParticleScales(1., 1., 1.)).square().mean()
    loss.backward()
    assert raw.a.grad is not None and torch.isfinite(raw.a.grad) and raw.a.grad.abs() > 0


def test_batch_independence_and_gradcheck():
    model = UnitFluxInventoryProjection(Polynomial(), 4)
    p = torch.tensor([[.4, .3, .01], [.8, .8, .02]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(model(p), torch.cat([model(row[None, :]) for row in p]))
    assert torch.autograd.gradcheck(model, (p,))
    assert torch.autograd.gradgradcheck(model, (p,))


def test_uniform_initial_value_preserved():
    class UniformInitial(nn.Module):
        def forward(self, p):
            return 1+p[:, 2:3]*(1+p[:, :1]**2)
    model = UnitFluxInventoryProjection(UniformInitial(), 4)
    p = points().detach()
    p[:, 2] = 0
    torch.testing.assert_close(model(p), torch.ones(3, 1, dtype=torch.float64), atol=1e-14, rtol=0)


def test_underresolved_quadrature_is_not_continuous_conservation():
    class HighPower(nn.Module):
        def forward(self, p):
            return 1+p[:, :1]**8
    model = UnitFluxInventoryProjection(HighPower(), 2)
    torch.testing.assert_close(integrated(model, 2), torch.tensor(.97, dtype=torch.float64))
    assert abs(integrated(model, 64).item()-.97) > .01


@pytest.mark.parametrize("order", [0, 1, True, 2.5])
def test_invalid_order(order):
    with pytest.raises(ValueError):
        UnitFluxInventoryProjection(Polynomial(), order)


def test_invalid_points():
    model = UnitFluxInventoryProjection(Polynomial(), 4)
    with pytest.raises(ValueError):
        model(torch.ones(3, 2, dtype=torch.float64))
    with pytest.raises(ValueError):
        model(points().float())
