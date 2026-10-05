"""Necessary analytic and autograd checks, no optimization."""

import pytest
import torch
from torch import nn

from dfn_pinn.coupled_training import network
from dfn_pinn.dfn_smoke import DFNSmoke
from dfn_pinn.particle_confined_startup import ParticleConfinedStartup
from dfn_pinn.spherical_diffusion import _gradient, center_residual, diffusion_residual, surface_residual


class ConstantRaw(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float64))

    def forward(self, p):
        return self.value+p[:, :1]*0


def build(k, raw):
    source = DFNSmoke()
    return ParticleConfinedStartup(raw, source.reactions[k].scales,
        source.settings["initial_stoichiometries"][k], source.jref[k], 1.)


@pytest.mark.parametrize("k,sign", [(0, 1.), (1, -1.)])
def test_initial_symmetry_flux_and_bulk_derivative(k, sign):
    field = build(k, ConstantRaw(-2*sign))
    times = torch.logspace(-8, 0, 17, dtype=torch.float64)/field.scales.time_s
    initial = torch.stack((torch.linspace(0, 1, 17, dtype=torch.float64), times*0+.5, times*0), dim=1)
    torch.testing.assert_close(field(initial), field.initial.expand(17, 1), rtol=0, atol=0)
    surface = torch.stack((times*0+1, times*0+.5, times), dim=1).requires_grad_()
    residual = surface_residual(field, surface, surface.new_tensor(sign*field.current_scale), field.scales, field.current_scale)
    assert residual.abs().max() < 1e-12
    derivative = torch.autograd.grad(residual.sum(), field.raw.value)[0]
    torch.testing.assert_close(derivative, derivative.new_tensor(-len(times)/2), rtol=1e-12, atol=1e-12)
    center = torch.stack((times*0, times*0+.5, times), dim=1).requires_grad_()
    assert center_residual(field, center).abs().max() == 0
    bulk, layer = field.contributions(center)
    bulk_dt = _gradient(bulk, center)[:, 2:3]
    expected = -2*sign*field.beta*field.scales.diffusion_number
    torch.testing.assert_close(bulk_dt, torch.full_like(bulk_dt, expected), rtol=1e-12, atol=1e-12)
    assert layer[:8].abs().max() < 1e-100
    assert bool((sign*(field(surface)-field.initial) < 0).all())
    # Boundary satisfaction is not diffusion accuracy or an inventory guarantee.
    assert diffusion_residual(field, surface, field.scales).abs().max() > .01


@pytest.mark.parametrize("k", [0, 1])
def test_neural_derivatives_and_bulk_startup_limit(k):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        field = build(k, network(4, [12, 12]))
    times = torch.logspace(-6, 0, 9, dtype=torch.float64)/field.scales.time_s
    for rho in (0., .5, .999, 1.):
        p = torch.stack((times*0+rho, times*0+.5, times), dim=1).requires_grad_()
        residual = diffusion_residual(field, p, field.scales)
        assert torch.isfinite(residual).all()
        assert torch.isfinite(_gradient(_gradient(field(p), p)[:, 2:3], p)).all()
        gradients = torch.autograd.grad(residual.square().mean(), tuple(field.parameters()))
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.square().sum()) for g in gradients) > 0
    # A smooth bulk map evaluated at linear time has a finite right derivative.
    early = torch.tensor([[.5, .5, 1e-10/field.scales.time_s], [.5, .5, 1e-8/field.scales.time_s]], dtype=torch.float64).requires_grad_()
    bulk = field.contributions(early)[0]
    dt = _gradient(bulk, early)[:, 2]
    features0 = torch.tensor([[.25, .5, 0., 0.]], dtype=torch.float64)
    limit = field.beta*field.scales.diffusion_number*field.raw(features0).squeeze()
    torch.testing.assert_close(dt, limit.expand_as(dt), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("rho,time", [(-.1, .01), (.5, -.01)])
def test_invalid_coordinates(rho, time):
    field = build(0, ConstantRaw(1.))
    with pytest.raises(ValueError):
        field(torch.tensor([[rho, .5, time]], dtype=torch.float64))
