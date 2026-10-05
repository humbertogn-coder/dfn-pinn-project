"""Analytic scales/signs and neural derivatives; no optimization."""

import pytest
import torch
from torch import nn

from dfn_pinn.coupled_training import network
from dfn_pinn.dfn_smoke import DFNSmoke
from dfn_pinn.particle_startup_candidate import ParticleStartupCandidate
from dfn_pinn.spherical_diffusion import _gradient, center_residual, diffusion_residual, surface_residual


class LayerFixture(nn.Module):
    def __init__(self, sign):
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(-2.*sign, dtype=torch.float64))

    def forward(self, features):
        return self.coefficient*features[:, 3:4]


def candidate(k, raw):
    model = DFNSmoke()
    return ParticleStartupCandidate(raw, model.reactions[k].scales,
        model.settings["initial_stoichiometries"][k], model.jref[k], model.settings["duration_s"])


@pytest.mark.parametrize("k,sign", [(0, 1.), (1, -1.)])
def test_initial_center_and_signed_flux(k, sign):
    field = candidate(k, LayerFixture(sign))
    t = torch.logspace(-6, 0, 13, dtype=torch.float64)/field.scales.time_s
    initial = torch.stack((torch.linspace(0, 1, 13, dtype=torch.float64), torch.full_like(t, .5), t*0), dim=1).requires_grad_()
    torch.testing.assert_close(field(initial), field.initial.expand(13, 1), rtol=0, atol=0)
    surface = torch.stack((torch.ones_like(t), torch.full_like(t, .5), t), dim=1).requires_grad_()
    flux = surface_residual(field, surface, surface.new_tensor(sign*field.current_scale), field.scales, field.current_scale)
    assert flux.abs().max() < 1e-12
    sensitivity = torch.autograd.grad(flux.sum(), field.raw.coefficient)[0]
    torch.testing.assert_close(sensitivity, sensitivity.new_tensor(-len(t)/2), rtol=1e-12, atol=1e-12)
    center = torch.stack((t*0, torch.full_like(t, .5), t), dim=1).requires_grad_()
    assert center_residual(field, center).abs().max() == 0
    values = field(surface).detach()[:, 0]-field.initial
    assert bool((sign*values < 0).all())
    # This fixture satisfies the flux, but is deliberately not a diffusion solution.
    assert diffusion_residual(field, surface, field.scales).abs().max() > .01


@pytest.mark.parametrize("k", [0, 1])
def test_neural_positive_time_derivatives_and_limit(k):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        field = candidate(k, network(4, [12, 12]))
    t = torch.logspace(-6, 0, 9, dtype=torch.float64)/field.scales.time_s
    for rho in (0., .5, .999, 1.):
        p = torch.stack((torch.full_like(t, rho), torch.full_like(t, .5), t), dim=1).requires_grad_()
        residual = diffusion_residual(field, p, field.scales)
        assert torch.isfinite(residual).all()
        mixed = _gradient(_gradient(field(p), p)[:, 2:3], p)
        assert torch.isfinite(mixed).all()
        grads = torch.autograd.grad(residual.square().mean(), tuple(field.parameters()), allow_unused=False)
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.square().sum()) for g in grads) > 0
    p = torch.tensor([[1., .5, 1e-12/field.scales.time_s]], dtype=torch.float64)
    assert float((field(p)-field.initial).abs().detach()) < 1e-6
    center = torch.stack((t*0, torch.full_like(t, .5), t), dim=1).requires_grad_()
    assert center_residual(field, center).abs().max() == 0


@pytest.mark.parametrize("current,duration", [(0., 1.), (-1.,1.), (1.,0.), (float("nan"),1.)])
def test_invalid_scales(current, duration):
    source = DFNSmoke()
    with pytest.raises(ValueError):
        ParticleStartupCandidate(LayerFixture(1), source.reactions[0].scales, .8, current, duration)
