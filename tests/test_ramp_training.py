from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from train_ramp_flux_particle import RampInventoryProjection, target_mean, flux, backward_terms, build_ramp_model
from train_constant_flux_particle import build_model, make_points, SCALES
from dfn_pinn.projection import gauss_legendre
from dfn_pinn.spherical_diffusion import _derivatives, diffusion_residual


def test_target_derivative_matches_flux():
    t = torch.tensor([0., .001, .02], dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(target_mean(t).sum(), t)[0]
    torch.testing.assert_close(derivative, -3*flux(t))
    assert abs(target_mean(t)[-1].item()-.97) < 1e-14


def test_ramp_mean_and_derivatives():
    model = RampInventoryProjection(lambda p: 1+p[:, :1]**2*p[:, 2:3], 4)
    r, w = gauss_legendre(20, 0., 1.)
    p = make_points(r, r*0+.01)
    torch.testing.assert_close((3*r*r*w*model(p).ravel()).sum(), target_mean(torch.tensor(.01, dtype=torch.float64)))
    dr, _, dt = _derivatives(model, p)
    torch.testing.assert_close(dr, 2*r[:, None]*.01)
    torch.testing.assert_close(dt, -150*p[:, 2:3]+p[:, :1]**2-.6)


def test_chunked_ramp_gradients_and_flux_target():
    model = RampInventoryProjection(build_model("startup"), 4)
    r = torch.linspace(.1, .9, 7, dtype=torch.float64)
    interior, initial, surface = make_points(r, r*.01), make_points(r, r*0), make_points(r*0+1, r*.01)
    dr = torch.autograd.grad(model(surface).sum(), surface, create_graph=True)[0][:, :1]
    losses = (diffusion_residual(model, interior, SCALES).square().mean(),
              (model(initial)-1).square().mean(), (dr+flux(surface[:, 2:3])).square().mean())
    (losses[0]+100*losses[1]+losses[2]).backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    actual = backward_terms(model, interior, initial, surface, batch_size=3)
    for a, b in zip(actual, losses):
        torch.testing.assert_close(a, b)
    for p, expected in zip(model.parameters(), grads):
        torch.testing.assert_close(p.grad, expected, rtol=1e-9, atol=1e-10)


def test_initial_parameters_and_rng_match_constant_control():
    torch.manual_seed(42)
    old = build_model("inventory")
    random = torch.rand(10)
    torch.manual_seed(42)
    new = build_ramp_model()
    assert all(torch.equal(a, b) for a, b in zip(old.parameters(), new.parameters()))
    assert torch.equal(random, torch.rand(10))
