from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from train_constant_flux_particle import mass_loss, StartupFluxNet, END_TIME


def test_exact_spherical_mean_and_flux_sign():
    times = torch.tensor([1e-5, .001, .02], dtype=torch.float64)
    # Mean(r^2) = 3/5. This polynomial has exact mean and outward unit flux.
    model = lambda p: 1-3*p[:, 2:3]+.3-p[:, :1]**2/2
    assert mass_loss(model, times).item() < 1e-25


def test_normalization_and_parameter_gradient():
    offset = torch.tensor(.01, dtype=torch.float64, requires_grad=True)
    model = lambda p: 1-3*p[:, 2:3]+offset
    loss = mass_loss(model, torch.tensor([.001, .02], dtype=torch.float64))
    torch.testing.assert_close(loss, (offset/(3*END_TIME))**2)
    loss.backward()
    torch.testing.assert_close(offset.grad, 2*offset.detach()/(3*END_TIME)**2)


def test_network_gradients_and_no_rng_consumption():
    model = StartupFluxNet()
    state = torch.get_rng_state().clone()
    mass_loss(model, torch.tensor([1e-5, .001, .02], dtype=torch.float64)).backward()
    assert torch.equal(state, torch.get_rng_state())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
