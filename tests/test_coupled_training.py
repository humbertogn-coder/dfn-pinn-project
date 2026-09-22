import json
from pathlib import Path

import torch

from dfn_pinn.coupled_training import CoupledParticle, sample_points, reference
from dfn_pinn.spherical_diffusion import center_residual


def build():
    c = json.loads((Path(__file__).resolve().parents[1]/"configs/coupled_particle_v1.json").read_text())
    torch.manual_seed(c["seed"])
    return CoupledParticle(c), c


def test_initial_center_and_fixed_fields():
    model, c = build()
    p = torch.tensor([[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(model.concentration(p), reference(p, c), atol=1e-15, rtol=0)
    center = torch.tensor([[0., 0., .001]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(center_residual(model.concentration, center), torch.zeros(1, 1, dtype=torch.float64))
    query = center[:, 1:]
    before = model.fields(query)["phi_s"].detach().clone()
    with torch.no_grad():
        next(model.concentration.parameters()).add_(.1)
    torch.testing.assert_close(before, model.fields(query)["phi_s"])


def test_both_networks_receive_gradients_and_replay():
    model, c = build()
    points = sample_points(c, True)
    losses = model.losses(points)
    sum(losses.values()).backward()
    for group in (model.concentration.parameters(), model.current.parameters()):
        gradients = [p.grad for p in group]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    other = CoupledParticle(c)
    other.load_state_dict(model.state_dict())
    torch.testing.assert_close(model.concentration(points["interior"]), other.concentration(points["interior"]), rtol=0, atol=0)
