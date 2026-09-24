import json
from pathlib import Path
import pytest
import torch
import pybamm

from dfn_pinn.potential_particle import PotentialParticle
from dfn_pinn.coupled_training import sample_points
from dfn_pinn.spherical_diffusion import center_residual


def config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/potential_particle_v1.json").read_text())


def test_hard_initial_conditions_and_center():
    model = PotentialParticle(config())
    p = torch.tensor([[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.]], dtype=torch.float64, requires_grad=True)
    concentration = model.concentration(p)
    torch.testing.assert_close(concentration, torch.full_like(concentration, .5), atol=1e-15, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(concentration.sum(), p)[0], torch.zeros_like(p), atol=0, rtol=0)
    current = model.current.integral.current_model(p[:, 1:])
    torch.testing.assert_close(current, torch.zeros_like(current), atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(current.sum(), p)[0], torch.zeros_like(p), atol=0, rtol=0)
    center = torch.tensor([[0., 0., .001]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(center_residual(model.concentration, center), torch.zeros(1, 1, dtype=torch.float64), atol=0, rtol=0)


def test_potential_matches_independent_protocol_and_is_fixed():
    c = config()
    model = PotentialParticle(c)
    seconds = torch.tensor([0., .1, 2., 5., 10.], dtype=torch.float64)
    p = torch.stack((seconds*0, seconds/c["time_reference_s"]), dim=1)
    actual = model.fields(p)["phi_s"]
    ocp = pybamm.ParameterValues("Chen2020")["Negative electrode OCP [V]"]
    equilibrium = float(ocp(pybamm.Scalar(.5)).evaluate())
    expected = equilibrium+.005*(1-torch.cos(torch.pi*seconds/10))/2
    torch.testing.assert_close(actual.flatten(), expected, atol=1e-14, rtol=0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.01)
    torch.testing.assert_close(model.fields(p)["phi_s"], actual, atol=0, rtol=0)


def test_losses_have_gradients_without_reference_files():
    c = config()
    c["reference_run"] = "deliberately_missing"
    torch.manual_seed(c["seed"])
    model = PotentialParticle(c)
    losses = model.losses(sample_points(c, smoke=True))
    assert set(losses) == {"pde", "surface_flux", "kinetics", "inventory"}
    sum(losses[k]*c["loss_weights"][k] for k in losses).backward()
    for group in (model.concentration.parameters(), model.current.parameters()):
        gradients = [p.grad for p in group]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_wrong_kinetics_mode_rejected():
    c = config()
    c["exchange_current_mode"] = "pybamm_26_8"
    with pytest.raises(ValueError, match="raw"):
        PotentialParticle(c)
