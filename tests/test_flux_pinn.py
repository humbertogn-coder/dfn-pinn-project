import importlib.util
from pathlib import Path
import sys

import torch

scripts = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("flux_pinn", scripts/"train_constant_flux_particle.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_flux_training_graph():
    torch.manual_seed(42)
    model = pilot.FluxNet()
    r = torch.linspace(.1, 1, 5, dtype=torch.float64)
    p = pilot.make_points(r, r*.01)
    initial = pilot.make_points(r, r*0)
    surface = pilot.make_points(r*0+1, r*.01)
    sum(pilot.terms(model, p, initial, surface)).backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())


def test_flux_residual_sign_and_uniform_initial():
    r = torch.linspace(.1, 1, 5, dtype=torch.float64)
    # Exact flux polynomial, intentionally NOT uniform at t=0.
    model = lambda p: 1-p[:, :1]**2/2-3*p[:, 2:3]
    pde, initial, flux = pilot.terms(model, pilot.make_points(r, r*.01),
                                   pilot.make_points(r, r*0), pilot.make_points(r*0+1, r*.01))
    assert pde.item() < 1e-25 and flux.item() < 1e-25
    assert initial.item() > 0


def test_startup_initial_and_center_are_exact():
    model = pilot.StartupFluxNet()
    r = torch.linspace(0, 1, 21, dtype=torch.float64)
    initial = pilot.make_points(r, r*0)
    assert torch.equal(model(initial), torch.ones(21, 1, dtype=torch.float64))
    for tau in (1e-8, 1e-5, .01):
        center = pilot.make_points(r*0, r*0+tau)
        assert pilot.center_residual(model, center).abs().max() == 0


def test_startup_gradients_and_matching_samples():
    torch.manual_seed(42)
    pilot.build_model("baseline")
    baseline_sample = torch.rand(512, 2, dtype=torch.float64)
    torch.manual_seed(42)
    model = pilot.build_model("startup")
    assert torch.equal(torch.rand(512, 2, dtype=torch.float64), baseline_sample)
    r = torch.linspace(.1, 1., 7, dtype=torch.float64)
    pde, ic, bc = pilot.terms(model, pilot.make_points(r, r*1e-4),
                             pilot.make_points(r, r*0), pilot.make_points(r*0+1, r*1e-4))
    assert ic.item() == 0
    (pde+ic+bc).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
