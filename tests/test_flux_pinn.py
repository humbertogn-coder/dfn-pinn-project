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
