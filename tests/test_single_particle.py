import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("single_particle_benchmark", Path(__file__).resolve().parents[1]/"scripts/train_single_particle.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_exact_initial_center():
    value = benchmark.exact(torch.tensor(0., dtype=torch.float64), torch.tensor(0., dtype=torch.float64))
    torch.testing.assert_close(value, torch.tensor(.6, dtype=torch.float64))


def test_network_loss_has_finite_parameter_gradients():
    torch.manual_seed(42)
    model = benchmark.ParticleNet()
    r = torch.linspace(.1, 1., 5, dtype=torch.float64)
    interior = benchmark.make_points(r, r*.05)
    initial = benchmark.make_points(r, r*0)
    surface = benchmark.make_points(r*0+1, r*.05)
    sum(benchmark.loss_terms(model, interior, initial, surface)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    center = benchmark.make_points(r*0, r*.05)
    assert benchmark.center_residual(model, center).abs().max() == 0
