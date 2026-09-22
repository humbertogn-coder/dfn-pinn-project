from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from train_constant_flux_particle import build_model, make_points, predict, projected_backward, terms
from dfn_pinn.particle_inventory import UnitFluxInventoryProjection


def test_matching_initial_parameters_and_rng():
    torch.manual_seed(42)
    raw = build_model("startup")
    sample = torch.rand(512, 2, dtype=torch.float64)
    torch.manual_seed(42)
    projected = build_model("inventory")
    assert all(torch.equal(a, b) for a, b in zip(raw.parameters(), projected.parameters()))
    assert torch.equal(sample, torch.rand(512, 2, dtype=torch.float64))


def test_chunked_objective_and_parameter_gradients():
    torch.manual_seed(42)
    model = UnitFluxInventoryProjection(build_model("startup"), 4)
    r = torch.linspace(.1, .9, 7, dtype=torch.float64)
    interior = make_points(r, .001+r*.01)
    initial = make_points(r, r*0)
    surface = make_points(r*0+1, .001+r*.01)
    expected = terms(model, interior, initial, surface)
    (expected[0]+100*expected[1]+expected[2]).backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    actual = projected_backward(model, interior, initial, surface, batch_size=3)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    for p, expected_gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(p.grad, expected_gradient, atol=1e-10, rtol=1e-9)


def test_prediction_chunks_and_checkpoint_round_trip():
    model = UnitFluxInventoryProjection(build_model("startup"), 4)
    clone = UnitFluxInventoryProjection(build_model("startup"), 4)
    clone.load_state_dict(model.state_dict(), strict=True)
    r = torch.linspace(0, 1, 133, dtype=torch.float64)
    p = make_points(r, r*0+.01)
    with torch.no_grad():
        torch.testing.assert_close(predict(model, p), model(p))
        torch.testing.assert_close(predict(model, p), predict(clone, p))
