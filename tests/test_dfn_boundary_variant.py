"""Neural wiring and strict identity checks only."""

import copy
import pytest
import torch

from dfn_pinn.dfn_boundary_variant import DFNBoundaryVariant, SCHEMA, VARIANT, restore_boundary_model
from dfn_pinn.dfn_smoke import DFNSmoke


def test_neural_wiring_boundaries_and_gradients():
    torch.manual_seed(42)
    original = DFNSmoke()
    torch.manual_seed(42)
    model = DFNBoundaryVariant()
    for k in range(2):
        for a, b in zip(original.phis[k].net.parameters(), model.phis[k].raw.net.parameters()):
            assert torch.equal(a, b)
    samples = model.sample()
    residuals, diagnostics = model.residuals(samples)
    assert set(residuals) == set(original.residuals(samples)[0])
    assert len(residuals) == 34
    for key in ("solid_insulating_0", "solid_insulating_1", "collector_positive_solid_current",
                "collector_negative_solid_potential_gauge"):
        assert residuals[key].abs().max() < 1e-12
    assert all(torch.isfinite(v).all() for v in diagnostics.values())
    sum(v.square().mean() for v in residuals.values()).backward()
    for group in (model.ce, model.phie, model.phis, model.cs, model.reactions):
        for branch in group:
            grads = [p.grad for p in branch.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert sum(float(g.square().sum()) for g in grads) > 0


@pytest.mark.parametrize("mutation", ["variant", "representation", "buffer"])
def test_checkpoint_identity_rejects_mutations(mutation):
    model = DFNBoundaryVariant()
    saved = {"schema": SCHEMA, "variant": copy.deepcopy(VARIANT), "settings": model.settings,
             "representation": model.representation_metadata(), "model": copy.deepcopy(model.state_dict())}
    restored = restore_boundary_model(saved)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    if mutation == "variant":
        saved["variant"]["coordinate_map"] = "linear"
    elif mutation == "representation":
        saved["representation"][1]["amplitude"] *= 2
    else:
        saved["model"]["phis.1.base"] += 1
    with pytest.raises(ValueError):
        restore_boundary_model(saved)
