"""Strict metadata and frozen-branch tests, without training."""

import copy
import pytest
import torch
from dfn_pinn.dfn_boundary_variant import DFNBoundaryVariant, VARIANT
from dfn_pinn.particle_startup_variant import (
    SCHEMA, REPRESENTATION, install_startup_particles, particle_metadata, restore_startup,
)


@pytest.mark.parametrize("mutation", [None, "scale", "buffer", "trainable"])
def test_restore_and_frozen_identity(mutation):
    model = DFNBoundaryVariant()
    frozen = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("cs.")}
    install_startup_particles(model)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in frozen.items())
    assert all(p.requires_grad == n.startswith("cs.") for n, p in model.named_parameters())
    saved = {"schema": SCHEMA, "variant": VARIANT, "settings": model.settings,
        "particle_seed": 42, "particle_representation": REPRESENTATION, "trainable_prefix": "cs.",
        "representation": model.representation_metadata(), "particle_metadata": particle_metadata(model),
        "model": copy.deepcopy(model.state_dict())}
    if mutation == "scale":
        saved["particle_metadata"][0]["beta"] *= 2
    elif mutation == "buffer":
        saved["model"]["cs.0.initial"] += .01
    elif mutation == "trainable":
        saved["trainable_prefix"] = "reactions."
    if mutation:
        with pytest.raises(ValueError):
            restore_startup(saved)
    else:
        replay = restore_startup(saved)
        for key, value in model.state_dict().items():
            assert torch.equal(value, replay.state_dict()[key])
