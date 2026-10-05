"""Explicit startup particle wiring for a frozen-current diagnostic only."""

from dataclasses import asdict
import torch
from torch import nn

from .coupled_training import network
from .dfn_boundary_variant import DFNBoundaryVariant, VARIANT
from .particle_startup_candidate import ParticleStartupCandidate


SCHEMA = "particle_startup_smoke_v1"
REPRESENTATION = {"name": "physical_sqrt_time_layer_v1", "inputs": 4,
                  "hard_inventory": False, "hard_surface_flux": False, "bounded_output": False}


def install_startup_particles(model, seed=42):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model.cs = nn.ModuleList([ParticleStartupCandidate(
            network(4, model.settings["hidden_widths"]), model.reactions[k].scales,
            model.settings["initial_stoichiometries"][k], model.jref[k],
            model.settings["duration_s"]) for k in range(2)])
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("cs."))
    return model


def particle_metadata(model):
    return [{"scales": asdict(f.scales), "beta": f.beta, "fo_end": f.fo_end,
             "initial": float(f.initial), "current_scale": f.current_scale,
             "duration_s": f.duration_s} for f in model.cs]


def restore_startup(saved):
    if saved.get("schema") != SCHEMA or saved.get("variant") != VARIANT or saved.get("particle_representation") != REPRESENTATION:
        raise ValueError("Unsupported startup diagnostic identity")
    if saved.get("trainable_prefix") != "cs.":
        raise ValueError("Only particle parameters may be trainable")
    model = install_startup_particles(DFNBoundaryVariant(saved["settings"]), saved["particle_seed"])
    if saved.get("representation") != model.representation_metadata() or saved.get("particle_metadata") != particle_metadata(model):
        raise ValueError("Representation metadata mismatch")
    for name, value in model.named_buffers():
        if name not in saved["model"] or not torch.equal(value, saved["model"][name]):
            raise ValueError(f"Fixed buffer mismatch: {name}")
    model.load_state_dict(saved["model"], strict=True)
    return model
