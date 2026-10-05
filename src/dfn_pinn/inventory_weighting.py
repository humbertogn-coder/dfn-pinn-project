"""Isolated fixed inventory weighting; physical residuals stay unchanged."""

import math
from .dfn_run_contract import adam_step
from .joint_kinetics import restore_joint

SCHEMA = "joint_inventory_weight_v1"
WEIGHTS = {"control": 1.0, "weighted": 1e8}
KEYS = {"inventory_0", "inventory_2"}


def weighted_step(model, optimizer, samples, weight):
    if weight not in WEIGHTS.values() or model.mode != "joint_direct":
        raise ValueError("Only pinned direct-kinetics weights are supported")

    class Objective:
        def __getattr__(self, name):
            return getattr(model, name)

        def residuals(self, points):
            residuals, diagnostics = model.residuals(points)
            if {k for k in residuals if k.startswith("inventory_")} != KEYS:
                raise ValueError("Unexpected inventory terms")
            self.raw = {k: float(v.detach().square().mean()) for k, v in residuals.items()}
            return {k: v * math.sqrt(weight) if k in KEYS and weight != 1 else v
                    for k, v in residuals.items()}, diagnostics

    objective = Objective()
    history = adam_step(objective, optimizer, samples)
    history["raw_losses"] = objective.raw
    return history


def restore_weighted(saved):
    if saved.get("schema") != SCHEMA or saved.get("arm") not in WEIGHTS:
        raise ValueError("Unknown inventory-weight checkpoint")
    if saved.get("inventory_weight") != WEIGHTS[saved["arm"]]:
        raise ValueError("Inventory weight metadata mismatch")
    model = restore_joint(saved["joint"])
    if model.mode != "joint_direct":
        raise ValueError("Direct kinetics required")
    return model
