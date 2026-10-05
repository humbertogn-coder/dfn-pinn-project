"""Explicit experimental solid-boundary variant; historical factories unchanged."""

import torch
from torch import nn

from .dfn_smoke import DFNSmoke
from .solid_boundary_potential import SolidBoundaryPotential


VARIANT = {"name": "direct_bv_solid_boundary_v1", "current": "fixed_positive",
           "coordinate_map": "cubic_smoothstep", "negative_gauge": "hard_zero",
           "source_projection": False}
SCHEMA = "dfn_boundary_smoke_v1"


class RawSolidCorrection(nn.Module):
    def __init__(self, field):
        super().__init__()
        self.net = field.net
        self.time_factor = field.time_factor

    def forward(self, points):
        return self.net(torch.cat((points[:, :1], points[:, 1:2]*self.time_factor), dim=1))


class DFNBoundaryVariant(DFNSmoke):
    def __init__(self, settings=None):
        super().__init__(settings)
        fields = []
        for k, region in enumerate((0, 2)):
            old = self.phis[k]
            fields.append(SolidBoundaryPotential(RawSolidCorrection(old), self.charge_scales,
                self.settings["solid_conductivities_S_m"][k], self.bounds[region],
                ("n", "p")[k], float(old.base)))
        self.phis = nn.ModuleList(fields)

    def representation_metadata(self):
        return [{"bounds": [f.left, f.right], "amplitude": f.amplitude,
                 "base": float(f.base), "time_factor": f.raw.time_factor,
                 "electrode": f.electrode} for f in self.phis]


def restore_boundary_model(saved):
    if saved.get("schema") != SCHEMA or saved.get("variant") != VARIANT:
        raise ValueError("Unsupported experimental boundary checkpoint")
    model = DFNBoundaryVariant(saved["settings"])
    if saved.get("representation") != model.representation_metadata():
        raise ValueError("Boundary representation metadata mismatch")
    buffers = dict(model.named_buffers())
    for name, expected in buffers.items():
        if name not in saved["model"] or not torch.equal(saved["model"][name], expected):
            raise ValueError(f"Fixed buffer mismatch: {name}")
    model.load_state_dict(saved["model"], strict=True)
    return model
