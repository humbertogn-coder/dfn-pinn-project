"""Explicit joint direct/inverse feasibility models with common physical metrics."""

import torch
from torch import nn

from .constitutive import ocp, exchange_current
from .kinetics import inverse_bv
from .dfn_boundary_variant import DFNBoundaryVariant
from .particle_startup_variant import install_startup_particles, particle_metadata
from .particle_confined_startup import ParticleConfinedStartup


SCHEMA = "joint_kinetics_feasibility_v1"
MODES = ("joint_direct", "joint_inverse")


class JointKinetics(DFNBoundaryVariant):
    def __init__(self, settings, mode):
        if mode not in MODES:
            raise ValueError("Unknown joint kinetic mode")
        super().__init__(settings)
        install_startup_particles(self, 42)
        self.cs = nn.ModuleList([ParticleConfinedStartup(f.raw, f.scales, float(f.initial),
                                  f.current_scale, f.duration_s) for f in self.cs])
        self.mode = mode
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def direct_residuals(self, samples):
        return super().residuals(samples)

    def residuals(self, samples):
        residuals, diagnostics = self.direct_residuals(samples)
        if self.mode == "joint_direct":
            return residuals, diagnostics
        # Direct residuals are also computed as a finite common-physics guard.
        for k, region in enumerate((0, 2)):
            p = samples[f"region_{region}"].detach().clone().requires_grad_()
            surface = torch.cat((torch.ones_like(p[:, :1]), p), dim=1)
            theta = self.cs[k](surface)
            j = self.reactions[k].integral.current_model(p)
            c_e = self.mass_scales.concentration_mol_m3*self.ce[region](p)
            temp = torch.full_like(j, self.settings["temperature_K"])
            electrode = ("n", "p")[k]
            j0 = exchange_current(c_e, theta*self.settings["cmax_mol_m3"][k], temp,
                                  electrode, mode=self.settings["kinetics_mode"])
            eta = self.charge_scales.potential_V*(self.phis[k](p)-self.phie[region](p))-ocp(theta, electrode)
            residuals[f"kinetics_{region}"] = (eta-inverse_bv(j, j0, temp))/self.charge_scales.potential_V
        return residuals, diagnostics

    def metadata(self):
        return {"mode": self.mode, "solid_boundary": self.representation_metadata(),
                "particle": particle_metadata(self), "particle_formula": "confined_sqrt_surface_linear_bulk",
                "trainable": "all", "term_count": 34}


def restore_joint(saved):
    if saved.get("schema") != SCHEMA:
        raise ValueError("Not a joint feasibility checkpoint")
    model = JointKinetics(saved["settings"], saved["mode"])
    if saved.get("metadata") != model.metadata():
        raise ValueError("Joint model metadata mismatch")
    for key, value in model.named_buffers():
        if not torch.equal(value, saved["model"][key]):
            raise ValueError(f"Fixed buffer mismatch: {key}")
    model.load_state_dict(saved["model"], strict=True)
    return model
