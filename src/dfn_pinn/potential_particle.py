"""Potential-driven particle models; numerical reference is validation-only."""

import torch
from torch import nn

from .constitutive import ocp
from .coupled_training import CoupledParticle, network
from .particle_current import ParticleCurrent
from .projection import gauss_legendre
from .spherical_diffusion import ParticleScales


class PotentialConcentration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = network(2, config["hidden_widths"])

    def forward(self, p):
        c = self.config
        s = p[:, 2:3]*c["time_reference_s"]/c["duration_s"]
        initial = torch.full_like(s, c["initial_mean"])
        correction = self.net(torch.cat((p[:, :1]**2, s), dim=1))
        return torch.sigmoid(torch.logit(initial)+s.square()*correction)


class PotentialCurrent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = network(1, config["hidden_widths"])

    def forward(self, p):
        c = self.config
        s = p[:, 1:2]*c["time_reference_s"]/c["duration_s"]
        return c["current_scale_A_m2"]*s.square()*self.net(s)


class PotentialParticle(CoupledParticle):
    """Reuse physical losses, replacing the manufactured fields and networks.

    s^2 enforces zero initial current and zero initial concentration rate.
    No positivity constraint is applied to current. No reference data is loaded.
    This implementation is scoped to the documented negative-electrode case.
    """

    def __init__(self, config):
        nn.Module.__init__(self)
        if config["electrode"] != "n" or config["exchange_current_mode"] != "raw":
            raise ValueError("Potential particle v1 requires negative electrode and raw kinetics")
        if config["initial_mean"] != .5 or config["concentration_scale_mol_m3"] != 33133.:
            raise ValueError("Potential particle v1 requires initial mean 0.5 and c_ref=c_max")
        self.config = config
        self.concentration = PotentialConcentration(config)
        scales = ParticleScales(config["radius_m"], config["diffusivity_m2_s"],
                                config["concentration_scale_mol_m3"], config["time_reference_s"])
        self.current = ParticleCurrent(PotentialCurrent(config), scales, config["time_quadrature_order"])
        r, w = gauss_legendre(config["radial_quadrature_order"], 0., 1.)
        self.register_buffer("radial_nodes", r)
        self.register_buffer("radial_weights", 3*r*r*w)

    def fields(self, p):
        c = self.config
        s = p[:, 1:2]*c["time_reference_s"]/c["duration_s"]
        theta0 = torch.full_like(s, c["initial_mean"])
        phie = torch.full_like(s, c["electrolyte_potential_V"])
        return {"c_e": torch.full_like(s, c["electrolyte_concentration_mol_m3"]),
                "T": torch.full_like(s, c["temperature_K"]), "phi_e": phie,
                "phi_s": phie+ocp(theta0, "n")+c["potential_amplitude_V"]*(1-torch.cos(torch.pi*s))/2}
