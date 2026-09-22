"""Manufactured single-particle training components; not a full DFN."""

import torch
from torch import nn

from .constitutive import FARADAY_CONSTANT, ocp, exchange_current
from .kinetics import inverse_bv
from .particle_current import ParticleCurrent
from .particle_interface import particle_interface
from .projection import gauss_legendre
from .spherical_diffusion import ParticleScales, diffusion_residual


def network(inputs, widths):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(inputs, width), nn.Tanh()))
        inputs = width
    layers.append(nn.Linear(inputs, 1))
    return nn.Sequential(*layers).double()


def reference(p, c):
    j, r, d, cref = (c[k] for k in ("reference_current_A_m2", "radius_m",
                                    "diffusivity_m2_s", "concentration_scale_mol_m3"))
    b = -j*r/(2*FARADAY_CONSTANT*d*cref)
    k = -3*c["time_reference_s"]*j/(FARADAY_CONSTANT*r*cref)
    return c["initial_mean"]+k*p[:, 2:3]+b*(p[:, :1]**2-.6)


class Concentration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = network(2, config["hidden_widths"])

    def forward(self, p):
        c = self.config
        s = p[:, 2:3]*c["time_reference_s"]/c["duration_s"]
        initial_points = torch.cat((p[:, :2], p[:, 2:3]*0), dim=1)
        initial = reference(initial_points, c)
        return torch.sigmoid(torch.logit(initial)+s*self.net(torch.cat((p[:, :1]**2, s), dim=1)))


class Current(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = network(1, config["hidden_widths"])

    def forward(self, p):
        c = self.config
        return c["current_scale_A_m2"]*self.net(p[:, 1:2]*c["time_reference_s"]/c["duration_s"])


class CoupledParticle(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.concentration = Concentration(config)
        scales = ParticleScales(config["radius_m"], config["diffusivity_m2_s"],
                                config["concentration_scale_mol_m3"], config["time_reference_s"])
        self.current = ParticleCurrent(Current(config), scales, config["time_quadrature_order"])
        r, w = gauss_legendre(config["radial_quadrature_order"], 0., 1.)
        self.register_buffer("radial_nodes", r)
        self.register_buffer("radial_weights", 3*r*r*w)

    def fields(self, p):
        c = self.config
        surface = torch.cat((torch.ones_like(p[:, :1]), p), dim=1)
        theta = reference(surface, c)
        ce = torch.full_like(theta, c["electrolyte_concentration_mol_m3"])
        temp = torch.full_like(theta, c["temperature_K"])
        phie = torch.full_like(theta, c["electrolyte_potential_V"])
        j = torch.full_like(theta, c["reference_current_A_m2"])
        j0 = exchange_current(ce, theta*c["concentration_scale_mol_m3"], temp,
                              c["electrode"], mode=c["exchange_current_mode"])
        return {"c_e": ce, "T": temp, "phi_e": phie,
                "phi_s": phie+ocp(theta, c["electrode"])+inverse_bv(j, j0, temp)}

    def losses(self, points):
        c = self.config
        interior = points["interior"].detach().clone().requires_grad_()
        boundary = points["boundary"].detach().clone().requires_grad_()
        inventory = points["inventory"]
        pde = diffusion_residual(self.concentration, interior, self.current.scales)
        interface = particle_interface(self.current, self.concentration, self.fields,
                                       boundary, c["electrode"], c["current_scale_A_m2"],
                                       mode=c["exchange_current_mode"])
        n, q = len(inventory), len(self.radial_nodes)
        radial = torch.stack((self.radial_nodes.expand(n, q), inventory[:, :1].expand(n, q),
                              inventory[:, 1:2].expand(n, q)), dim=-1).reshape(-1, 3)
        mean = (self.concentration(radial).reshape(n, q)*self.radial_weights).sum(1, keepdim=True)
        target = self.current.mean_target(inventory, inventory.new_tensor(c["initial_mean"]))
        return {"pde": pde.square().mean(),
                "surface_flux": interface["flux_residual"].square().mean(),
                "kinetics": interface["kinetics_residual"].square().mean(),
                "inventory": (mean-target).square().mean()}


def sample_points(c, smoke=False):
    generator = torch.Generator().manual_seed(c["seed"]+1)
    n, nb, ni = (16, 8, 4) if smoke else (c["collocation_points"], c["boundary_times"], c["inventory_times"])
    tau_end = c["duration_s"]/c["time_reference_s"]
    random = torch.rand(n, 2, generator=generator, dtype=torch.float64)
    interior = torch.stack((random[:, 0], torch.zeros(n, dtype=torch.float64),
                            (1-random[:, 1])*tau_end), dim=1)
    def times(count):
        t = torch.linspace(tau_end/count, tau_end, count, dtype=torch.float64)
        return torch.stack((torch.zeros_like(t), t), dim=1)
    return {"interior": interior, "boundary": times(nb), "inventory": times(ni)}
