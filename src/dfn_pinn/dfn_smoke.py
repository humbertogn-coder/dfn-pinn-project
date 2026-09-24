"""Small fully connected DFN residual assembly for wiring checks only."""

import torch
from torch import nn

from .constitutive import ocp
from .coupled_training import network
from .charge_conservation import ChargeScales, electrolyte_charge_terms, solid_charge_terms
from .electrolyte_mass import ElectrolyteScales, electrolyte_mass_terms
from .dfn_boundaries import (ElectrolyteBranch, electrolyte_interface, collector_conditions,
                             solid_separator_condition)
from .particle_current import ParticleCurrent
from .particle_interface import particle_interface
from .spherical_diffusion import ParticleScales, diffusion_residual, center_residual
from .projection import gauss_legendre


SETTINGS = {
    "scope": "Synthetic-IC DFN wiring smoke, not a validated Chen2020 discharge",
    "seed": 42, "duration_s": 1., "time_reference_s": 3600.,
    "lengths_m": [85.2e-6, 12e-6, 75.6e-6], "porosities": [.25, .47, .335],
    "solid_fractions": [.75, .665], "radii_m": [5.86e-6, 5.22e-6],
    "diffusivities_m2_s": [3.3e-14, 4e-15], "cmax_mol_m3": [33133., 63104.],
    "solid_conductivities_S_m": [215., .18], "initial_stoichiometries": [.8, .4],
    "initial_ce_mol_m3": 1000., "temperature_K": 298.15, "area_m2": .1027,
    "applied_current_A": 5., "kinetics_mode": "pybamm_26_8",
    "hidden_widths": [12, 12], "interior_points_per_region": 8,
    "boundary_times": 3, "inventory_quadrature": 8,
    "adam_steps": 3, "learning_rate": 1e-4, "dtype": "float64", "threads": 1,
}


class Field(nn.Module):
    def __init__(self, dimensions, base, kind, settings, amplitude=.01):
        super().__init__()
        self.net = network(dimensions, settings["hidden_widths"])
        self.register_buffer("base", torch.tensor(base, dtype=torch.float64))
        self.kind, self.amplitude = kind, amplitude
        self.time_factor = settings["time_reference_s"]/settings["duration_s"]
        self.particle = dimensions == 3

    def forward(self, p):
        s = p[:, -1:]*self.time_factor
        features = torch.cat((p[:, :1]**2, p[:, 1:2], s), dim=1) if self.particle else torch.cat((p[:, :1], s), dim=1)
        correction = self.amplitude*self.net(features)
        if self.kind == "solid_concentration":
            return torch.sigmoid(torch.logit(self.base)+s*correction)
        if self.kind == "electrolyte_concentration":
            return self.base*torch.exp(s*correction)
        return self.base+correction


class DFNSmoke(nn.Module):
    def __init__(self, settings=None):
        super().__init__()
        self.settings = dict(SETTINGS if settings is None else settings)
        c = self.settings
        length = sum(c["lengths_m"])
        self.mass_scales = ElectrolyteScales(length, c["time_reference_s"], c["initial_ce_mol_m3"])
        self.charge_scales = ChargeScales(length_m=length, concentration_mol_m3=c["initial_ce_mol_m3"],
                                          current_A_m2=c["applied_current_A"]/c["area_m2"])
        a, b = c["lengths_m"][0]/length, sum(c["lengths_m"][:2])/length
        self.bounds = ((0., a), (a, b), (b, 1.))
        initial_u = [float(ocp(torch.tensor(theta, dtype=torch.float64), e))
                     for theta, e in zip(c["initial_stoichiometries"], ("n", "p"))]
        phi_scale = self.charge_scales.potential_V
        self.ce = nn.ModuleList([Field(2, 1., "electrolyte_concentration", c) for _ in range(3)])
        self.phie = nn.ModuleList([Field(2, -initial_u[0]/phi_scale, "potential", c) for _ in range(3)])
        self.phis = nn.ModuleList([Field(2, base, "potential", c) for base in (0., (initial_u[1]-initial_u[0])/phi_scale)])
        self.cs = nn.ModuleList([Field(3, theta, "solid_concentration", c) for theta in c["initial_stoichiometries"]])
        self.active_area = [3*epsilon/r for epsilon, r in zip(c["solid_fractions"], c["radii_m"])]
        self.jref = [self.charge_scales.current_A_m2/(self.active_area[k]*c["lengths_m"][i]) for k, i in enumerate((0, 2))]
        self.reactions = nn.ModuleList([
            ParticleCurrent(Field(2, sign*self.jref[k], "current", c, amplitude=.01*self.jref[k]),
                            ParticleScales(c["radii_m"][k], c["diffusivities_m2_s"][k], c["cmax_mol_m3"][k], c["time_reference_s"]),
                            order=c["inventory_quadrature"])
            for k, sign in enumerate((1, -1))])
        r, w = gauss_legendre(c["inventory_quadrature"], 0., 1.)
        self.register_buffer("radial_nodes", r)
        self.register_buffer("radial_weights", 3*r*r*w)

    def branch(self, index):
        return ElectrolyteBranch(self.ce[index], self.phie[index], self.settings["porosities"][index])

    def sample(self):
        c = self.settings
        generator = torch.Generator().manual_seed(c["seed"]+1)
        n = c["interior_points_per_region"]
        end = c["duration_s"]/c["time_reference_s"]
        result = {}
        for index, (left, right) in enumerate(self.bounds):
            u = torch.rand(n, 3, generator=generator, dtype=torch.float64)
            result[f"region_{index}"] = torch.stack((left+(right-left)*u[:, 0], end*(1-u[:, 1])), dim=1)
            if index != 1:
                result[f"particle_{index}"] = torch.cat((u[:, 2:3], result[f"region_{index}"]), dim=1)
        result["times"] = torch.linspace(end/c["boundary_times"], end, c["boundary_times"], dtype=torch.float64)[:, None]
        return result

    def residuals(self, samples):
        c, m, s = self.settings, self.mass_scales, self.charge_scales
        residuals, diagnostics = {}, {}
        for index in range(3):
            p = samples[f"region_{index}"].detach().clone().requires_grad_()
            k = 0 if index == 0 else 1
            j = self.reactions[k].integral.current_model(p) if index != 1 else p[:, :1]*0
            aj = self.active_area[k]*j if index != 1 else j
            mass = electrolyte_mass_terms(self.ce[index], p, aj, m, porosity=c["porosities"][index])
            charge = electrolyte_charge_terms(self.ce[index], self.phie[index], p, aj, s, porosity=c["porosities"][index])
            residuals[f"salt_{index}"] = mass["normalized_residual"]
            residuals[f"charge_e_{index}"] = charge["normalized_residual"]
            if index == 1:
                continue
            solid = solid_charge_terms(self.phis[k], p, aj, s, conductivity_S_m=c["solid_conductivities_S_m"][k])
            residuals[f"charge_s_{index}"] = solid["normalized_residual"]
            diagnostics[f"total_current_{index}"] = (solid["current_A_m2"]+charge["current_A_m2"])/s.current_A_m2-1
            particle = samples[f"particle_{index}"].detach().clone().requires_grad_()
            residuals[f"particle_{index}"] = diffusion_residual(self.cs[k], particle, self.reactions[k].scales)
            def fields(z):
                return {"c_e": m.concentration_mol_m3*self.ce[index](z), "phi_e": s.potential_V*self.phie[index](z),
                        "phi_s": s.potential_V*self.phis[k](z), "T": torch.full_like(z[:, :1], c["temperature_K"])}
            interface = particle_interface(self.reactions[k], self.cs[k], fields, p, "n" if k == 0 else "p", self.jref[k], mode=c["kinetics_mode"])
            residuals[f"particle_flux_{index}"] = interface["flux_residual"]
            residuals[f"kinetics_{index}"] = interface["kinetics_residual"]
            center = torch.cat((torch.zeros_like(p[:, :1]), p), dim=1)
            residuals[f"center_{index}"] = center_residual(self.cs[k], center)
            q = len(self.radial_nodes)
            radial = torch.stack((self.radial_nodes.expand(len(p), q), p[:, :1].expand(len(p), q), p[:, 1:2].expand(len(p), q)), dim=-1).reshape(-1, 3)
            mean = (self.cs[k](radial).reshape(len(p), q)*self.radial_weights).sum(1, keepdim=True)
            residuals[f"inventory_{index}"] = mean-self.reactions[k].mean_target(p, p.new_tensor(c["initial_stoichiometries"][k]))
        times = samples["times"]
        for k, position in enumerate((self.bounds[0][1], self.bounds[1][1])):
            jumps = electrolyte_interface(self.branch(k), self.branch(k+1), times, position, m, s)
            residuals.update({f"interface_{k}_{name}": value for name, value in jumps.items()})
            residuals[f"solid_insulating_{k}"] = solid_separator_condition(self.phis[k], times, position, s, conductivity_S_m=c["solid_conductivities_S_m"][k])
        for side, index, k in (("negative", 0, 0), ("positive", 2, 1)):
            bc = collector_conditions(side, self.branch(index), self.phis[k], times, m, s,
                                      conductivity_S_m=c["solid_conductivities_S_m"][k], applied_current=times.new_tensor(s.current_A_m2))
            residuals.update({f"collector_{side}_{name}": value for name, value in bc["conditions"].items()})
            diagnostics[f"collector_{side}_total_current"] = bc["total_current_diagnostic"]
        return residuals, diagnostics

    def snapshot(self, samples):
        values = {}
        for i in range(3):
            p = samples[f"region_{i}"]
            values[f"ce_{i}"] = self.ce[i](p)
            values[f"phie_{i}"] = self.phie[i](p)
        for k, i in enumerate((0, 2)):
            values[f"cs_{i}"] = self.cs[k](samples[f"particle_{i}"])
            values[f"phis_{i}"] = self.phis[k](samples[f"region_{i}"])
            values[f"j_{i}"] = self.reactions[k].integral.current_model(samples[f"region_{i}"])
        return values
