"""Three-region integral identities; not a complete DFN solution."""

import pytest
import torch

from dfn_pinn.constitutive import electrolyte_conductivity
from dfn_pinn.charge_conservation import ChargeScales, electrolyte_charge_terms, solid_charge_terms
from dfn_pinn.electrolyte_mass import ElectrolyteScales, electrolyte_mass_terms
from dfn_pinn.dfn_boundaries import (ElectrolyteBranch, electrolyte_interface,
    electrolyte_trace, boundary_points, collector_conditions, solid_separator_condition)
from dfn_pinn.projection import gauss_legendre


M, S = ElectrolyteScales(), ChargeScales()
A, B = 85.2/172.8, (85.2+12)/172.8
BOUNDS = ((0., A), (A, B), (B, 1.))
POROSITIES = (.25, .47, .335)


def manufactured(current):
    bulk = electrolyte_conductivity(torch.tensor(1000., dtype=torch.float64))
    kn, ks, kp = [epsilon**1.5*bulk for epsilon in POROSITIES]
    factor = current*S.length_m/S.potential_V
    negative = lambda p: -factor*p[:, :1]**2/(2*A*kn)
    separator = lambda p: -factor*(A/(2*kn)+(p[:, :1]-A)/ks)
    positive = lambda p: -factor*(A/(2*kn)+(B-A)/ks+((p[:, :1]-B)-(p[:, :1]-B)**2/(2*(1-B)))/kp)
    concentration = lambda p: 1+p[:, :1]*0
    branches = [ElectrolyteBranch(concentration, phi, epsilon)
                for phi, epsilon in zip((negative, separator, positive), POROSITIES)]
    solid_n = lambda p: -factor/215*(p[:, :1]-p[:, :1]**2/(2*A))
    solid_p = lambda p: 4/S.potential_V-factor/.18*(p[:, :1]-B)**2/(2*(1-B))
    return branches, (solid_n, solid_p)


def region_points(index, order=16):
    left, right = BOUNDS[index]
    x, weights = gauss_legendre(order, left*M.length_m, right*M.length_m)
    points = torch.stack((x/M.length_m, x*0+.01), dim=1).requires_grad_()
    return points, weights


@pytest.mark.parametrize("sign", [-1., 0., 1.])
def test_three_region_charge_and_closed_salt_integral(sign):
    current = torch.tensor(sign*20., dtype=torch.float64, requires_grad=True)
    branches, solids = manufactured(current)
    sources = [current/(A*M.length_m), current*0, -current/((1-B)*M.length_m)]
    integrals, salt_residuals, local_salt = [], [], []
    for index, branch in enumerate(branches):
        p, weights = region_points(index)
        source = sources[index]
        charge = electrolyte_charge_terms(branch.concentration, branch.potential, p, source, S, porosity=branch.porosity)
        torch.testing.assert_close(charge["normalized_residual"], torch.zeros_like(p[:, :1]), atol=1e-12, rtol=0)
        integrals.append((source*weights).sum())
        mass = electrolyte_mass_terms(branch.concentration, p, source, M, porosity=branch.porosity)
        salt_residuals.append((mass["balance_mol_m3_s"].flatten()*weights).sum())
        local_salt.append(mass["balance_mol_m3_s"])
        if index != 1:
            solid = solid_charge_terms(solids[0 if index == 0 else 1], p, source, S,
                                       conductivity_S_m=215. if index == 0 else .18)
            torch.testing.assert_close(solid["normalized_residual"], torch.zeros_like(p[:, :1]), atol=1e-12, rtol=0)
            torch.testing.assert_close(solid["current_A_m2"]+charge["current_A_m2"], current.expand(len(p), 1), atol=1e-10, rtol=0)
    torch.testing.assert_close(integrals[0], current)
    torch.testing.assert_close(integrals[2], -current)
    torch.testing.assert_close(sum(integrals), current*0, atol=1e-12, rtol=0)
    torch.testing.assert_close(sum(salt_residuals), current*0, atol=1e-15, rtol=0)
    # Global cancellation does not solve local salt balance under nonzero reaction.
    if sign:
        assert local_salt[0].abs().max() > 0
    t = torch.tensor([[.01]], dtype=torch.float64)
    for i, position in enumerate((A, B)):
        for value in electrolyte_interface(branches[i], branches[i+1], t, position, M, S).values():
            torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-12, rtol=0)
        condition = solid_separator_condition(solids[i], t, position, S, conductivity_S_m=215. if i == 0 else .18)
        torch.testing.assert_close(condition, torch.zeros_like(condition), atol=1e-12, rtol=0)
    for side, idx, solid, sigma in (("negative", 0, solids[0], 215.), ("positive", 2, solids[1], .18)):
        result = collector_conditions(side, branches[idx], solid, t, M, S, conductivity_S_m=sigma, applied_current=current)
        for value in result["conditions"].values():
            torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-12, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(integrals[0], current)[0], torch.ones_like(current))


def test_wrong_positive_source_cannot_hide_in_global_balance():
    current = torch.tensor(20., dtype=torch.float64)
    branches, _ = manufactured(current)
    p, w = region_points(2)
    wrong_source = current/((1-B)*M.length_m)
    charge = electrolyte_charge_terms(branches[2].concentration, branches[2].potential, p, wrong_source, S, porosity=POROSITIES[2])
    torch.testing.assert_close((charge["balance_A_m3"].flatten()*w).sum(), -2*current)
    assert charge["normalized_residual"].abs().max() > 0


def test_salt_integral_keeps_interface_flux_jumps():
    branches = [ElectrolyteBranch(lambda p, g=g: 1+g*p[:, :1], lambda p: p[:, :1]*0, e)
                for g, e in zip((.1, .2, -.1), POROSITIES)]
    t = torch.tensor([[.01]], dtype=torch.float64)
    integral = t.new_tensor(0.)
    traces = []
    flux_scale = M.concentration_mol_m3*M.length_m/M.time_s
    for index, branch in enumerate(branches):
        p, w = region_points(index, 32)
        terms = electrolyte_mass_terms(branch.concentration, p, p.new_tensor(0.), M, porosity=branch.porosity)
        integral = integral+(terms["balance_mol_m3_s"].flatten()*w).sum()
        traces.append([electrolyte_trace(branch, boundary_points(t, x), M, S)["diffusive_flux"].squeeze()*flux_scale
                       for x in BOUNDS[index]])
    external = traces[2][1]-traces[0][0]
    internal = traces[0][1]-traces[1][0]+traces[1][1]-traces[2][0]
    torch.testing.assert_close(integral, external+internal, atol=1e-14, rtol=1e-10)
    assert float((integral-external).detach().abs()) > 1e-8
