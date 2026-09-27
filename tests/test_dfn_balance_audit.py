"""Necessary independent-integration checks and deliberately inconsistent fields."""

import numpy as np
import pytest
import torch

from dfn_pinn.constitutive import FARADAY_CONSTANT
from dfn_pinn.dfn_smoke import DFNSmoke
from dfn_pinn.dfn_balance_audit import inventory_curves, local_inventory_error, reaction_error, evaluate


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, p):
        return p[:, :1]*0+self.value


class UniformParticle(torch.nn.Module):
    def __init__(self, initial, rate):
        super().__init__()
        self.initial, self.rate = initial, rate

    def forward(self, p):
        return self.initial-self.rate*p[:, -1:]+p[:, :1]*0


def balanced_fields():
    m = DFNSmoke()
    for i in range(3):
        m.ce[i] = Constant(1.)
    c = m.settings
    for k, sign in enumerate((1, -1)):
        j = sign*m.jref[k]
        m.reactions[k].integral.current_model = Constant(j)
        rate = 3*c['time_reference_s']*j/(FARADAY_CONSTANT*c['radii_m'][k]*c['cmax_mol_m3'][k])
        m.cs[k] = UniformParticle(c['initial_stoichiometries'][k], rate)
    return m


def test_physical_geometry_signs_and_inventory():
    m = balanced_fields()
    t = np.array([0., .2, 1.])
    values = inventory_curves(m, t, 8)
    c = m.settings
    expected_e = c['area_m2']*sum(l*eps for l, eps in zip(c['lengths_m'], c['porosities']))*1000
    expected_s = c['area_m2']*sum(c['lengths_m'][i]*c['solid_fractions'][k]*c['cmax_mol_m3'][k]*c['initial_stoichiometries'][k] for k, i in enumerate((0, 2)))
    np.testing.assert_allclose(values['electrolyte_mol'], expected_e, rtol=1e-14)
    np.testing.assert_allclose(values['total_mol'], expected_e+expected_s, rtol=1e-14)
    assert reaction_error(values['electrode_current_A'], 5.) < 1e-14
    assert max(values['local_inventory_errors'].values()) < 1e-14


def test_wrong_positive_current_and_inventory_are_detected():
    m = balanced_fields()
    m.reactions[1].integral.current_model = Constant(m.jref[1])
    values = inventory_curves(m, np.array([0., 1.]), 8)
    assert reaction_error(values['electrode_current_A'], 5.) == pytest.approx(2.)
    assert values['local_inventory_errors']['2'] > 1e-6


def test_exact_inventory_does_not_hide_wrong_surface_flux():
    m = balanced_fields()
    xt = np.array([[.1, 1/3600]])
    assert np.abs(local_inventory_error(m, 0, xt, 8)).max() < 1e-14
    surface = torch.tensor([[1., .1, 1/3600]], dtype=torch.float64, requires_grad=True)
    flux = m.reactions[0].flux_residual(m.cs[0], surface, m.jref[0])
    assert float(flux.detach().abs().max()) == pytest.approx(1.)


def test_independent_radial_orders_expose_underresolution():
    m = balanced_fields()
    original = m.cs[0]
    class Perturbed(torch.nn.Module):
        def forward(self, p):
            return original(p)+.01*p[:, :1]**6
    m.cs[0] = Perturbed()
    xt = np.array([[.1, 1/3600]])
    coarse = local_inventory_error(m, 0, xt, 2)
    fine = local_inventory_error(m, 0, xt, 8)
    assert abs(coarse[0]-fine[0]) > .05*1e-6
    assert fine[0] == pytest.approx(.01/3)


def test_nonfinite_fields_cannot_pass():
    with pytest.raises(ValueError, match='finite'):
        evaluate(Constant(float('nan')), np.array([[.1, .01]]))
