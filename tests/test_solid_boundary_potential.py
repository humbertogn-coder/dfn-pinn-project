"""Boundary identities and deliberate local-charge failures, without training."""

import pytest
import torch
from torch import nn

from dfn_pinn.charge_conservation import ChargeScales, solid_charge_terms
from dfn_pinn.dfn_smoke import SETTINGS
from dfn_pinn.solid_boundary_potential import SolidBoundaryPotential
from dfn_pinn.spherical_diffusion import _gradient


class Raw(nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(slope, dtype=torch.float64))
        self.offset = nn.Parameter(torch.tensor(.3, dtype=torch.float64))

    def forward(self, p):
        return self.offset + self.weight*p[:, :1].square()*(1+3600*p[:, 1:2])


def setup(electrode, slope=.4):
    scales = ChargeScales(length_m=sum(SETTINGS["lengths_m"]))
    k = 0 if electrode == "n" else 1
    width = SETTINGS["lengths_m"][(0, 2)[k]]/scales.length_m
    bounds = (0., width) if k == 0 else (1-width, 1.)
    sigma = SETTINGS["solid_conductivities_S_m"][k]
    field = SolidBoundaryPotential(Raw(slope), scales, sigma, bounds, electrode,
                                   base=0. if k == 0 else 150.)
    return field, scales, sigma, bounds


@pytest.mark.parametrize("electrode", ["n", "p"])
def test_boundary_currents_survive_parameter_changes(electrode):
    field, scales, sigma, (left, right) = setup(electrode)
    times = torch.tensor([0., 1e-6/3600, .5/3600, 1/3600], dtype=torch.float64)
    for weight in (-2., .4, 3.):
        with torch.no_grad():
            field.raw.weight.fill_(weight)
        for x, target in ((left, 1. if electrode == "n" else 0.),
                          (right, 0. if electrode == "n" else 1.)):
            p = torch.stack((torch.full_like(times, x), times), dim=1).requires_grad_()
            result = solid_charge_terms(field, p, p[:, :1]*0, scales, conductivity_S_m=sigma)
            normalized = result["current_A_m2"]/scales.current_A_m2
            torch.testing.assert_close(normalized, torch.full_like(normalized, target), atol=1e-12, rtol=1e-12)
            gradient = torch.autograd.grad(normalized.sum(), field.raw.weight)[0]
            assert abs(float(gradient)) < 1e-12
        if electrode == "n":
            p = torch.stack((torch.zeros_like(times), times), dim=1)
            torch.testing.assert_close(field(p), torch.zeros_like(times[:, None]), atol=1e-14, rtol=0)


@pytest.mark.parametrize("electrode", ["n", "p"])
def test_uniform_source_and_nonuniform_correction(electrode):
    field, scales, sigma, (left, right) = setup(electrode, 0.)
    x = torch.linspace(left, right, 19, dtype=torch.float64)
    p = torch.stack((x, torch.full_like(x, .2/3600)), dim=1).requires_grad_()
    source = p.new_tensor((1 if electrode == "n" else -1)*scales.current_A_m2/((right-left)*scales.length_m))
    result = solid_charge_terms(field, p, source, scales, conductivity_S_m=sigma)
    assert result["normalized_residual"].abs().max() < 1e-12
    with torch.no_grad():
        field.raw.weight.fill_(.4)
    result = solid_charge_terms(field, p, source, scales, conductivity_S_m=sigma)
    assert result["normalized_residual"].abs().max() > .01
    gradient = torch.autograd.grad(result["normalized_residual"].square().mean(), field.raw.weight)[0]
    assert torch.isfinite(gradient) and gradient.abs() > 1e-8
    time_derivative = _gradient(field(p), p)[:, 1:2]
    assert torch.isfinite(_gradient(time_derivative, p)).all()


def test_positive_offset_free_negative_gauge_fixed():
    for electrode in ("n", "p"):
        field, _, _, (left, right) = setup(electrode)
        p = torch.tensor([[(left+right)/2, .1/3600]], dtype=torch.float64)
        derivative = torch.autograd.grad(field(p).sum(), field.raw.offset)[0]
        expected = 0. if electrode == "n" else field.amplitude
        torch.testing.assert_close(derivative, derivative.new_tensor(expected))


@pytest.mark.parametrize("electrode,bounds,base", [("bad", (0., .4), 0.), ("n", (.1,.4),0.),
    ("p", (.5,.9), 0.), ("n", (0.,.4),1.), ("p", (1.,1.),0.)])
def test_invalid_geometry(electrode, bounds, base):
    with pytest.raises(ValueError):
        SolidBoundaryPotential(Raw(.4), ChargeScales(), 1., bounds, electrode, base)
