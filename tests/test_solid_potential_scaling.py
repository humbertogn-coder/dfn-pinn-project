"""Analytic and paired-network checks only; no optimizer or DFN solve."""

import pytest
import torch

from dfn_pinn.charge_conservation import ChargeScales, solid_charge_terms
from dfn_pinn.dfn_smoke import Field, SETTINGS
from dfn_pinn.solid_potential_scaling import ohmic_amplitude, scaled_solid_potential


SCALES = ChargeScales(length_m=sum(SETTINGS["lengths_m"]))


@pytest.mark.parametrize("electrode", [0, 1])
def test_manufactured_charge_and_current_sensitivity(electrode):
    sigma = SETTINGS["solid_conductivities_S_m"][electrode]
    amplitude = ohmic_amplitude(SCALES, sigma)
    width = SETTINGS["lengths_m"][(0, 2)[electrode]] / SCALES.length_m
    left = 0. if electrode == 0 else 1. - width
    x = torch.linspace(left, left + width, 17, dtype=torch.float64)
    points = torch.stack((x, torch.full_like(x, 1e-4)), dim=1).requires_grad_()
    strength = torch.tensor(1., dtype=torch.float64, requires_grad=True)

    def potential(p):
        xi = (p[:, :1] - left) / width
        shape = width * (-xi + xi.square()/2 if electrode == 0 else -xi.square()/2)
        return amplitude * strength * shape + p[:, 1:]*0

    source = points.new_tensor((1 if electrode == 0 else -1)*SCALES.current_A_m2/(width*SCALES.length_m))
    result = solid_charge_terms(potential, points, source, SCALES, conductivity_S_m=sigma)
    xi = (x-left)/width
    expected = 1-xi if electrode == 0 else xi
    current = result["current_A_m2"][:, 0]/SCALES.current_A_m2
    torch.testing.assert_close(current, expected, atol=1e-12, rtol=1e-12)
    assert result["normalized_residual"].abs().max() < 1e-12
    collector = current[0 if electrode == 0 else -1]
    sensitivity = torch.autograd.grad(collector, strength)[0]
    torch.testing.assert_close(sensitivity, torch.ones_like(sensitivity))


@pytest.mark.parametrize("sigma", SETTINGS["solid_conductivities_S_m"])
def test_same_network_only_amplitude_changes(sigma):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        old = Field(2, 0., "potential", SETTINGS)
        new = scaled_solid_potential(0., SETTINGS, SCALES, sigma)
    new.load_state_dict(old.state_dict(), strict=True)
    points = torch.tensor([[.1, 1e-6], [.8, 1e-4]], dtype=torch.float64, requires_grad=True)
    ratio = new.amplitude/old.amplitude
    torch.testing.assert_close(new(points), ratio*old(points))
    currents = []
    gradients = []
    for model in (old, new):
        current = solid_charge_terms(model, points, points[:, :1]*0, SCALES,
                                     conductivity_S_m=sigma)["current_A_m2"]/SCALES.current_A_m2
        currents.append(current)
        gradients.append(torch.autograd.grad(current.sum(), tuple(model.parameters()),
                                             allow_unused=True, retain_graph=True))
    torch.testing.assert_close(currents[1], ratio*currents[0])
    for before, after in zip(*gradients):
        if before is None:
            assert after is None
        else:
            assert torch.isfinite(after).all()
            torch.testing.assert_close(after, ratio*before)
    time_gradient = torch.autograd.grad(new(points).sum(), points, create_graph=True)[0][:, 1]
    assert torch.isfinite(time_gradient).all()
    assert torch.isfinite(torch.autograd.grad(time_gradient.sum(), points)[0]).all()


@pytest.mark.parametrize("sigma", [0., -1., float("nan"), float("inf")])
def test_invalid_conductivity(sigma):
    with pytest.raises(ValueError, match="Conductivity"):
        ohmic_amplitude(SCALES, sigma)
