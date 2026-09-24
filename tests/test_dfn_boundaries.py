import pytest
import torch

from dfn_pinn.charge_conservation import ChargeScales
from dfn_pinn.electrolyte_mass import ElectrolyteScales
from dfn_pinn.dfn_boundaries import (ElectrolyteBranch, electrolyte_interface,
    collector_conditions, solid_separator_condition, boundary_points)
from dfn_pinn.constitutive import FARADAY_CONSTANT, GAS_CONSTANT, electrolyte_conductivity


def times():
    return torch.tensor([[.01], [.1]], dtype=torch.float64)


@pytest.mark.parametrize("position", [85.2/172.8, (85.2+12)/172.8])
def test_continuous_flux_with_discontinuous_gradient(position):
    m, s = ElectrolyteScales(), ChargeScales()
    b = 2*GAS_CONSTANT*298.15/FARADAY_CONSTANT*(1-.2594)
    k = float(electrolyte_conductivity(torch.tensor(1000., dtype=torch.float64)))
    def branch(epsilon):
        gradient = .01/epsilon**1.5
        slope = b/s.potential_V*gradient-s.length_m*2/(epsilon**1.5*k*s.potential_V)
        return ElectrolyteBranch(lambda p: 1+gradient*(p[:, :1]-position),
                                  lambda p: .2+slope*(p[:, :1]-position), epsilon)
    jumps = electrolyte_interface(branch(.25), branch(.47), times(), position, m, s)
    for value in jumps.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-12, rtol=0)


def test_jump_and_parameter_time_gradients():
    m, s = ElectrolyteScales(), ChargeScales()
    t = times().requires_grad_()
    a = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    left = ElectrolyteBranch(lambda p: 1+p[:, :1]*0, lambda p: a*p[:, 1:2], .25)
    right = ElectrolyteBranch(lambda p: 1+p[:, :1]*0, lambda p: p[:, :1]*0, .47)
    jump = electrolyte_interface(left, right, t, .5, m, s)["potential"]
    torch.testing.assert_close(jump, a*t)
    torch.testing.assert_close(torch.autograd.grad(jump.sum(), a, retain_graph=True)[0], t.sum())
    torch.testing.assert_close(torch.autograd.grad(jump.sum(), t)[0], a.expand_as(t))


@pytest.mark.parametrize("sign", [-1., 0., 1.])
def test_collector_sign_and_single_gauge(sign):
    m, s = ElectrolyteScales(), ChargeScales()
    branch = ElectrolyteBranch(lambda p: 1+p[:, :1]*0, lambda p: 3+p[:, :1]*0, .25)
    current = torch.tensor(sign*20., dtype=torch.float64)
    solid = lambda p: -current*s.length_m/(215*s.potential_V)*p[:, :1]
    for side in ("negative", "positive"):
        result = collector_conditions(side, branch, solid, times(), m, s,
                                       conductivity_S_m=215., applied_current=current)
        assert len(result["conditions"]) == 3
        assert "solid_potential_gauge" in result["conditions"] if side == "negative" else "solid_potential_gauge" not in result["conditions"]
        for value in result["conditions"].values():
            torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-12, rtol=0)
        torch.testing.assert_close(result["total_current_diagnostic"], torch.zeros_like(times()), atol=1e-12, rtol=0)


def test_wrong_negative_current_remains_diagnostic_only():
    m, s = ElectrolyteScales(), ChargeScales()
    zero = lambda p: p[:, :1]*0
    branch = ElectrolyteBranch(lambda p: 1+zero(p), zero, .25)
    result = collector_conditions("negative", branch, zero, times(), m, s,
                                   conductivity_S_m=215., applied_current=torch.tensor(10., dtype=torch.float64))
    assert "solid_current" not in result["conditions"]
    assert (result["total_current_diagnostic"] != 0).all()
    torch.testing.assert_close(solid_separator_condition(zero, times(), .5, s, conductivity_S_m=215.), torch.zeros_like(times()))


def test_invalid_times_and_scale_mismatch():
    with pytest.raises(ValueError):
        boundary_points(torch.ones(2, dtype=torch.float64), .5)
    branch = ElectrolyteBranch(lambda p: 1+p[:, :1]*0, lambda p: p[:, :1]*0, .25)
    with pytest.raises(ValueError, match="share"):
        electrolyte_interface(branch, branch, times(), .5, ElectrolyteScales(), ChargeScales(length_m=1.))
