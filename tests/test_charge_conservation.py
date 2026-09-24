import pytest
import torch

from dfn_pinn.charge_conservation import ChargeScales, solid_charge_terms, electrolyte_charge_terms
from dfn_pinn.constitutive import FARADAY_CONSTANT, GAS_CONSTANT


def points():
    return torch.tensor([[.1, .01], [.4, .1], [.8, .2]], dtype=torch.float64, requires_grad=True)


@pytest.mark.parametrize("sigma", [215., .18])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_solid_manufactured_balance(sigma, sign):
    p, s = points(), ChargeScales()
    aj = p.new_tensor(sign*2*sigma*s.potential_V/s.length_m**2)
    result = solid_charge_terms(lambda z: sign*z[:, :1]**2, p, aj, s, conductivity_S_m=sigma)
    torch.testing.assert_close(result["current_A_m2"], -sign*2*sigma*s.potential_V/s.length_m*p[:, :1])
    torch.testing.assert_close(result["normalized_residual"], torch.zeros_like(p[:, :1]), atol=1e-12, rtol=0)


@pytest.mark.parametrize("epsilon", [.25, .47, .335])
def test_electrolyte_nonlinear_product_rule(epsilon):
    p, s = points(), ChargeScales()
    x = p[:, :1]
    c = 1+.2*x
    k = .1297*c**3-2.51*c**1.5+3.329*c
    dk = .3891*c**2-3.765*c.sqrt()+3.329
    b = 2*GAS_CONSTANT*298.15/FARADAY_CONSTANT*(1-.2594)
    drive = b*.2/c-s.potential_V*.6*x
    aj = epsilon**1.5/s.length_m**2*(dk*.2*drive+k*(-b*.04/c**2-s.potential_V*.6))
    result = electrolyte_charge_terms(lambda z: 1+.2*z[:, :1], lambda z: .3*z[:, :1]**2,
                                      p, aj, s, porosity=epsilon)
    torch.testing.assert_close(result["current_A_m2"], epsilon**1.5*k*drive/s.length_m)
    torch.testing.assert_close(result["normalized_residual"], torch.zeros_like(x), atol=1e-12, rtol=0)


def test_gauge_and_zero_current_diffusion_potential():
    p, s = points(), ChargeScales()
    zero = p.new_tensor(0.)
    b = 2*GAS_CONSTANT*298.15/FARADAY_CONSTANT*(1-.2594)
    concentration = lambda z: 1+.2*z[:, :1]
    potential = lambda z: b/s.potential_V*torch.log(concentration(z))
    result = electrolyte_charge_terms(concentration, potential, p, zero, s, porosity=.47)
    shifted = electrolyte_charge_terms(concentration, lambda z: potential(z)+7., p, zero, s, porosity=.47)
    torch.testing.assert_close(result["current_A_m2"], torch.zeros_like(p[:, :1]), atol=1e-12, rtol=0)
    torch.testing.assert_close(shifted["current_A_m2"], result["current_A_m2"], atol=1e-12, rtol=0)
    a = solid_charge_terms(lambda z: z[:, :1]**2, p, zero, s, conductivity_S_m=215.)
    b = solid_charge_terms(lambda z: z[:, :1]**2+7., p, zero, s, conductivity_S_m=215.)
    torch.testing.assert_close(a["current_A_m2"], b["current_A_m2"])


def test_opposite_source_signs_and_gradients():
    p, s = points(), ChargeScales()
    source = p.new_tensor(1e5, requires_grad=True)
    potential = lambda z: z[:, :1]*0
    solid = solid_charge_terms(potential, p, source, s, conductivity_S_m=215.)
    liquid = electrolyte_charge_terms(lambda z: 1+z[:, :1]*0, potential, p, source, s, porosity=.25)
    torch.testing.assert_close(solid["balance_A_m3"], -liquid["balance_A_m3"])
    assert torch.autograd.grad(solid["balance_A_m3"].sum(), source)[0] == 3
    assert torch.autograd.grad(liquid["balance_A_m3"].sum(), source)[0] == -3


def test_parameter_derivatives():
    p, s = points(), ChargeScales()
    a = p.new_tensor(.2, requires_grad=True)
    def residual(amplitude):
        return electrolyte_charge_terms(lambda z: 1+amplitude*z[:, :1], lambda z: amplitude*z[:, :1]**2,
                                        p, p.new_tensor(0.), s, porosity=.25)["normalized_residual"]
    assert torch.autograd.gradcheck(residual, (a,))
    assert torch.autograd.gradgradcheck(residual, (a,))


def test_invalid_input_rejected():
    p, s = points(), ChargeScales()
    with pytest.raises(ValueError):
        solid_charge_terms(lambda z: z[:, :1], p, p.new_zeros(3), s, conductivity_S_m=215.)
    with pytest.raises(ValueError):
        electrolyte_charge_terms(lambda z: z[:, :1]*0, lambda z: z[:, :1], p, p.new_tensor(0.), s, porosity=.25)
    with pytest.raises(ValueError):
        ChargeScales(current_A_m2=0)
