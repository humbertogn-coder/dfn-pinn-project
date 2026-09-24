import pytest
import torch

from dfn_pinn.constitutive import FARADAY_CONSTANT
from dfn_pinn.electrolyte_mass import ElectrolyteScales, electrolyte_mass_terms


def points():
    return torch.tensor([[.1, .01], [.4, .1], [.8, .2]], dtype=torch.float64, requires_grad=True)


@pytest.mark.parametrize("epsilon", [.25, .47, .335])
def test_nonlinear_diffusion_product_rule_and_si_chain(epsilon):
    p = points()
    scales = ElectrolyteScales()
    c = 1+.2*p[:, :1]**2+.1*p[:, 1:2]
    dx = .4*p[:, :1]
    d = 8.794e-11*c**2-3.972e-10*c+4.862e-10
    dp = 2*8.794e-11*c-3.972e-10
    diffusion = epsilon**1.5*1000/scales.length_m**2*(d*.4+dp*dx**2)
    accumulation = epsilon*1000/scales.time_s*.1
    source = (accumulation-diffusion)*FARADAY_CONSTANT/(1-.2594)
    result = electrolyte_mass_terms(lambda z: 1+.2*z[:, :1]**2+.1*z[:, 1:2], p, source, scales, porosity=epsilon)
    torch.testing.assert_close(result["normalized_residual"], torch.zeros_like(c), atol=1e-13, rtol=0)
    torch.testing.assert_close(result["diffusive_flux_mol_m2_s"], -epsilon**1.5*d*1000/scales.length_m*dx)
    torch.testing.assert_close(result["flux_divergence_mol_m3_s"], -diffusion)


@pytest.mark.parametrize("sign", [-1., 0., 1.])
def test_uniform_separator_and_reaction_sign(sign):
    p = points()
    aj = torch.tensor(sign*1e5, dtype=torch.float64, requires_grad=True)
    result = electrolyte_mass_terms(lambda z: 1+z[:, :1]*0, p, aj, ElectrolyteScales(), porosity=.47)
    torch.testing.assert_close(result["balance_mol_m3_s"], torch.full((3, 1), -(1-.2594)*sign*1e5/FARADAY_CONSTANT, dtype=torch.float64))
    gradient = torch.autograd.grad(result["balance_mol_m3_s"].sum(), aj)[0]
    assert gradient < 0


def test_parameter_gradients():
    p = points()
    zero = torch.tensor(0., dtype=torch.float64)
    def residual(amplitude):
        return electrolyte_mass_terms(lambda z: 1+amplitude*z[:, :1]**2, p, zero,
                                      ElectrolyteScales(), porosity=.25)["normalized_residual"]
    a = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(residual, (a,))
    assert torch.autograd.gradgradcheck(residual, (a,))


def test_invalid_inputs():
    p = points()
    zero = torch.tensor(0., dtype=torch.float64)
    with pytest.raises(ValueError):
        electrolyte_mass_terms(lambda z: z[:, :1]*0, p, zero, ElectrolyteScales(), porosity=.25)
    with pytest.raises(ValueError):
        electrolyte_mass_terms(lambda z: 1+z[:, :1]*0, p, torch.zeros(3, dtype=torch.float64), ElectrolyteScales(), porosity=.25)
    with pytest.raises(ValueError):
        ElectrolyteScales(length_m=0)
