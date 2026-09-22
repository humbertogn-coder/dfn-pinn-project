import pytest
import torch
from torch import nn

from dfn_pinn.constitutive import MAX_CONCENTRATION, exchange_current, ocp
from dfn_pinn.kinetics import inverse_bv
from dfn_pinn.particle_current import ParticleCurrent
from dfn_pinn.spherical_diffusion import ParticleScales


class Current(nn.Module):
    def __init__(self, sign):
        super().__init__()
        self.amplitude = nn.Parameter(torch.tensor(sign, dtype=torch.float64))

    def forward(self, p):
        return self.amplitude*(1+p[:, :1])*(1+p[:, 1:2])


def setup(electrode, sign=1.):
    scale = ((5.86e-6, 3.3e-14, 33133.) if electrode == "n"
             else (5.22e-6, 4e-15, 63104.))
    model = ParticleCurrent(Current(sign), ParticleScales(*scale))
    p = torch.tensor([[.2, .001], [.7, .002]], dtype=torch.float64, requires_grad=True)
    fields = [torch.full((2, 1), value, dtype=torch.float64, requires_grad=True)
              for value in (.5, 1000., .02, 298.15)]
    return model, p, fields


@pytest.mark.parametrize("electrode", ["n", "p"])
@pytest.mark.parametrize("sign", [-1., 0., 1.])
@pytest.mark.parametrize("mode", ["raw", "pybamm_26_8"])
def test_manufactured_kinetics_consistency(electrode, sign, mode):
    model, p, (theta, ce, phie, temp) = setup(electrode, sign)
    j = model.integral.current_model(p)
    j0 = exchange_current(ce, theta*MAX_CONCENTRATION[electrode], temp, electrode, mode=mode)
    phis = phie+ocp(theta, electrode)+inverse_bv(j, j0, temp)
    residual = model.kinetics_residual(p, theta, ce, phis, phie, temp, electrode, 2., mode=mode)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)
    shifted = model.kinetics_residual(p, theta, ce, phis+.01, phie, temp, electrode, 2., mode=mode)
    assert bool((shifted < 0).all())


@pytest.mark.parametrize("electrode", ["n", "p"])
def test_independent_field_gradients(electrode):
    model, p, (theta, ce, phie, temp) = setup(electrode)
    phis = (phie+ocp(theta, electrode)+.03).detach().requires_grad_()
    def residual(th, c, ps, pe, t):
        return model.kinetics_residual(p, th, c, ps, pe, t, electrode, 1., mode="pybamm_26_8")
    assert torch.autograd.gradcheck(residual, (theta, ce, phis, phie, temp))
    assert torch.autograd.gradgradcheck(residual, (theta, ce, phis, phie, temp))
    r = residual(theta, ce, phis, phie, temp)
    fields = (theta, ce, phis, phie, temp, model.integral.current_model.amplitude)
    gradients = torch.autograd.grad(r.sum(), fields)
    assert all(torch.isfinite(g).all() and (g.abs() > 0).all() for g in gradients)
    torch.testing.assert_close(gradients[2], -gradients[3])
    mean = model.mean_target(p, torch.tensor(.8, dtype=torch.float64))
    assert torch.autograd.grad(mean.sum(), model.integral.current_model.amplitude)[0] < 0


def test_invalid_fields_rejected():
    model, p, (theta, ce, phie, temp) = setup("n")
    for bad in (theta.flatten(), theta*0, theta*2):
        with pytest.raises(ValueError):
            model.kinetics_residual(p, bad, ce, phie, phie, temp, "n", 1., mode="raw")
    with pytest.raises(ValueError):
        model.kinetics_residual(p, theta, ce, phie, phie, temp, "n", 0., mode="raw")
