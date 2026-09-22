import pytest
import torch

from dfn_pinn.constitutive import FARADAY_CONSTANT, MAX_CONCENTRATION, ocp, exchange_current
from dfn_pinn.kinetics import inverse_bv
from dfn_pinn.particle_balance import particle_mean_rate
from dfn_pinn.particle_current import ParticleCurrent
from dfn_pinn.particle_interface import particle_interface
from dfn_pinn.spherical_diffusion import ParticleScales, diffusion_residual, center_residual
from dfn_pinn.projection import gauss_legendre


def case(electrode, sign, ratio=1.):
    maximum = MAX_CONCENTRATION[electrode]
    r, d = (5.86e-6, 3.3e-14) if electrode == "n" else (5.22e-6, 4e-15)
    scales = ParticleScales(r, d, maximum*ratio)
    current = lambda p: sign*.1*(1+p[:, :1])
    coupling = ParticleCurrent(current, scales)
    initial = torch.tensor(.5/ratio, dtype=torch.float64)

    def concentration(p):
        j = current(p[:, 1:])
        b = -j*r/(2*FARADAY_CONSTANT*d*scales.concentration_mol_m3)
        return initial+particle_mean_rate(j, scales)*p[:, 2:3]+b*(p[:, :1]**2-.6)

    def fields(p):
        surface = torch.cat((torch.ones_like(p[:, :1]), p), dim=1)
        theta = concentration(surface)*ratio
        ce, temp = p[:, :1]*0+1000., p[:, :1]*0+298.15
        phie = p[:, :1]*0-.1
        j0 = exchange_current(ce, theta*maximum, temp, electrode, mode="pybamm_26_8")
        return {"c_e": ce, "T": temp, "phi_e": phie,
                "phi_s": phie+ocp(theta, electrode)+inverse_bv(current(p), j0, temp)}
    return coupling, concentration, fields, initial


@pytest.mark.parametrize("electrode", ["n", "p"])
@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("ratio", [1., .5])
def test_manufactured_assembly(electrode, sign, ratio):
    coupling, concentration, fields, initial = case(electrode, sign, ratio)
    p = torch.tensor([[.2, .001], [.6, .002]], dtype=torch.float64, requires_grad=True)
    out = particle_interface(coupling, concentration, fields, p, electrode, 1., mode="pybamm_26_8")
    for key in ("flux_residual", "kinetics_residual"):
        torch.testing.assert_close(out[key], torch.zeros_like(out[key]), atol=1e-12, rtol=0)
    for radius in (0., .5, 1.):
        z = torch.cat((torch.full_like(p[:, :1], radius), p), dim=1)
        residual = diffusion_residual(concentration, z, coupling.scales)
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)
        if radius == 0:
            torch.testing.assert_close(center_residual(concentration, z), torch.zeros_like(residual))
    r, w = gauss_legendre(16, 0., 1.)
    z = torch.stack((r, p[0, 0].expand_as(r), p[0, 1].expand_as(r)), dim=1)
    mean = (3*r*r*w*concentration(z).flatten()).sum()
    torch.testing.assert_close(mean, coupling.mean_target(p, initial)[0, 0])


@pytest.mark.parametrize("electrode", ["n", "p"])
def test_surface_feedback_and_potential_gradients(electrode):
    coupling, concentration, fields, _ = case(electrode, 1.)
    p = torch.tensor([[.2, .001]], dtype=torch.float64, requires_grad=True)
    # Freeze manufactured potentials so they cannot cancel the perturbation.
    fixed = {k: v.detach() for k, v in fields(p).items()}
    delta = torch.tensor(.001, dtype=torch.float64, requires_grad=True)
    potential = torch.tensor(.002, dtype=torch.float64, requires_grad=True)
    modified = lambda z: concentration(z)+delta
    field_model = lambda z: {**fixed, "phi_s": fixed["phi_s"]+potential}
    out = particle_interface(coupling, modified, field_model, p, electrode, 1., mode="pybamm_26_8")
    gradients = torch.autograd.grad(out["kinetics_residual"].sum(), (delta, potential), create_graph=True)
    assert all(torch.isfinite(g) and g.abs() > 0 for g in gradients)
    assert torch.isfinite(torch.autograd.grad(gradients[0], delta)[0])
    torch.testing.assert_close(out["flux_residual"], torch.zeros_like(out["flux_residual"]), atol=1e-12, rtol=0)
    assert out["kinetics_residual"].abs().max() > 1e-4


def test_invalid_interface_inputs():
    coupling, concentration, fields, _ = case("n", 1.)
    for p in (torch.tensor([[.2, 0.]], dtype=torch.float64, requires_grad=True),
              torch.tensor([[.2, .001]], dtype=torch.float64)):
        with pytest.raises(ValueError):
            particle_interface(coupling, concentration, fields, p, "n", 1., mode="raw")
    p = torch.tensor([[.2, .001]], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="Field model"):
        particle_interface(coupling, concentration, lambda z: {}, p, "n", 1., mode="raw")
