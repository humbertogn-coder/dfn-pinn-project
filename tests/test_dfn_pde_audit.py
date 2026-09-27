"""Focused checks for independent DFN equation assembly and physical measures."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dfn_pinn.dfn_smoke import DFNSmoke
from dfn_pinn.dfn_pde_audit import audit_pdes, equation, physical_rms
from dfn_pinn.constitutive import FARADAY_CONSTANT


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, p):
        return p[:, :1]*0+self.value


def test_spherical_and_physical_time_weights():
    settings = {"duration_s": 2., "time_reference_s": 10., "lengths_m": [.1, .2, .3]}
    value = physical_rms(lambda p: p[:, :1]+0*p[:, -1:], (.1, .6), settings, .01, 8, True)
    assert value == pytest.approx(np.sqrt(3/5))
    value = physical_rms(lambda p: 10*p[:, -1:], (.1, .6), settings, 1., 8)
    assert value == pytest.approx(np.sqrt(7/3))


def test_constant_fields_have_signed_reaction_sources():
    model = DFNSmoke()
    p = torch.tensor([[.2, .0001]], dtype=torch.float64, requires_grad=True)
    for i in range(3):
        model.ce[i], model.phie[i] = Constant(1.), Constant(0.)
    for k, i in enumerate((0, 2)):
        model.phis[k] = Constant(0.)
        model.reactions[k].integral.current_model = Constant((-1)**k*.2)
        aj = model.active_area[k]*((-1)**k*.2)
        charge = aj/(model.charge_scales.current_A_m2/model.charge_scales.length_m)
        assert equation(model, i, "charge_s")(p).item() == pytest.approx(charge)
        assert equation(model, i, "charge_e")(p).item() == pytest.approx(-charge)
        salt = -(1-.2594)*aj/FARADAY_CONSTANT/(1000/3600)
        assert equation(model, i, "salt")(p).item() == pytest.approx(salt)
    assert equation(model, 1, "salt")(p).item() == 0
    assert equation(model, 1, "charge_e")(p).item() == 0


def test_radial_polynomial_chain_rule():
    model = DFNSmoke()
    class Polynomial(torch.nn.Module):
        def forward(self, p):
            return .5+.01*p[:, :1]**2+.2*p[:, 2:3]
    model.cs[0] = Polynomial()
    p = torch.tensor([[0., .1, .0001], [1., .1, .0002]], dtype=torch.float64, requires_grad=True)
    expected = .2-.06*model.reactions[0].scales.diffusion_number
    np.testing.assert_allclose(equation(model, 0, "particle")(p).detach().numpy(), expected)


def test_audit_covers_ten_equations_without_training_assembly():
    config = json.loads((Path(__file__).resolve().parents[1]/"configs/dfn_baseline_v1.json").read_text())
    config["audit"].update(uniform_time_points=3, log_time_points=3, region_x_points=3,
                           particle_r_points=3, quadrature_orders=[2, 3], quadrature_refinement_order=4)
    torch.manual_seed(42)
    model = DFNSmoke().eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    def forbidden(*args):
        raise AssertionError("Training residual assembly must not be called")
    model.residuals = forbidden
    report = audit_pdes(model, config)
    assert len(report["metrics"]) == 30
    assert len(report["diagnostics"]) == 10
    assert report["status"] == "PDE_DIAGNOSTIC_ONLY_NOT_ACCEPTANCE"
    assert any(not m["pass"] for m in report["metrics"].values())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    json.dumps(report, allow_nan=False)


def test_underresolution_and_nonfinite_are_visible():
    settings = {"duration_s": 1., "time_reference_s": 1., "lengths_m": [1.]}
    f = lambda p: p[:, :1]**8
    coarse = physical_rms(f, (0., 1.), settings, .001, 2)
    fine = physical_rms(f, (0., 1.), settings, .001, 12)
    assert abs(coarse-fine) > .01
    assert fine == pytest.approx(1/np.sqrt(17))
    with pytest.raises(ValueError, match="finite"):
        physical_rms(Constant(float("nan")), (0., 1.), settings, .001, 4)
