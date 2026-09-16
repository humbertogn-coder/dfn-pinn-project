"""Value and first/second derivative audits of raw and processed expressions."""

import numpy as np
import pybamm
import pytest
import torch

from dfn_pinn.constitutive import (
    MAX_CONCENTRATION, electrolyte_conductivity, electrolyte_diffusivity,
    exchange_current, ocp,
)


def tensor(values):
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def compare(function, expression, symbol, samples, scale=1.0, inputs=None):
    x = tensor(samples)
    actual = function(x)
    for order in range(3):
        expected = np.array([float(np.asarray(expression.evaluate(inputs={**(inputs or {}), "z": float(v)})).item())
                             for v in samples])
        np.testing.assert_allclose(actual.detach().numpy() * scale, expected * scale,
                                   rtol=2e-9, atol=2e-10)
        if order < 2:
            actual = torch.autograd.grad(actual.sum(), x, create_graph=True)[0]
            expression = expression.diff(symbol)


@pytest.fixture
def parameters():
    assert pybamm.__version__ == "26.8.0.0", "Re-audit constitutive provenance after version changes"
    return pybamm.ParameterValues("Chen2020")


@pytest.mark.parametrize("electrode,domain", [("n", "Negative"), ("p", "Positive")])
def test_ocp_raw_and_processed(parameters, electrode, domain):
    z = pybamm.InputParameter("z")
    samples = np.linspace(.01, .99, 41)
    raw = parameters[f"{domain} electrode OCP [V]"](z)
    compare(lambda x: ocp(x, electrode), raw, z, samples)
    param = pybamm.LithiumIonParameters()
    phase = getattr(param, electrode).prim
    processed = parameters.process_symbol(phase.U(z, pybamm.Scalar(298.15)))
    compare(lambda x: ocp(x, electrode), processed, z, samples)


@pytest.mark.parametrize("function,key", [
    (electrolyte_diffusivity, "Electrolyte diffusivity [m2.s-1]"),
    (electrolyte_conductivity, "Electrolyte conductivity [S.m-1]"),
])
def test_transport_derivatives(parameters, function, key):
    z = pybamm.InputParameter("z")
    # Input is concentration / 1000; scale small diffusivities before comparison.
    scale = 1e10 if function is electrolyte_diffusivity else 1
    expression = parameters[key](1000*z, pybamm.Scalar(298.15))
    compare(lambda x: function(1000*x), expression, z, np.linspace(.1, 3, 31), scale)


@pytest.mark.parametrize("electrode,domain", [("n", "Negative"), ("p", "Positive")])
@pytest.mark.parametrize("axis", ["electrolyte", "surface", "temperature"])
def test_exchange_derivatives(parameters, electrode, domain, axis):
    z = pybamm.InputParameter("z")
    maximum = MAX_CONCENTRATION[electrode]
    symbolic = {"electrolyte": 1000*z, "surface": maximum*z, "temperature": 298.15*z}
    base = {"electrolyte": pybamm.Scalar(1000), "surface": pybamm.Scalar(.5*maximum),
            "temperature": pybamm.Scalar(298.15)}
    base[axis] = symbolic[axis]
    raw = parameters[f"{domain} electrode exchange-current density [A.m-2]"](
        base["electrolyte"], base["surface"], maximum, base["temperature"])

    def function(x, mode="raw"):
        values = {"electrolyte": x*0+1000, "surface": x*0+.5*maximum,
                  "temperature": x*0+298.15}
        values[axis] = x * {"electrolyte": 1000, "surface": maximum, "temperature": 298.15}[axis]
        return exchange_current(values["electrolyte"], values["surface"], values["temperature"], electrode, mode=mode)

    samples = np.linspace(.01, .99, 17) if axis == "surface" else np.linspace(.95, 1.05, 17)
    if axis == "electrolyte":
        samples = np.linspace(.1, 3, 17)
    compare(function, raw, z, samples)
    phase = getattr(pybamm.LithiumIonParameters(), electrode).prim
    # Keep every concentration symbolic so processing regularizes all three
    # square roots, as in the DFN, rather than folding fixed concentrations.
    for name in base:
        if name != axis:
            base[name] = pybamm.InputParameter(name)
    processed = parameters.process_symbol(phase.j0(base["electrolyte"], base["surface"], base["temperature"]))
    assert pybamm.settings.tolerances["reg_power"] == .001
    compare(lambda x: function(x, "pybamm_26_8"), processed, z, samples,
            inputs={"electrolyte": 1000., "surface": .5*maximum, "temperature": 298.15})


@pytest.mark.parametrize("electrode", ["n", "p"])
@pytest.mark.parametrize("mode", ["raw", "pybamm_26_8"])
def test_exchange_gradcheck(electrode, mode):
    inputs = (tensor([500., 2000.]), tensor([.2, .8])*MAX_CONCENTRATION[electrode], tensor(298.15))
    function = lambda *args: exchange_current(*args, electrode, mode=mode)
    assert torch.autograd.gradcheck(function, inputs)
    assert torch.autograd.gradgradcheck(function, inputs)


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf")])
def test_invalid_concentration(bad):
    with pytest.raises(ValueError):
        electrolyte_conductivity(tensor(bad))


def test_invalid_stoichiometry_and_electrode():
    for sto in [0., 1., -0.1, 1.1]:
        with pytest.raises(ValueError):
            ocp(tensor(sto), "n")
    with pytest.raises(ValueError):
        ocp(tensor(.5), "unknown")
    with pytest.raises(TypeError):
        ocp(torch.tensor(1), "n")


def test_dtype_and_broadcast():
    x = torch.tensor([[.2], [.8]], dtype=torch.float32, requires_grad=True)
    result = ocp(x, "p")
    assert result.dtype == x.dtype and result.shape == x.shape and result.device == x.device
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
