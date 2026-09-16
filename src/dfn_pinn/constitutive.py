"""Differentiable SI Chen2020 raw fits, audited against PyBaMM 26.8.0.0.

No PyBaMM/NumPy calls or clipping are used. Exchange-current regularization
is available only through an explicit mode.
Inputs must be floating tensors; dtype, device and autograd history are retained.
Raw fits are not replacements for PyBaMM's endpoint-regularized expressions.
"""

import torch

GAS_CONSTANT = 8.31446261815324  # J/(mol K), exporting PyBaMM constant
FARADAY_CONSTANT = 96485.33212331001  # C/mol
REFERENCE_TEMPERATURE = 298.15
MAX_CONCENTRATION = {"n": 33133.0, "p": 63104.0}
SOLID_DIFFUSIVITY = {"n": 3.3e-14, "p": 4e-15}  # m2/s at reference T
SOLID_CONDUCTIVITY = {"n": 215.0, "p": 0.18}  # S/m, before transport factor


def _check(value, name, lower=None, upper=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    if lower is not None and not bool((value > lower).all()):
        raise ValueError(f"{name} must be greater than {lower}")
    if upper is not None and not bool((value < upper).all()):
        raise ValueError(f"{name} must be less than {upper}")


def _electrode(electrode):
    if electrode not in ("n", "p"):
        raise ValueError("Electrode must be 'n' or 'p'")


def ocp(stoichiometry, electrode):
    """Raw open-circuit potential [V]; Chen2020 entropic correction is zero."""
    _electrode(electrode)
    _check(stoichiometry, "stoichiometry", 0, 1)
    s = stoichiometry
    if electrode == "n":
        return (1.9793 * torch.exp(-39.3631 * s) + 0.2482
                - 0.0909 * torch.tanh(29.8538 * (s - 0.1234))
                - 0.04478 * torch.tanh(14.9159 * (s - 0.2769))
                - 0.0205 * torch.tanh(30.4444 * (s - 0.6103)))
    return (-0.8090 * s + 4.4875
            - 0.0428 * torch.tanh(18.5138 * (s - 0.5542))
            - 17.7326 * torch.tanh(15.7890 * (s - 0.3117))
            + 17.5842 * torch.tanh(15.9308 * (s - 0.3120)))


def exchange_current(c_e, c_s_surface, temperature, electrode, *, mode="raw"):
    """Exchange current [A/m2]; mode='pybamm_26_8' uses explicit RegPower."""
    _electrode(electrode)
    if mode not in ("raw", "pybamm_26_8"):
        raise ValueError("Unknown exchange-current mode")
    maximum = MAX_CONCENTRATION[electrode]
    _check(c_e, "electrolyte concentration", 0)
    _check(c_s_surface, "surface concentration", 0, maximum)
    _check(temperature, "temperature", 0)
    if any(v.dtype != c_e.dtype or v.device != c_e.device for v in (c_s_surface, temperature)):
        raise ValueError("Inputs must share dtype and device")
    rate, activation = (6.48e-7, 35000.0) if electrode == "n" else (3.42e-6, 17800.0)
    arrhenius = torch.exp(activation / GAS_CONSTANT * (1 / REFERENCE_TEMPERATURE - 1 / temperature))
    if mode == "pybamm_26_8":
        # PyBaMM's scaled RegPower with exponent 1/2 and delta=0.001.
        def regularized_sqrt(value, scale):
            x = value / scale
            return x * (x*x + 0.001**2)**(-0.25) * scale**0.5

        return (rate * arrhenius * regularized_sqrt(c_e, 1000.0)
                * regularized_sqrt(c_s_surface, maximum)
                * regularized_sqrt(maximum - c_s_surface, maximum))
    return rate * arrhenius * c_e.sqrt() * c_s_surface.sqrt() * (maximum - c_s_surface).sqrt()


def electrolyte_diffusivity(c_e):
    """Nyman2008 bulk D_e [m2/s]; this fit has no temperature dependence."""
    _check(c_e, "electrolyte concentration", 0)
    c = c_e / 1000
    return 8.794e-11 * c**2 - 3.972e-10 * c + 4.862e-10


def electrolyte_conductivity(c_e):
    """Nyman2008 bulk kappa_e [S/m]; porosity correction is not included."""
    _check(c_e, "electrolyte concentration", 0)
    c = c_e / 1000
    return 0.1297 * c**3 - 2.51 * c**1.5 + 3.329 * c
