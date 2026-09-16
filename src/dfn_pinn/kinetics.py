"""Symmetric, one-electron Butler-Volmer with unit interface utilization.

SI units: eta [V], j and j0 [A/m2 active area], temperature [K].
No clipping, j0 floor, film resistance or endpoint regularization is applied.
"""

import torch

from .constitutive import FARADAY_CONSTANT, GAS_CONSTANT, _check


def _inputs(signal, j0, temperature):
    _check(signal, "signal")
    _check(j0, "exchange current", 0)
    _check(temperature, "temperature", 0)
    if any(x.dtype != signal.dtype or x.device != signal.device for x in (j0, temperature)):
        raise ValueError("Inputs must share dtype and device")


def _finite(value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Butler-Volmer exceeded the numerical range; values were not clipped")
    return value


def direct_bv(eta, j0, temperature):
    """Return interfacial current; positive eta gives positive oxidation current."""
    _inputs(eta, j0, temperature)
    argument = _finite((FARADAY_CONSTANT / (2 * GAS_CONSTANT)) * (eta / temperature))
    return _finite(j0 * _finite(2 * torch.sinh(argument)))


def inverse_bv(current, j0, temperature):
    """Return overpotential for strictly positive j0; not defined at j0=0.

Uses the ordinary asinh ratio, not a log-domain extreme-range extension.
Overflow in that ratio is explicitly rejected even when a mathematical
overpotential would still be finite. Finite outputs do not ensure finite AD.
"""
    _inputs(current, j0, temperature)
    ratio = _finite((current * 0.5) / j0)
    return _finite((2 * GAS_CONSTANT / FARADAY_CONSTANT) * temperature * torch.asinh(ratio))
