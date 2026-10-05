"""Differentiable Chen2020 constitutive laws for training (torch).

Same expressions as ``dfn_pinn.constitutive`` (audited against PyBaMM), but
without the hard range checks: during optimization a network can transiently
leave the admissible range, and raising an exception there would kill the run.
Instead, the *arguments* of the square roots / fractional powers are softly
floored. This only matters away from the physical solution; the test
``tests/test_v2_constitutive.py`` checks equality with v1 in the interior.
"""

from __future__ import annotations

import torch

from .params import FARADAY, GAS_CONSTANT

_EPS = 1e-6


def _floor(x, lo):
    """Smooth lower bound: ~x for x >> lo, -> lo/... never below lo/2."""
    return lo + torch.nn.functional.softplus(x - lo, beta=1.0 / lo) if lo > 0 else x


def ocp_n(theta):
    s = theta
    return (1.9793 * torch.exp(-39.3631 * s) + 0.2482
            - 0.0909 * torch.tanh(29.8538 * (s - 0.1234))
            - 0.04478 * torch.tanh(14.9159 * (s - 0.2769))
            - 0.0205 * torch.tanh(30.4444 * (s - 0.6103)))


def ocp_p(theta):
    s = theta
    return (-0.8090 * s + 4.4875
            - 0.0428 * torch.tanh(18.5138 * (s - 0.5542))
            - 17.7326 * torch.tanh(15.7890 * (s - 0.3117))
            + 17.5842 * torch.tanh(15.9308 * (s - 0.3120)))


def ocp(theta, k):
    return ocp_n(theta) if k == "n" else ocp_p(theta)


def exchange_current(c_e, theta, k, cell):
    """j0 [A/m2] = k c_e^0.5 c_s^0.5 (cmax - c_s)^0.5 (T = 298.15 K, no Arrhenius)."""
    cmax = cell.cmax_n if k == "n" else cell.cmax_p
    rate = cell.k_n if k == "n" else cell.k_p
    th = _floor(theta, 1e-4)
    om = _floor(1 - theta, 1e-4)
    ce = _floor(c_e, 1.0)
    return rate * torch.sqrt(ce) * cmax * torch.sqrt(th * om)


def electrolyte_diffusivity(c_e):
    c = c_e / 1000.0
    return 8.794e-11 * c * c - 3.972e-10 * c + 4.862e-10


def electrolyte_conductivity(c_e):
    c = _floor(c_e, 1.0) / 1000.0
    return 0.1297 * c ** 3 - 2.51 * c ** 1.5 + 3.329 * c


def bv_current(eta, j0, T):
    """Symmetric BV, oxidation positive: j = 2 j0 sinh(F eta / (2 R T))."""
    return 2 * j0 * torch.sinh(FARADAY * eta / (2 * GAS_CONSTANT * T))


def bv_overpotential(j, j0, T):
    """Inverse symmetric BV: eta = (2 R T / F) asinh(j / (2 j0))."""
    return 2 * GAS_CONSTANT * T / FARADAY * torch.asinh(j / (2 * j0))
