"""Local spherical inventory contract for signed interfacial current in SI."""

from .constitutive import FARADAY_CONSTANT, _check
from .spherical_diffusion import ParticleScales


def particle_mean_rate(current, scales: ParticleScales):
    """Return d(mean(c/c_ref))/d(t/t_ref) from j [A/m2 active area].

    Positive j removes lithium; negative j inserts it. This is local in x.
    Do not supply applied cell current [A] or volumetric a*j [A/m3].
    """
    _check(current, "interfacial current")
    rate = -3*scales.time_s*current/(FARADAY_CONSTANT*scales.radius_m*scales.concentration_mol_m3)
    _check(rate, "normalized mean rate")
    return rate


def particle_mean_from_charge(initial_mean, surface_charge, scales: ParticleScales):
    """Mean c/c_ref from signed integral j dt [C/m2 active area] since t=0.

    Integration uses PHYSICAL seconds: no extra t_ref factor belongs here.
    initial_mean is dimensionless; it is scalar or matches surface_charge.
    Charge must come from the same current used in kinetics/flux residuals.
    This function neither integrates current nor solves or projects diffusion.
    """
    _check(initial_mean, "initial normalized mean")
    _check(surface_charge, "cumulative interfacial charge")
    if initial_mean.dtype != surface_charge.dtype or initial_mean.device != surface_charge.device:
        raise ValueError("Initial mean and charge must share dtype and device")
    if initial_mean.ndim != 0 and initial_mean.shape != surface_charge.shape:
        raise ValueError("Initial mean must be scalar or match cumulative charge")
    result = initial_mean-3*surface_charge/(FARADAY_CONSTANT*scales.radius_m*scales.concentration_mol_m3)
    _check(result, "normalized mean inventory")
    return result
