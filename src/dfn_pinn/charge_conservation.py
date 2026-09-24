"""Region-local DFN electrical currents and charge balances."""

from dataclasses import dataclass
import math

from .constitutive import FARADAY_CONSTANT, GAS_CONSTANT, _check, electrolyte_conductivity
from .spherical_diffusion import _gradient


@dataclass(frozen=True)
class ChargeScales:
    length_m: float = 172.8e-6
    concentration_mol_m3: float = 1000.
    potential_V: float = GAS_CONSTANT*298.15/FARADAY_CONSTANT
    current_A_m2: float = 5/.1027

    def __post_init__(self):
        for value in (self.length_m, self.concentration_mol_m3, self.potential_V, self.current_A_m2):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Charge scales must be positive finite constants")


def _inputs(points, aj):
    _check(points, "charge points")
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not points.requires_grad:
        raise ValueError("Expected nonempty (N,2) points requiring gradients")
    if not bool(((points[:, 0] >= 0) & (points[:, 0] <= 1)).all()) or not bool((points[:, 1] >= 0).all()):
        raise ValueError("Require global X in [0,1] and tau>=0")
    _check(aj, "volumetric reaction current")
    if aj.dtype != points.dtype or aj.device != points.device or (aj.ndim != 0 and aj.shape != (len(points), 1)):
        raise ValueError("aj must be scalar or (N,1) with input dtype/device")


def _field(model, points, name, positive=False):
    value = model(points)
    _check(value, name, 0 if positive else None)
    if value.shape != (len(points), 1) or value.dtype != points.dtype or value.device != points.device:
        raise ValueError("Field must return (N,1) with input dtype/device")
    return value


def _balance(current, points, source, scales):
    divergence = _gradient(current, points)[:, :1]/scales.length_m
    residual = divergence+source
    normalized = residual/(scales.current_A_m2/scales.length_m)
    for value in (current, divergence, residual, normalized):
        _check(value, "charge output")
    return {"current_A_m2": current, "divergence_A_m3": divergence,
            "balance_A_m3": residual, "normalized_residual": normalized}


def solid_charge_terms(potential_model, points, aj, scales, *, conductivity_S_m):
    """i_s=-sigma_eff*d_x(phi_s), balance=d_x(i_s)+aj.

    Inputs [global x/L,t/t_ref], output phi_s/phi_ref. Supply EFFECTIVE
    conductivity in S/m explicitly; no hidden solid Bruggeman factor.
    aj is A/m3; i_s is per geometric area. Constants and pointwise fields only.
    """
    _inputs(points, aj)
    if not math.isfinite(conductivity_S_m) or conductivity_S_m <= 0:
        raise ValueError("Effective conductivity must be positive and finite")
    potential = _field(potential_model, points, "solid potential")
    current = -conductivity_S_m*scales.potential_V/scales.length_m*_gradient(potential, points)[:, :1]
    return _balance(current, points, aj, scales)


def electrolyte_charge_terms(concentration_model, potential_model, points, aj, scales,
                             *, porosity, bruggeman=1.5, transference=.2594,
                             thermodynamic_factor=1., temperature_K=298.15):
    """i_e=kappa_eff*[B*d_x(log c_e)-d_x(phi_e)], balance=d_x(i_e)-aj.

    B=2*R*T/F*(1-t_plus)*chi. Models output c_e/c_ref and phi_e/phi_ref.
    Constants are region-local; do not differentiate coefficient jumps.
    Use aj=0 in the separator. No gauge, boundary or interface is imposed.
    """
    _inputs(points, aj)
    if not math.isfinite(porosity) or not 0 < porosity <= 1:
        raise ValueError("Porosity must lie in (0,1]")
    if not math.isfinite(bruggeman) or bruggeman < 0:
        raise ValueError("Bruggeman exponent must be finite and nonnegative")
    if not math.isfinite(transference) or not 0 <= transference < 1:
        raise ValueError("Transference must lie in [0,1)")
    if not math.isfinite(temperature_K) or temperature_K <= 0 or not math.isfinite(thermodynamic_factor) or thermodynamic_factor <= 0:
        raise ValueError("Temperature and thermodynamic factor must be positive finite constants")
    concentration = _field(concentration_model, points, "electrolyte concentration", True)
    potential = _field(potential_model, points, "electrolyte potential")
    conductivity = porosity**bruggeman*electrolyte_conductivity(scales.concentration_mol_m3*concentration)
    _check(conductivity, "effective electrolyte conductivity", 0)
    log_gradient = _gradient(concentration, points)[:, :1]/concentration/scales.length_m
    potential_gradient = scales.potential_V/scales.length_m*_gradient(potential, points)[:, :1]
    coefficient = 2*GAS_CONSTANT*temperature_K/FARADAY_CONSTANT*(1-transference)*thermodynamic_factor
    current = conductivity*(coefficient*log_gradient-potential_gradient)
    return _balance(current, points, -aj, scales)
