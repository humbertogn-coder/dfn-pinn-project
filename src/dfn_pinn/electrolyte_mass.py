"""Region-local electrolyte salt balance for constant porosity and t_plus."""

from dataclasses import dataclass
import math

from .constitutive import FARADAY_CONSTANT, _check, electrolyte_diffusivity
from .spherical_diffusion import _gradient


@dataclass(frozen=True)
class ElectrolyteScales:
    length_m: float = 172.8e-6
    time_s: float = 3600.
    concentration_mol_m3: float = 1000.

    def __post_init__(self):
        for value in (self.length_m, self.time_s, self.concentration_mol_m3):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Electrolyte scales must be positive finite constants")


def electrolyte_mass_terms(model, points, aj, scales, *, porosity,
                           bruggeman=1.5, transference=0.2594):
    """Return flux [mol/m2/s], SI balance and normalized balance at (X,tau).

    model returns (N,1) c_e/c_ref, X=x/L is GLOBAL and tau=t/t_ref.
    aj is signed volumetric reaction current [A/m3], scalar or (N,1).
    Use aj=0 in the separator. Positive aj produces electrolyte salt.
    Porosity, exponent and transference are constants within this region.
    N=-epsilon^b*D(c_e)*dc_e/dx; balance=epsilon*dc_e/dt+dN/dx-(1-t_plus)*aj/F.
    The normalized balance divides by c_ref/t_ref. No interface/BC is imposed.
    Models must be pointwise; do not differentiate through coefficient jumps.
    """
    if not math.isfinite(porosity) or not 0 < porosity <= 1:
        raise ValueError("Porosity must lie in (0,1]")
    if not math.isfinite(bruggeman) or bruggeman < 0:
        raise ValueError("Bruggeman exponent must be finite and nonnegative")
    if not math.isfinite(transference) or not 0 <= transference < 1:
        raise ValueError("Transference must lie in [0,1)")
    _check(points, "electrolyte points")
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not points.requires_grad:
        raise ValueError("Expected nonempty (N,2) points requiring gradients")
    if not bool(((points[:, 0] >= 0) & (points[:, 0] <= 1)).all()) or not bool((points[:, 1] >= 0).all()):
        raise ValueError("Require global X in [0,1] and tau>=0")
    _check(aj, "volumetric reaction current")
    if aj.dtype != points.dtype or aj.device != points.device or (aj.ndim != 0 and aj.shape != (len(points), 1)):
        raise ValueError("aj must be scalar or (N,1), with input dtype/device")
    concentration = model(points)
    _check(concentration, "normalized electrolyte concentration", 0)
    if concentration.shape != (len(points), 1) or concentration.dtype != points.dtype or concentration.device != points.device:
        raise ValueError("Model must return (N,1) with input dtype/device")
    derivative = _gradient(concentration, points)
    physical_c = scales.concentration_mol_m3*concentration
    diffusivity = porosity**bruggeman*electrolyte_diffusivity(physical_c)
    _check(diffusivity, "effective diffusivity", 0)
    flux = -diffusivity*scales.concentration_mol_m3/scales.length_m*derivative[:, :1]
    divergence = _gradient(flux, points)[:, :1]/scales.length_m
    accumulation = porosity*scales.concentration_mol_m3/scales.time_s*derivative[:, 1:2]
    source = (1-transference)*aj/FARADAY_CONSTANT
    balance = accumulation+divergence-source
    normalized = balance/(scales.concentration_mol_m3/scales.time_s)
    for value in (flux, balance, normalized):
        _check(value, "electrolyte balance output")
    return {"concentration_mol_m3": physical_c, "diffusive_flux_mol_m2_s": flux,
            "accumulation_mol_m3_s": accumulation, "flux_divergence_mol_m3_s": divergence,
            "source_mol_m3_s": source, "balance_mol_m3_s": balance,
            "normalized_residual": normalized}
