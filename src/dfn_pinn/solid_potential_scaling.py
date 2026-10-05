"""Isolated output-amplitude candidate; not installed in any DFN trainer."""

import math

from .dfn_smoke import Field


def ohmic_amplitude(scales, conductivity_S_m):
    """Use global X=x/L so normalized solid current equals -d(raw)/dX.

    This is a representation scale, not a boundary condition or a constraint
    on the predicted current. Use the positive reference current magnitude.
    """
    if not math.isfinite(conductivity_S_m) or conductivity_S_m <= 0:
        raise ValueError("Conductivity must be positive and finite")
    amplitude = (scales.current_A_m2 * scales.length_m
                 / (conductivity_S_m * scales.potential_V))
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("Ohmic amplitude must be positive and finite")
    return amplitude


def scaled_solid_potential(base, settings, scales, conductivity_S_m):
    """Keep Field architecture, base, global coordinates and time features.

    Like Field, amplitude is constructor metadata, not part of state_dict.
    Future checkpoint integration must explicitly persist and verify it.
    """
    return Field(2, base, "potential", settings,
                 amplitude=ohmic_amplitude(scales, conductivity_S_m))
