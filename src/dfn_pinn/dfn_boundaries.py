"""One-sided electrolyte interface and collector residuals in global coordinates."""

from dataclasses import dataclass
import math
import torch

from .constitutive import _check
from .charge_conservation import electrolyte_charge_terms, solid_charge_terms, _field
from .electrolyte_mass import electrolyte_mass_terms


@dataclass(frozen=True)
class ElectrolyteBranch:
    concentration: object
    potential: object
    porosity: float
    bruggeman: float = 1.5
    transference: float = .2594
    thermodynamic_factor: float = 1.
    temperature_K: float = 298.15


def boundary_points(times, position):
    """Construct [global x/L,tau] with coordinate derivatives and time graph."""
    _check(times, "boundary times")
    if times.ndim != 2 or times.shape[1] != 1 or len(times) == 0 or not bool((times >= 0).all()):
        raise ValueError("Times must be nonempty (N,1), tau>=0")
    if not math.isfinite(position) or not 0 <= position <= 1:
        raise ValueError("Boundary position must lie in [0,1]")
    return torch.cat((torch.full_like(times, position), times), dim=1).requires_grad_()


def electrolyte_trace(branch, points, mass_scales, charge_scales):
    """Evaluate one branch at an exact boundary, not a last interior node.

    Local operators are reused only to extract fluxes; their zero-source
    divergence residuals are not boundary conditions or part of the output.
    """
    if mass_scales.length_m != charge_scales.length_m or mass_scales.concentration_mol_m3 != charge_scales.concentration_mol_m3:
        raise ValueError("Mass and charge scales must share global length and concentration")
    zero = points.new_tensor(0.)
    mass = electrolyte_mass_terms(branch.concentration, points, zero, mass_scales,
                                  porosity=branch.porosity, bruggeman=branch.bruggeman,
                                  transference=branch.transference)
    charge = electrolyte_charge_terms(branch.concentration, branch.potential, points, zero, charge_scales,
                                       porosity=branch.porosity, bruggeman=branch.bruggeman,
                                       transference=branch.transference, thermodynamic_factor=branch.thermodynamic_factor,
                                       temperature_K=branch.temperature_K)
    return {"concentration": mass["concentration_mol_m3"]/mass_scales.concentration_mol_m3,
            "potential": _field(branch.potential, points, "electrolyte potential"),
            "current": charge["current_A_m2"]/charge_scales.current_A_m2,
            "diffusive_flux": mass["diffusive_flux_mol_m2_s"]/(mass_scales.concentration_mol_m3*mass_scales.length_m/mass_scales.time_s)}


def electrolyte_interface(left, right, times, position, mass_scales, charge_scales):
    """Normalized left-minus-right c, phi, i_e, N jumps; gradients may differ.

    Caller supplies the correct adjacent branch pair and global interface
    position (L_n/L or (L_n+L_s)/L). No solid potential continuity is imposed.
    """
    if not 0 < position < 1:
        raise ValueError("An interface must be strictly inside the cell")
    left_trace = electrolyte_trace(left, boundary_points(times, position), mass_scales, charge_scales)
    right_trace = electrolyte_trace(right, boundary_points(times, position), mass_scales, charge_scales)
    return {name: left_trace[name]-right_trace[name] for name in left_trace}


def collector_conditions(side, branch, solid_potential, times, mass_scales,
                         charge_scales, *, conductivity_S_m, applied_current):
    """Independent collector BCs plus a separately labelled current diagnostic.

    side='negative': phi_s=0; side='positive': i_s=i_app. Both: i_e=N=0.
    Applied current is signed geometric density [A/m2], not cell current [A].
    Never add the returned diagnostic as another independent BC.
    """
    if side not in ("negative", "positive"):
        raise ValueError("Collector side must be negative or positive")
    p = boundary_points(times, 0. if side == "negative" else 1.)
    _check(applied_current, "applied geometric current")
    if applied_current.dtype != p.dtype or applied_current.device != p.device or (applied_current.ndim != 0 and applied_current.shape != (len(p), 1)):
        raise ValueError("Applied current must be scalar or (N,1), with input dtype/device")
    trace = electrolyte_trace(branch, p, mass_scales, charge_scales)
    solid = solid_charge_terms(solid_potential, p, p.new_tensor(0.), charge_scales,
                               conductivity_S_m=conductivity_S_m)["current_A_m2"]/charge_scales.current_A_m2
    applied = applied_current/charge_scales.current_A_m2
    conditions = {"electrolyte_current": trace["current"], "diffusive_flux": trace["diffusive_flux"]}
    if side == "negative":
        conditions["solid_potential_gauge"] = _field(solid_potential, p, "solid potential")
    else:
        conditions["solid_current"] = solid-applied
    return {"conditions": conditions, "total_current_diagnostic": solid+trace["current"]-applied}


def solid_separator_condition(potential, times, position, scales, *, conductivity_S_m):
    """Zero electronic current at either electrode/separator interface."""
    if not 0 < position < 1:
        raise ValueError("Solid/separator boundary must be inside the cell")
    p = boundary_points(times, position)
    return solid_charge_terms(potential, p, p.new_tensor(0.), scales,
                              conductivity_S_m=conductivity_S_m)["current_A_m2"]/scales.current_A_m2
