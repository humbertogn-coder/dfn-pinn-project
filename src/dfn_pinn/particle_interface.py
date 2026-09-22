"""Assemble local particle surface and kinetics constraints without detaching."""

import torch

from .constitutive import MAX_CONCENTRATION, _check, _electrode


def particle_interface(particle_current, concentration_model, field_model,
                       points, electrode, current_scale, *, mode):
    """Return surface concentration, flux and BV residuals at (x/L,tau).

    concentration_model: (N,3) [rho,x/L,tau] -> (N,1) c/c_ref.
    field_model: (N,2) -> dict of (N,1) SI tensors named c_e, phi_s, phi_e, T.
    Points require gradients and tau>0. Models must be pointwise and
    deterministic. This does not solve diffusion, impose initial conditions,
    enforce inventory, or solve electrolyte/charge equations.
    """
    _electrode(electrode)
    _check(points, "interface points")
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("Expected nonempty (N,2) interface points")
    if not points.requires_grad:
        raise ValueError("Interface points must require gradients")
    if not bool(((points[:, 0] >= 0) & (points[:, 0] <= 1)).all()) or not bool((points[:, 1] > 0).all()):
        raise ValueError("Require x/L in [0,1] and tau>0")
    surface = torch.cat((torch.ones_like(points[:, :1]), points), dim=1)
    concentration = concentration_model(surface)
    _check(concentration, "surface concentration")
    if concentration.shape != (len(points), 1) or concentration.dtype != points.dtype or concentration.device != points.device:
        raise ValueError("Concentration model must return (N,1) with input dtype and device")
    theta = concentration*particle_current.scales.concentration_mol_m3/MAX_CONCENTRATION[electrode]
    fields = field_model(points)
    if not isinstance(fields, dict) or set(fields) != {"c_e", "phi_s", "phi_e", "T"}:
        raise ValueError("Field model must return c_e, phi_s, phi_e, T")
    kinetics = particle_current.kinetics_residual(
        points, theta, fields["c_e"], fields["phi_s"], fields["phi_e"],
        fields["T"], electrode, current_scale, mode=mode)
    flux = particle_current.flux_residual(concentration_model, surface, current_scale)
    return {"surface_concentration": concentration, "surface_stoichiometry": theta,
            "flux_residual": flux, "kinetics_residual": kinetics}
