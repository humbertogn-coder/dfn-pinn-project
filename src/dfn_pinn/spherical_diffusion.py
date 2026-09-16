"""Constant-D spherical diffusion residuals for pointwise concentration models."""

from dataclasses import dataclass
import math

import torch

from .constitutive import FARADAY_CONSTANT, _check


@dataclass(frozen=True)
class ParticleScales:
    radius_m: float
    diffusivity_m2_s: float
    concentration_mol_m3: float
    time_s: float = 3600.0

    def __post_init__(self):
        for value in (self.radius_m, self.diffusivity_m2_s, self.concentration_mol_m3, self.time_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Particle scales must be positive finite constants")

    @property
    def diffusion_number(self):
        return self.diffusivity_m2_s * self.time_s / self.radius_m**2


def _gradient(output, points):
    if not output.requires_grad:
        raise ValueError("Model derivatives must retain their autograd graph")
    gradient = torch.autograd.grad(output.sum(), points, create_graph=True,
                                   retain_graph=True, allow_unused=True)[0]
    # Keep exactly constant/linear derivatives differentiable at higher orders.
    return points*0 if gradient is None else gradient + points*0


def _derivatives(model, points):
    _check(points, "normalized particle points")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("Points must have shape (N,3): rho, global x/L, tau")
    if not points.requires_grad:
        raise ValueError("Points must require gradients")
    if not bool(((points[:, :2] >= 0) & (points[:, :2] <= 1)).all()) or not bool((points[:, 2] >= 0).all()):
        raise ValueError("Require rho,x/L in [0,1] and tau >= 0")
    concentration = model(points)
    _check(concentration, "normalized concentration")
    if concentration.shape != (len(points), 1) or concentration.dtype != points.dtype or concentration.device != points.device:
        raise ValueError("Model must return (N,1) with the points' dtype and device")
    first = _gradient(concentration, points)
    second_r = _gradient(first[:, :1], points)[:, :1]
    return first[:, :1], second_r, first[:, 2:3]


def diffusion_residual(model, points, scales):
    """PDE residual divided by c_ref/t_ref; includes regular limit at rho=0.

    Assumes a pointwise model (no batch normalization, attention or sample
    coupling), constant D and radial smoothness. Center symmetry must ALSO
    be enforced; the center limit alone does not enforce d_r c=0.
    """
    dr, drr, dt = _derivatives(model, points)
    rho = points[:, :1]
    denominator = torch.where(rho == 0, torch.ones_like(rho), rho)
    laplacian = torch.where(rho == 0, 3*drr, drr + 2*dr/denominator)
    return dt - scales.diffusion_number * laplacian


def center_residual(model, points):
    """Symmetry residual d(c/c_ref)/d(r/R), evaluated only at rho=0."""
    dr, _, _ = _derivatives(model, points)
    if not bool((points[:, 0] == 0).all()):
        raise ValueError("Center points must have rho=0")
    return dr


def surface_residual(model, points, current, scales, current_scale):
    """(-D*d_r c - j/F)/(j_ref/F), only at rho=1 and t>0.

    current is signed j [A/m2 active area], scalar or (N,1). The initial
    current-step corner is excluded rather than imposing incompatible data.
    """
    dr, _, _ = _derivatives(model, points)
    _check(current, "interfacial current")
    if current.dtype != points.dtype or current.device != points.device:
        raise ValueError("Current must share points dtype and device")
    if current.ndim != 0 and current.shape != (len(points), 1):
        raise ValueError("Current must be scalar or (N,1)")
    if not math.isfinite(current_scale) or current_scale <= 0:
        raise ValueError("Current scale must be positive and finite")
    if not bool((points[:, 0] == 1).all()) or not bool((points[:, 2] > 0).all()):
        raise ValueError("Surface flux requires rho=1 and tau>0")
    outward_flux = -scales.diffusivity_m2_s * scales.concentration_mol_m3 / scales.radius_m * dr
    return (FARADAY_CONSTANT*outward_flux-current)/current_scale


def volume_average(concentration, rho, weights):
    """Return 3*integral_0^1 c_hat*rho^2 d_rho along the last axis.

    Caller supplies a quadrature covering [0,1]; weights are d_rho, NOT r^2 dr
    and NOT already volume weighted. The geometric volume factor is explicit.
    """
    for value, name in ((concentration, "concentration"), (rho, "rho"), (weights, "weights")):
        _check(value, name)
    if rho.ndim != 1 or rho.numel() == 0 or weights.shape != rho.shape or concentration.ndim < 1 or concentration.shape[-1] != len(rho):
        raise ValueError("Final concentration axis must match 1D quadrature")
    if any(v.dtype != concentration.dtype or v.device != concentration.device for v in (rho, weights)):
        raise ValueError("Quadrature must share concentration dtype and device")
    if not bool(((rho >= 0) & (rho <= 1)).all()) or not bool((weights > 0).all()):
        raise ValueError("Invalid radial nodes or weights")
    return 3*(concentration*rho**2*weights).sum(dim=-1)
