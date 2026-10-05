"""Dimensionally scaled startup candidate; not wired into DFN training."""

import math
import torch
from torch import nn

from .constitutive import FARADAY_CONSTANT


class ParticleStartupCandidate(nn.Module):
    """Raw pointwise map: [rho^2, global X, sqrt(Fo/Fo_end), layer] -> scalar.

    Fo=D*t/R^2; layer=exp(-(1-rho^2)/(4*sqrt(Fo))). The exact zero-time
    branch defines the initial value only. PDE/flux derivatives are for t>0.
    No hard surface flux, inventory, concentration bounds or PDE solution.
    Scales and duration are constructor metadata for future checkpoint wiring.
    """

    def __init__(self, raw, scales, initial_concentration, current_scale, duration_s):
        super().__init__()
        if not math.isfinite(initial_concentration) or not 0 < initial_concentration < 1:
            raise ValueError("Initial normalized concentration must be in (0,1)")
        if not all(math.isfinite(v) and v > 0 for v in (current_scale, duration_s)):
            raise ValueError("Current scale and duration must be positive finite")
        self.raw, self.scales = raw, scales
        self.duration_s, self.current_scale = duration_s, current_scale
        self.beta = current_scale*scales.radius_m/(FARADAY_CONSTANT*scales.diffusivity_m2_s*scales.concentration_mol_m3)
        self.fo_end = scales.diffusivity_m2_s*duration_s/scales.radius_m**2
        self.register_buffer("initial", torch.tensor(initial_concentration, dtype=torch.float64))

    def forward(self, points):
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype != torch.float64:
            raise ValueError("Expected float64 [rho,X,tau] points")
        if not torch.isfinite(points).all() or not ((points[:, :2] >= 0) & (points[:, :2] <= 1)).all() or not (points[:, 2] >= 0).all():
            raise ValueError("Coordinates must be finite, rho/X in [0,1], tau>=0")
        rho2, tau = points[:, :1].square(), points[:, 2:3]
        positive = tau > 0
        fo = self.scales.diffusion_number*tau
        root = torch.sqrt(torch.where(positive, fo, torch.ones_like(fo)))
        layer = torch.exp(-(1-rho2)/(4*root))
        features = torch.cat((rho2, points[:, 1:2], root/math.sqrt(self.fo_end), layer), dim=1)
        correction = self.raw(features)
        if correction.shape != tau.shape:
            raise ValueError("Raw field must return (N,1)")
        return torch.where(positive, self.initial+self.beta*root*correction,
                           self.initial+points[:, :1]*0)
