"""Surface-confined startup candidate; no training or factory integration."""

import math
import torch

from .particle_startup_candidate import ParticleStartupCandidate


class ParticleConfinedStartup(ParticleStartupCandidate):
    """Same four-input raw network, evaluated separately for bulk and layer.

    c_hat = initial + beta*(Fo*raw_bulk + sqrt(Fo)*layer*raw_surface).
    Bulk time input is Fo/Fo_end; surface time input is sqrt(Fo/Fo_end).
    Shared weights do not impose flux, inventory, positivity or a PDE solution.
    Exact t=0 defines values only; differentiate the physics at t>0.
    """

    def contributions(self, points):
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype != torch.float64:
            raise ValueError("Expected float64 [rho,X,tau] points")
        if not torch.isfinite(points).all() or not ((points[:, :2] >= 0) & (points[:, :2] <= 1)).all() or not (points[:, 2] >= 0).all():
            raise ValueError("Coordinates must be finite, rho/X in [0,1], tau>=0")
        rho2, tau = points[:, :1].square(), points[:, 2:3]
        positive = tau > 0
        fo = self.scales.diffusion_number*tau
        root = torch.sqrt(torch.where(positive, fo, torch.ones_like(fo)))
        layer = torch.exp(-(1-rho2)/(4*root))
        bulk_features = torch.cat((rho2, points[:, 1:2], fo/self.fo_end, torch.zeros_like(fo)), dim=1)
        surface_features = torch.cat((rho2, points[:, 1:2], root/math.sqrt(self.fo_end), layer), dim=1)
        bulk_raw, surface_raw = self.raw(bulk_features), self.raw(surface_features)
        if bulk_raw.shape != tau.shape or surface_raw.shape != tau.shape:
            raise ValueError("Raw field must return (N,1)")
        bulk = self.beta*fo*bulk_raw
        surface = torch.where(positive, self.beta*root*layer*surface_raw, points[:, :1]*0)
        return bulk, surface

    def forward(self, points):
        bulk, surface = self.contributions(points)
        return self.initial+bulk+surface
