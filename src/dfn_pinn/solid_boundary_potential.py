"""Fixed-current solid-potential boundary candidate, not wired into a trainer."""

import math
import torch
from torch import nn

from .solid_potential_scaling import ohmic_amplitude


class SolidBoundaryPotential(nn.Module):
    """Raw smooth field accepts global [X,tau]; returns an unscaled correction.

    Positive reference current and constant effective conductivity only.
    The negative collector gauge is zero; the positive potential stays free.
    Geometry and scale are constructor metadata and must be persisted by any
    future checkpoint integration. No full-DFN factory supports this class yet.
    """

    def __init__(self, raw, scales, conductivity_S_m, bounds, electrode, base=0.):
        super().__init__()
        if electrode not in ("n", "p"):
            raise ValueError("Electrode must be n or p")
        left, right = bounds
        if not all(math.isfinite(v) for v in (left, right, base)) or not 0 <= left < right <= 1:
            raise ValueError("Invalid global bounds or base")
        if (electrode == "n" and (left != 0 or base != 0)) or (electrode == "p" and right != 1):
            raise ValueError("Collector geometry and negative zero gauge are required")
        self.raw = raw
        self.left, self.right, self.electrode = left, right, electrode
        self.amplitude = ohmic_amplitude(scales, conductivity_S_m)*(right-left)
        self.register_buffer("base", torch.tensor(base, dtype=torch.float64))

    def forward(self, points):
        width = self.right-self.left
        xi = (points[:, :1]-self.left)/width
        # Smoothstep makes the raw correction's spatial derivative zero at both ends.
        warped_x = self.left+width*xi.square()*(3-2*xi)
        correction = self.raw(torch.cat((warped_x, points[:, 1:2]), dim=1))
        if self.electrode == "n":
            anchor = torch.cat((torch.full_like(xi, self.left), points[:, 1:2]), dim=1)
            correction = correction-self.raw(anchor)
            lifting = -xi+xi.square()/2
        else:
            lifting = -xi.square()/2
        return self.base+self.amplitude*(lifting+correction)
