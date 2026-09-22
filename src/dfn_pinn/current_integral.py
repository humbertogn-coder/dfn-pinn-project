"""Differentiable local interfacial charge from a smooth current model."""

import math
import torch
from torch import nn

from .constitutive import _check
from .projection import gauss_legendre


class CurrentIntegral(nn.Module):
    """Integrate j(x,t) from physical time zero to t, returning C/m2.

    current_model consumes (N,2) [x/L,t/t_ref], returns (N,1) j in A/m2
    active particle area, and must be pointwise (no batch coupling). Quadrature
    approximates the integral; endpoint derivative equality is not guaranteed
    for an underresolved current. No detach, clipping or interpolation is used.
    """

    def __init__(self, current_model, time_reference_s, order=32):
        super().__init__()
        if isinstance(time_reference_s, bool) or not math.isfinite(time_reference_s) or time_reference_s <= 0:
            raise ValueError("Time reference must be positive and finite")
        self.current_model = current_model
        nodes, weights = gauss_legendre(order, 0., 1.)
        self.register_buffer("nodes", nodes)
        self.register_buffer("weights", weights)
        self.register_buffer("time_reference_s", torch.tensor(time_reference_s, dtype=torch.float64))

    def forward(self, points):
        _check(points, "current integration points")
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            raise ValueError("Expected nonempty (N,2) [x/L,t/t_ref] points")
        if points.dtype != self.nodes.dtype or points.device != self.nodes.device:
            raise ValueError("Points and quadrature must share dtype and device")
        if not bool(((points[:, 0] >= 0) & (points[:, 0] <= 1)).all()) or not bool((points[:, 1] >= 0).all()):
            raise ValueError("Require x/L in [0,1] and normalized time >= 0")
        n, q = len(points), len(self.nodes)
        tau = points[:, 1:2]
        # Moving quadrature nodes retain derivatives of the integration limit.
        queries = torch.stack((points[:, :1].expand(n, q), tau*self.nodes), dim=-1).reshape(-1, 2)
        values = self.current_model(queries)
        _check(values, "interfacial current")
        if values.shape != (n*q, 1) or values.dtype != points.dtype or values.device != points.device:
            raise ValueError("Current model must return (N,1) with input dtype and device")
        charge = self.time_reference_s*tau*(values.reshape(n, q)*self.weights).sum(dim=1, keepdim=True)
        _check(charge, "cumulative interfacial charge")
        return charge
