"""Quadrature inventory projection for the synthetic constant-unit-flux pilot."""

import torch
from torch import nn

from .constitutive import _check
from .projection import gauss_legendre


class UnitFluxInventoryProjection(nn.Module):
    """Preserve radial derivatives while enforcing mean concentration 1 - 3*t.

    The wrapped model must be pointwise and return (N,1) concentrations for
    (rho,x,t) inputs. The target is specific to the dimensionless unit sphere,
    unit diffusivity, unit outward flux and initial concentration one.
    This is NOT a general DFN inventory model. Conservation is discrete at
    the registered quadrature; independent integration must still be audited.
    """

    def __init__(self, raw_model, order=64):
        super().__init__()
        if isinstance(order, bool) or not isinstance(order, int) or order < 2:
            raise ValueError("Inventory projection requires integer order >= 2")
        self.raw_model = raw_model
        nodes, weights = gauss_legendre(order, 0., 1.)
        measure = 3*nodes.square()*weights
        self.register_buffer("nodes", nodes)
        self.register_buffer("measure", measure/measure.sum())

    def forward(self, points):
        _check(points, "particle points")
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("Expected nonempty (N,3) points")
        if points.dtype != self.nodes.dtype or points.device != self.nodes.device:
            raise ValueError("Projection buffers and points must share dtype and device")
        if not bool(((points[:, :2] >= 0) & (points[:, :2] <= 1)).all()) or not bool((points[:, 2] >= 0).all()):
            raise ValueError("Require rho,x in [0,1] and t >= 0")
        n, q = len(points), len(self.nodes)
        # Each query carries its own x,t graph. Do not detach or deduplicate time.
        radial_points = torch.stack((self.nodes.expand(n, q),
                                     points[:, 1:2].expand(n, q),
                                     points[:, 2:3].expand(n, q)), dim=-1).reshape(-1, 3)
        raw = self.raw_model(points)
        quadrature = self.raw_model(radial_points)
        for value, shape in ((raw, (n, 1)), (quadrature, (n*q, 1))):
            _check(value, "raw concentration")
            if value.shape != shape or value.dtype != points.dtype or value.device != points.device:
                raise ValueError("Raw model must return (N,1) with input dtype and device")
        mean = (quadrature.reshape(n, q)*self.measure).sum(dim=1, keepdim=True)
        return raw-mean+(1-3*points[:, 2:3])
