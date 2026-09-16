"""Electrode-integrated current projection with explicit physical quadrature."""

import numpy as np
import torch

from .constitutive import _check


def gauss_legendre(order, left_m, right_m, *, dtype=torch.float64, device=None):
    """Return fixed electrode x nodes [m] and dx weights [m].

    Bounds are physical, not normalized neural coordinates. For network input
    use x/L but retain these physical weights. Geometry is not differentiable.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("Quadrature order must be a positive integer")
    if not np.isfinite([left_m, right_m]).all() or right_m <= left_m:
        raise ValueError("Quadrature bounds must be finite and increasing")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Quadrature requires float32 or float64")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = (right_m - left_m) / 2
    return (torch.as_tensor(left_m + half * (nodes + 1), dtype=dtype, device=device),
            torch.as_tensor(half * weights, dtype=dtype, device=device))


def _weighted_area(raw, area, weights):
    _check(raw, "raw interfacial current")
    _check(area, "active area density", 0)
    _check(weights, "physical dx weights", 0)
    if raw.ndim < 1 or raw.shape[-1] == 0:
        raise ValueError("Current must have a nonempty final quadrature axis")
    if weights.ndim != 1 or weights.shape[0] != raw.shape[-1]:
        raise ValueError("Weights must match the final current axis")
    if area.ndim != 0 and area.shape != weights.shape:
        raise ValueError("Area density must be scalar or one value per node")
    if any(x.dtype != raw.dtype or x.device != raw.device for x in (area, weights)):
        raise ValueError("Inputs must share dtype and device")
    aw = area * weights
    if not bool(torch.isfinite(aw).all()) or not bool(torch.isfinite(aw.sum())) or aw.sum() <= 0:
        raise ValueError("Integrated active area must be positive and finite")
    return aw


def integrated_current(raw, area, weights):
    """Integral a*j dx [A/m2 geometric area]; raw shape (..., quadrature)."""
    result = (raw * _weighted_area(raw, area, weights)).sum(dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Integrated current is nonfinite")
    return result


def current_correction(raw, area, weights, target):
    """Uniform additive correction [A/m2 active area], shape raw.shape[:-1].

    target is the SIGNED geometric current: +I/A for negative electrode,
    -I/A for positive electrode. It is scalar or has exactly the batch shape.
    Evaluate raw at fixed quadrature nodes at each time; reuse this correction
    at all other x query points at that SAME time. Do not detach the integral.
    """
    aw = _weighted_area(raw, area, weights)
    _check(target, "signed integrated-current target")
    if target.dtype != raw.dtype or target.device != raw.device:
        raise ValueError("Target must share current dtype and device")
    if target.ndim != 0 and target.shape != raw.shape[:-1]:
        raise ValueError("Target must be scalar or match the current batch shape")
    correction = (target - (raw * aw).sum(dim=-1)) / aw.sum()
    if not bool(torch.isfinite(correction).all()):
        raise FloatingPointError("Current correction is nonfinite")
    return correction


def project_current(raw, area, weights, target):
    """Project quadrature-node currents; preserves the autograd computation graph."""
    result = raw + current_correction(raw, area, weights, target).unsqueeze(-1)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Projected current is nonfinite")
    return result
