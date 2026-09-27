"""Direct-kinetics DFN variant with a shared electrode-current projection."""

import torch
from torch import nn

from .dfn_smoke import DFNSmoke
from .projection import current_correction, gauss_legendre


class ProjectedCurrent(nn.Module):
    """Pointwise queries with a fixed, differentiable spatial integral per time.

    Projection enforces the discrete electrode source integral, not local
    phase-current conservation or pointwise current positivity.
    """

    def __init__(self, raw, left_m, right_m, length_m, active_area, target, order=32):
        super().__init__()
        self.raw = raw
        nodes, weights = gauss_legendre(order, left_m, right_m)
        self.register_buffer('nodes', nodes/length_m)
        self.register_buffer('weights', weights)
        self.register_buffer('area', torch.tensor(active_area, dtype=torch.float64))
        self.register_buffer('target', torch.tensor(target, dtype=torch.float64))

    def forward(self, p):
        if p.ndim != 2 or p.shape[1] != 2 or len(p) == 0:
            raise ValueError('Expected nonempty (N,2) current queries')
        q = len(self.nodes)
        # Each row gets its own fixed quadrature; never integrate query-batch x.
        nodes = self.nodes.expand(len(p), q)
        times = p[:, 1:2].expand(len(p), q)
        fixed = torch.stack((nodes, times), dim=-1).reshape(-1, 2)
        raw_nodes = self.raw(fixed).reshape(len(p), q)
        correction = current_correction(raw_nodes, self.area, self.weights, self.target)
        return self.raw(p)+correction[:, None]


class DFNProjectedCurrent(DFNSmoke):
    """Variant F building block: same 34 terms and direct Butler-Volmer.

    Architecture metadata must be serialized separately from physical settings.
    Historical DFNSmoke loaders must not load this class implicitly.
    """

    def __init__(self, settings=None, projection_order=32):
        super().__init__(settings)
        c = self.settings
        length = sum(c['lengths_m'])
        self.variant_spec = {'name': 'direct_bv_hard_current', 'projection_order': projection_order,
                             'projection': 'uniform_additive_fixed_physical_quadrature'}
        for k, i in enumerate((0, 2)):
            left, right = self.bounds[i]
            raw = self.reactions[k].integral.current_model
            target = (1 if k == 0 else -1)*c['applied_current_A']/c['area_m2']
            self.reactions[k].integral.current_model = ProjectedCurrent(
                raw, left*length, right*length, length, self.active_area[k], target, projection_order)
