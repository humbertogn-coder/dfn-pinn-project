"""Focused analytical checks for gradient attribution and current sensitivity."""

import pytest
import torch

from dfn_pinn.dfn_gradient_diagnosis import collector_sensitivity, cosine, vector_gradient
from dfn_pinn.dfn_smoke import DFNSmoke


def test_analytic_gradient_sum_and_unused_parameter():
    a = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    b = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    first, second = a*a, -3*a
    g1, g2 = vector_gradient(first, [a, b]), vector_gradient(second, [a, b])
    torch.testing.assert_close(g1, torch.tensor([4., 0.], dtype=torch.float64))
    torch.testing.assert_close(g1+g2, vector_gradient(first+second, [a, b]))
    assert cosine(g1, g2) == pytest.approx(-1.)
    assert cosine(g1, g1*0) is None
    assert a.grad is None and b.grad is None


def test_nonfinite_gradient_is_rejected():
    a = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match='gradient'):
        vector_gradient(a.sqrt(), [a])


def test_solid_current_linear_slope_sensitivity():
    class LinearField(torch.nn.Module):
        amplitude = 1.
        time_factor = 1.
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(2, 1, dtype=torch.float64))
            with torch.no_grad():
                self.net[0].weight[:] = torch.tensor([[2., 0.]])
                self.net[0].bias.zero_()
        def forward(self, p):
            return self.net(p)
    model = DFNSmoke()
    model.phis[0], model.phis[1] = LinearField(), LinearField()
    result = collector_sensitivity(model, torch.tensor([.1, .5, 1.], dtype=torch.float64))
    for entry in result.values():
        coefficient = entry['current_coefficient_per_normalized_potential_slope']
        assert entry['normalized_current_min'] == pytest.approx(-2*coefficient)
        assert entry['current_jacobian_row_norm_rms'] == pytest.approx(coefficient)
        assert entry['required_normalized_potential_slope'] == pytest.approx(-1/coefficient)
