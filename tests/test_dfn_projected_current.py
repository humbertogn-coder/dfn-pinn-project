"""Focused projection wiring, independent integration and derivative checks."""

import io

import pytest
import torch

from dfn_pinn.dfn_projected_current import DFNProjectedCurrent, ProjectedCurrent
from dfn_pinn.projection import gauss_legendre
from dfn_pinn.spherical_diffusion import _gradient


class Polynomial(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))

    def forward(self, p):
        return self.a*p[:, :1]**2*(1+p[:, 1:2])+3*p[:, 1:2]**2


def test_derivatives_and_parameter_gradient_match_analytic_projection():
    raw = Polynomial()
    model = ProjectedCurrent(raw, 0., 1., 1., 2., 6., 4)
    p = torch.tensor([[.2, .1], [.7, .3]], dtype=torch.float64, requires_grad=True)
    value = model(p)
    expected = 3+raw.a*(p[:, :1]**2-1/3)*(1+p[:, 1:2])
    torch.testing.assert_close(value, expected, rtol=1e-13, atol=1e-13)
    gradient = _gradient(value, p)
    torch.testing.assert_close(gradient[:, :1], 2*raw.a*p[:, :1]*(1+p[:, 1:2]))
    torch.testing.assert_close(gradient[:, 1:2], raw.a*(p[:, :1]**2-1/3))
    torch.testing.assert_close(_gradient(gradient[:, :1], p)[:, :1], 2*raw.a*(1+p[:, 1:2]))
    got = torch.autograd.grad(value.sum(), raw.a)[0]
    torch.testing.assert_close(got, ((p[:, :1]**2-1/3)*(1+p[:, 1:2])).sum())


def test_batches_chunks_and_cross_query_jacobian():
    model = ProjectedCurrent(Polynomial(), 0., 1., 1., 2., -6., 8)
    p = torch.tensor([[.1, .2], [.8, .2], [.4, .7]], dtype=torch.float64, requires_grad=True)
    together = model(p)
    split = torch.cat([model(row[None]) for row in p])
    torch.testing.assert_close(together, split, rtol=1e-14, atol=1e-14)
    for i in range(len(p)):
        derivative = torch.autograd.grad(together[i].sum(), p, retain_graph=True)[0]
        assert torch.count_nonzero(derivative[[j for j in range(len(p)) if j != i]]) == 0


def test_both_electrodes_independent_integrals_and_shared_inventory():
    torch.manual_seed(42)
    model = DFNProjectedCurrent()
    for k, region in enumerate((0, 2)):
        left, right = model.bounds[region]
        x, w = gauss_legendre(64, left, right)
        length = sum(model.settings['lengths_m'])
        for tau in (0., .5/3600, 1/3600):
            p = torch.stack((x, torch.full_like(x, tau)), dim=1)
            j = model.reactions[k].integral.current_model(p).squeeze(1)
            integral = (j*w*length*model.active_area[k]).sum()
            expected = (1 if k == 0 else -1)*5/.1027
            assert float(integral.detach()) == pytest.approx(expected, rel=1e-12)
            charge = model.reactions[k].integral(p).squeeze(1)
            total_charge = (charge*w*length*model.active_area[k]).sum()
            assert float(total_charge.detach()) == pytest.approx(expected*tau*3600, abs=1e-11)


def test_underresolved_projection_is_visible_to_independent_quadrature():
    class Sixth(torch.nn.Module):
        def forward(self, p):
            return p[:, :1]**6+0*p[:, 1:2]
    model = ProjectedCurrent(Sixth(), 0., 1., 1., 1., 1., 2)
    x, w = gauss_legendre(16, 0., 1.)
    value = (model(torch.stack((x, x*0), dim=1)).squeeze(1)*w).sum()
    assert abs(float(value)-1) > 1e-3


def test_full_residual_backward_and_exact_checkpoint_replay():
    torch.manual_seed(42)
    model = DFNProjectedCurrent(projection_order=8)
    samples = model.sample()
    residuals, diagnostics = model.residuals(samples)
    assert len(residuals) == 34 and len(diagnostics) == 4
    loss = sum(v.square().mean() for v in residuals.values())
    loss.backward()
    for name in ('ce', 'phie', 'phis', 'cs', 'reactions'):
        for branch in getattr(model, name):
            gradients = [p.grad for p in branch.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert sum(g.square().sum() for g in gradients) > 0
    buffer = io.BytesIO()
    torch.save({'model': model.state_dict(), 'variant': model.variant_spec}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    replay = DFNProjectedCurrent(projection_order=saved['variant']['projection_order'])
    replay.load_state_dict(saved['model'])
    for name, value in model.snapshot(samples).items():
        torch.testing.assert_close(value, replay.snapshot(samples)[name], rtol=0, atol=0)
