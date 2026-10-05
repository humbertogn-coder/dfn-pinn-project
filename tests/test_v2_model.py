"""Fast checks of the v2 PINN building blocks (no PyBaMM solve, no training)."""

import numpy as np
import pytest
import torch

from dfn_pinn import constitutive as v1
from dfn_pinn.v2 import constitutive as cv
from dfn_pinn.v2.model import DFNPINN
from dfn_pinn.v2.params import CellParams, Protocol, Scales, FARADAY
from dfn_pinn.v2.residuals import Residuals, grad

DT = torch.float64


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = DFNPINN(Scales(CellParams(), Protocol())).to(DT)
    with torch.no_grad():  # make the networks non-trivial (physical parameters untouched)
        for name, p in m.named_parameters():
            if not name.startswith("log_mult"):
                p.add_(0.2 * torch.randn_like(p))
    return m


def test_constitutive_matches_v1_in_the_interior():
    theta = torch.linspace(0.05, 0.95, 37, dtype=DT)
    ce = torch.linspace(300.0, 2500.0, 37, dtype=DT)
    for k in ("n", "p"):
        torch.testing.assert_close(cv.ocp(theta, k), v1.ocp(theta, k))
        cmax = v1.MAX_CONCENTRATION[k]
        torch.testing.assert_close(cv.exchange_current(ce, theta, k, CellParams()),
                                   v1.exchange_current(ce, theta * cmax, torch.full_like(ce, 298.15), k),
                                   rtol=1e-9, atol=0)
    torch.testing.assert_close(cv.electrolyte_diffusivity(ce), v1.electrolyte_diffusivity(ce))
    torch.testing.assert_close(cv.electrolyte_conductivity(ce), v1.electrolyte_conductivity(ce))


def test_inverse_and_direct_bv_are_consistent():
    j0 = torch.tensor([0.3, 2.0, 5.0], dtype=DT)
    eta = torch.tensor([-0.05, 0.02, 0.12], dtype=DT)
    j = cv.bv_current(eta, j0, 298.15)
    torch.testing.assert_close(cv.bv_overpotential(j, j0, 298.15), eta)


def test_hard_projection_gives_exact_electrode_current(model):
    c = model.sc.cell
    t = torch.rand(5, 1, dtype=DT)
    x, w = np.polynomial.legendre.leggauss(80)
    xq = torch.tensor(0.5 * (x + 1), dtype=DT)
    wq = torch.tensor(0.5 * w, dtype=DT)
    for k, a, L, sign in (("n", c.a_n, c.L_n, 1), ("p", c.a_p, c.L_p, -1)):
        J = torch.stack([(model.j(k, xq.view(-1, 1), ti.expand(80, 1)).view(-1) * wq).sum() * a * L for ti in t])
        target = sign * model.sc.i_ref * model.g(t).view(-1)
        torch.testing.assert_close(J, target, rtol=1e-8, atol=1e-8)


def test_particle_hard_constraints(model):
    c = model.sc.cell
    n = 6
    xk, t = torch.rand(n, 1, dtype=DT), torch.rand(n, 1, dtype=DT)
    for k in ("n", "p"):
        R, D, cm = (c.R_n, c.D_n, c.cmax_n) if k == "n" else (c.R_p, c.D_p, c.cmax_p)
        s1 = torch.ones(n, 1, dtype=DT, requires_grad=True)
        j = model.j(k, xk, t)
        th = model.theta(k, s1, xk, t, j=j)
        flux = -D * cm / R * 2 * grad(th, s1)            # -D dc/dr at r = R
        torch.testing.assert_close(flux, j / FARADAY, rtol=1e-10, atol=1e-14)
        theta0 = c.theta_n0 if k == "n" else c.theta_p0
        th0 = model.theta(k, torch.rand(n, 1, dtype=DT), xk, torch.zeros(n, 1, dtype=DT))
        torch.testing.assert_close(th0, torch.full_like(th0, theta0))


def test_grouped_particle_residual_equals_reference(model):
    res = Residuals(model)
    P, nr = 4, 3
    s, xk, t = torch.rand(P, nr, dtype=DT), torch.rand(P, 1, dtype=DT), torch.rand(P, 1, dtype=DT)
    for k in ("n", "p"):
        rg = res.particle_grouped(k, s, xk.clone(), t.clone().requires_grad_(True))
        rr = res.particle(k, s.reshape(-1, 1).clone().requires_grad_(True),
                          xk.expand(P, nr).reshape(-1, 1).clone().requires_grad_(True),
                          t.expand(P, nr).reshape(-1, 1).clone().requires_grad_(True))
        torch.testing.assert_close(rg, rr, rtol=1e-10, atol=1e-10)


def test_electrolyte_continuity_by_construction(model):
    c = model.sc.cell
    t = torch.rand(7, 1, dtype=DT)
    one, zero = torch.ones_like(t), torch.zeros_like(t)
    for X, left, right in ((c.X1, (zero, zero), (one, zero)), (c.X2, (one, zero), (one, one))):
        Xi = torch.full_like(t, X)
        torch.testing.assert_close(model.ce(Xi, t, *left), model.ce(Xi, t, *right))
        torch.testing.assert_close(model.phie(Xi, t, *left), model.phie(Xi, t, *right))


def test_initial_and_gauge_conditions(model):
    t0 = torch.zeros(5, 1, dtype=DT)
    X = torch.rand(5, 1, dtype=DT)
    s1, s2 = (X > model.sc.cell.X1).to(DT), (X > model.sc.cell.X2).to(DT)
    torch.testing.assert_close(model.ce(X, t0, s1, s2), torch.ones_like(X))
    t = torch.rand(5, 1, dtype=DT)
    torch.testing.assert_close(model.phis("n", torch.zeros_like(t), t), torch.zeros_like(t))
    torch.testing.assert_close(model.phis("p", torch.ones_like(t), t), model.voltage(t))


def test_parameter_multipliers_enter_the_physics(model):
    """Doubling D_p must double the particle diffusion number and halve the parabola."""
    res = Residuals(model)
    P, nr = 3, 4
    s, xk, t = torch.rand(P, nr, dtype=DT), torch.rand(P, 1, dtype=DT), torch.rand(P, 1, dtype=DT)
    base = res.particle_grouped("p", s, xk.clone(), t.clone().requires_grad_(True)).detach()
    model.set_trainable_parameters(["D_p"], {"D_p": 2.0})
    assert model.log_mult["D_p"].requires_grad
    assert abs(model.parameter_values()["D_p"] - 2 * CellParams().D_p) < 1e-25
    changed = res.particle_grouped("p", s, xk.clone(), t.clone().requires_grad_(True)).detach()
    model.set_trainable_parameters([], {"D_p": 1.0})
    assert not torch.allclose(base, changed)
    again = res.particle_grouped("p", s, xk.clone(), t.clone().requires_grad_(True)).detach()
    torch.testing.assert_close(base, again)


def test_salt_inventory_is_zero_initially(model):
    res = Residuals(model)
    torch.testing.assert_close(res.salt_inventory(torch.zeros(3, 1, dtype=DT)), torch.zeros(3, 1, dtype=DT))


def test_soft_current_residual_is_zero_with_projection_and_nonzero_without():
    torch.manual_seed(1)
    t = torch.rand(4, 1, dtype=DT) * 0.9 + 0.05
    for projection, expect_zero in ((True, True), (False, False)):
        m = DFNPINN(Scales(CellParams(), Protocol()), projection=projection).to(DT)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if not name.startswith("log_mult"):
                    p.add_(0.3 * torch.randn_like(p))
        res = Residuals(m)
        for k in ("n", "p"):
            r = res.electrode_current(k, t)
            if expect_zero:
                torch.testing.assert_close(r, torch.zeros_like(r), atol=1e-10, rtol=0)
            else:
                assert r.abs().max() > 1e-3


def test_derived_inventory_identities():
    """inventory='derived': exact inventory, exact projection, j(x,0)=0, exact flux BC."""
    torch.manual_seed(2)
    cell = CellParams()
    m = DFNPINN(Scales(cell, Protocol()), inventory="derived").to(DT)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if not name.startswith("log_mult"):
                p.add_(0.2 * torch.randn_like(p))
    n = 5
    xk = torch.rand(n, 1, dtype=DT)
    t = (torch.rand(n, 1, dtype=DT) * 0.9 + 0.05).requires_grad_(True)
    x, w = np.polynomial.legendre.leggauss(60)
    xq, wq = torch.tensor(0.5 * (x + 1), dtype=DT), torch.tensor(0.5 * w, dtype=DT)
    for k, a, L, sign in (("n", cell.a_n, cell.L_n, 1.0), ("p", cell.a_p, cell.L_p, -1.0)):
        j, tb = m.current_and_mean(k, xk, t)
        torch.testing.assert_close(grad(tb, t), -m.inv_coef[k] * j)
        J = torch.stack([(m.j(k, xq.view(-1, 1), ti.view(1, 1).expand(60, 1).clone().requires_grad_(True)).view(-1) * wq).sum() * a * L
                         for ti in t.detach()])
        target = sign * m.sc.i_ref * m.g(t.detach()).view(-1)
        assert float(((J - target).abs() / m.sc.i_ref).max()) < 1e-3
        t0 = torch.zeros(n, 1, dtype=DT).requires_grad_(True)
        torch.testing.assert_close(m.j(k, xk, t0), torch.zeros(n, 1, dtype=DT))
        s1 = torch.ones(n, 1, dtype=DT, requires_grad=True)
        th = m.theta(k, s1, xk, t, j=j, tb=tb)
        R, D, cm = (cell.R_n, cell.D_n, cell.cmax_n) if k == "n" else (cell.R_p, cell.D_p, cell.cmax_p)
        torch.testing.assert_close(-D * cm / R * 2 * grad(th, s1), j / FARADAY, rtol=1e-10, atol=1e-14)


def test_hard_collector_bc_zero_slope():
    """collector_bc='hard': d c_e/dX = d phi_e/dX = 0 at both collectors for a random network."""
    from dfn_pinn.v2.model import DFNPINN
    from dfn_pinn.v2.params import CellParams, Protocol, Scales
    torch.manual_seed(3)
    m = DFNPINN(Scales(CellParams(), Protocol(5.0, 30.0, 3000.0)), width=16, depth=2, fourier_t=2,
                collector_bc="hard").to(torch.float64)
    t = torch.rand(5, 1, dtype=torch.float64)
    for X0, s1, s2 in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)):
        X = torch.full((5, 1), X0, dtype=torch.float64, requires_grad=True)
        side1, side2 = torch.full_like(X, s1), torch.full_like(X, s2)
        for f in (m.ce, m.phie):
            y = f(X, t, side1, side2)
            dy = torch.autograd.grad(y.sum(), X)[0]
            assert torch.allclose(dy, torch.zeros_like(dy), atol=1e-9), (X0, f.__name__, dy)
    # interior slope is not zero (the features are not degenerate)
    X = torch.full((5, 1), 0.3, dtype=torch.float64, requires_grad=True)
    y = m.ce(X, t, torch.zeros_like(X), torch.zeros_like(X))
    assert torch.autograd.grad(y.sum(), X)[0].abs().max() > 0
