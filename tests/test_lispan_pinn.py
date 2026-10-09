"""Unit checks of the Li-SPAN PINN building blocks (no training)."""

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.lispan.params import F, LiSPANParams, LiSPANProtocol  # noqa: E402
from dfn_pinn.lispan.pinn import LiSPANPINN, Sampler, LiSPANTrainConfig, residuals  # noqa: E402


def _model(**kw):
    p = LiSPANParams(reversible=(True, False, False))
    prot = LiSPANProtocol.from_crate(0.1, ramp_s=30.0)
    torch.manual_seed(0)
    return LiSPANPINN(p, prot, t_end=30000.0, width=16, depth=2, quad_order=8, **kw).double(), p, prot


def test_charge_passed_matches_quadrature():
    m, p, prot = _model()
    t = np.linspace(0.0, m.t_end, 200001)
    I = prot.current * np.tanh(t / prot.ramp_s)
    Q_num = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))])
    T = torch.tensor([[0.0], [1e-4], [0.01], [0.5], [1.0]], dtype=torch.float64)
    Q = m.charge_passed(T).view(-1).numpy()
    Q_ref = np.interp(T.view(-1).numpy() * m.t_end, t, Q_num)
    assert np.allclose(Q, Q_ref, rtol=1e-6, atol=1e-6)


def test_extent_inventory_is_charge():
    """Full conversion of the three reactions carries the theoretical capacity 6 F c_S4,0 L_cat."""
    m, p, prot = _model()
    assert abs(2.0 * F * p.c_init[0] * p.L_cat * 3.0 - p.Q_theo) < 1e-9 * p.Q_theo


def test_charge_total_residual_present_and_scaled():
    m, p, prot = _model()
    m.charge_total_scale = 2.0
    cfg = LiSPANTrainConfig(n_cathode=8, n_separator=4, n_boundary=5)
    batch = Sampler(cfg, torch.float64, torch.Generator().manual_seed(1)).draw()
    out = residuals(m, batch)
    assert "charge_total" in out and out["charge_total"].shape == (5, 1)
    # residual = (c40 int sum xi dY - Q/(2 F L)) / scale, checked against an independent evaluation
    Tb = batch["b"]
    q, w = m.q_nodes, m.q_weights
    vals = []
    for tb in Tb.view(-1):
        Y = q.view(-1, 1)
        T = torch.full_like(Y, float(tb))
        xi = m.extents(Y, T)
        inv = p.c_init[0] * float(((xi[0] + xi[1] + xi[2]).view(-1) * w).sum())
        vals.append((inv - float(m.charge_passed(tb.view(1, 1))) / (2 * F * p.L_cat)) / 2.0)
    assert np.allclose(out["charge_total"].detach().view(-1).numpy(), vals, rtol=1e-8, atol=1e-8)


def test_parameter_values_roundtrip():
    m, p, prot = _model()
    m.set_parameter_values({"k0_2": 0.5, "U0_1": 0.012, "Z_CC": 1.3})
    v = m.parameter_values()
    assert abs(v["k0_2"] - 0.5) < 1e-12 and abs(v["U0_1"] - 0.012) < 1e-12 and abs(v["Z_CC"] - 1.3) < 1e-12
    assert abs(float(m.k0(1)) - 0.5 * p.k0[1]) < 1e-20
    assert abs(float(m.U0(0)) - (p.U0[0] + 0.012)) < 1e-12


def test_runs_load_model_roundtrip(tmp_path):
    """dfn_pinn.lispan.runs.load_model rebuilds a checkpoint written in the layout of pinn.train (paper scripts)."""
    from dfn_pinn.lispan import runs
    cfg = LiSPANTrainConfig(width=16, depth=2, quad_order=8, fourier_t=4)
    p = LiSPANParams(reversible=(True, False, False))
    prot = LiSPANProtocol.from_crate(0.1, ramp_s=30.0)
    torch.manual_seed(1)
    m = LiSPANPINN(p, prot, 30000.0, cfg.width, cfg.depth, cfg.act, cfg.fourier_t, cfg.fourier_period,
                   tuple(cfg.short_t), cfg.ic_tau_s, cfg.quad_order)
    m.set_parameter_values({"k0_2": 1.7, "U0_1": 0.012})
    torch.save({"model": m.state_dict(), "train": cfg.to_dict(), "params": p.to_dict(), "protocol": prot.to_dict(),
                "t_end": 30000.0}, tmp_path / "final.pt")
    r = runs.load_model(tmp_path / "final.pt")
    T = torch.linspace(0.0, 1.0, 7).view(-1, 1)
    with torch.no_grad():
        assert torch.allclose(m.voltage(T), r.voltage(T), atol=1e-6)
    pv = r.parameter_values()
    assert abs(pv["k0_2"] - 1.7) < 1e-5 and abs(pv["U0_1"] - 0.012) < 1e-6


def test_fdjac_absolute_steps_and_bounds():
    """least_squares_fd: forward differences with absolute steps (also at x = 0) and a backward step at an upper bound."""
    from dfn_pinn.lispan.fdjac import least_squares_fd
    A = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.0]])
    calls = []

    def resid(x):
        calls.append(np.array(x))
        return A @ x + 0.1 * x[0] ** 2 * np.ones(3)          # d r / d x0 = A[:, 0] + 0.2 x0

    fun, jac = least_squares_fd(resid, [1e-3, 1e-3], lo=np.array([-1.0, -1.0]), hi=np.array([1.0, 1.0]))
    fun(np.zeros(2))
    J = jac(np.zeros(2))
    assert np.allclose(J, A, atol=1e-3) and len(calls) == 3   # residual at x reused, one solve per column
    xb = np.array([1.0, 0.0])                                  # at the upper bound: backward difference
    fun(xb)
    Jb = jac(xb)
    assert calls[-2][0] < 1.0                                  # the x0 column stepped down, not outside the bound
    assert np.allclose(Jb[:, 0], A[:, 0] + 0.2, atol=2e-3)


def test_protocol_dict_roundtrip():
    """Run folders store the protocol with 'current_A_m2', config files with 'current': both must give the same
    protocol, and a dict with neither key must fail instead of falling back to the default (0.1 C) current."""
    import pytest
    prot = LiSPANProtocol(current=10.0, ramp_s=30.0)
    for d in (prot.to_dict(), {"current": 10.0, "ramp_s": 30.0}):
        q = LiSPANProtocol.from_dict(d)
        assert q.current == 10.0 and q.ramp_s == 30.0 and q.crate == 1.0
    with pytest.raises(ValueError):
        LiSPANProtocol.from_dict({"ramp_s": 30.0})
