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
