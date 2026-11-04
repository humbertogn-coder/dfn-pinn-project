"""Network v5 (geometry): geometry helpers, sampler domain, per-row scaling and the hard inventory with geometry rows."""

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from dfn_pinn.lispan.chem_pinn import n_window_features
from dfn_pinn.lispan.geo_pinn import (GeoSampler, GeoTrainConfig, build_geo_model, geo_residuals, network_params)
from dfn_pinn.lispan.geometry import (cell_params, f_from_porosity, in_domain, loading_mg_cm2, porosity,
                                      thickness_from_loading, transport_load)
from dfn_pinn.lispan.params import LiSPANParams
from dfn_pinn.lispan.rate_pinn import ONE_C

ROOT = Path(__file__).resolve().parents[1]


def _model(dtype=torch.float64):
    d3 = json.loads((ROOT / "configs" / "lispan_chem_design_v3.json").read_text())
    win = {"coef_base": d3["window"]["coef"], "degree_base": d3["window"]["degree"],
           "coef": [0.0] * n_window_features(7, 2), "degree": 2}
    p = network_params(replace(LiSPANParams(), c_DL=1e-6, Z_CC=0.0, reversible=(True, False, False)))
    cfg = GeoTrainConfig(inventory_hard=True, dtype="float64" if dtype == torch.float64 else "float32")
    torch.manual_seed(0)
    return build_geo_model(cfg, p, win).to(dtype), cfg


def test_geometry_helpers():
    p = LiSPANParams()
    q = cell_params(p, 50.0, 3.0)
    assert q.m_S_model == pytest.approx(p.m_S_model * 1.5)
    assert loading_mg_cm2(p, 100.0, 1.0) == pytest.approx(p.m_S_model * 100)
    assert porosity(p, 1.0) == pytest.approx(p.eps_e0)
    assert q.eps_e0 == pytest.approx(porosity(p, 3.0))
    assert f_from_porosity(p, porosity(p, 2.5)) == pytest.approx(2.5)
    assert thickness_from_loading(p, loading_mg_cm2(p, 70.0, 4.0), 4.0) == pytest.approx(70.0)
    assert q.c_ref[0] / q.c_init[0] == pytest.approx(p.c_ref[0] / p.c_init[0])       # activities unchanged
    assert in_domain(p, 2.0, 65.0, 4.0) and not in_domain(p, 3.0, 150.0, 4.0) and not in_domain(p, 0.1, 200.0, 5.0)
    assert transport_load(p, 1.0, 100.0, 1.0) == pytest.approx(10.0 * 1e-4 / porosity(p, 1.0) ** 1.5)


def test_sampler_stays_in_domain():
    _, cfg = _model()
    s = GeoSampler(cfg, torch.float64, torch.Generator().manual_seed(3), "cpu")
    z = s.conds(4000)
    p = LiSPANParams()
    assert z.shape == (4000, 8)
    assert float(z[:, 0].min()) >= cfg.rate_min - 1e-12 and float(z[:, 0].max()) <= cfg.rate_max + 1e-12
    assert all(in_domain(p, float(r), math.exp(float(lnL)), float(f)) for r, lnL, f in z[:, [0, 6, 7]].tolist())


def test_current_and_inventory_with_geometry_rows():
    m, cfg = _model()
    s = GeoSampler(cfg, torch.float64, torch.Generator().manual_seed(1), "cpu")
    z = s.conds(64)
    assert torch.allclose(m.I0(z), ONE_C * z[:, :1] * z[:, 7:8] * torch.exp(z[:, 6:7]) / 100.0)
    T = torch.rand(64, 1, dtype=torch.float64) * 0.9 + 0.05
    q = m.q_nodes.shape[0]
    Yq = m.q_nodes.view(1, q).expand(64, q).reshape(-1, 1)
    xi = m.extents(Yq, T.view(64, 1).expand(64, q).reshape(-1, 1), z.view(64, 1, 8).expand(64, q, 8).reshape(-1, 8))
    inv = (sum(xi).view(64, q) * m.q_weights.view(1, q)).sum(1, keepdim=True)
    # 2 F L f c_S4,0 int sum xi dY = Q(t): exact by construction for every geometry row
    Q = 2.0 * 96485.33212 * m.L_cat(z) * m.f_span(z) * m.p.c_init[0] * inv
    assert torch.allclose(Q, m.charge_passed(T, z), rtol=1e-8)


def test_chemistry_per_chain_does_not_depend_on_f():
    """With the geometry columns of the networks zeroed, rows that differ only in f (same C-rate) have the same extents,
    the same reaction rate per SPAN area and the same extent residuals at early times (before Li2S changes eps_e)."""
    m, cfg = _model()
    with torch.no_grad():
        for net in (m.net_R, m.net_xi, m.net_ce, m.net_phie):
            w = net.net[0].weight
            w[:, -2:] = 0.0                     # the last two condition features are the geometry
    z = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, math.log(60.0), 1.0],
                      [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, math.log(60.0), 3.0]], dtype=torch.float64)
    Yc = torch.full((2, 1), 0.3, dtype=torch.float64)
    T = torch.full((2, 1), 0.01, dtype=torch.float64)
    sp = m.species(Yc, T, z)
    assert torch.allclose(sp["xi"][0][0], sp["xi"][0][1])
    assert float(sp["a"][1] / sp["a"][0]) == pytest.approx(3.0, rel=1e-3)
    Rr, _ = m.current_fields(Yc, T, z, sp)
    assert float(Rr[1] / Rr[0]) == pytest.approx(1.0, rel=1e-3)
    batch = {"c": (Yc, T, z), "s": (Yc, T, z), "b": (T, z)}
    res = geo_residuals(m, batch)
    assert torch.allclose(res["ode1"][0], res["ode1"][1], rtol=1e-3, atol=1e-9)
    assert all(torch.isfinite(v).all() for v in res.values())


def test_residuals_and_gradients_finite():
    m, cfg = _model(torch.float32)
    s = GeoSampler(cfg, torch.float32, torch.Generator().manual_seed(2), "cpu")
    b = s.draw()
    b = {"c": tuple(x[:256] for x in b["c"]), "s": tuple(x[:128] for x in b["s"]), "b": tuple(x[:32] for x in b["b"])}
    res = geo_residuals(m, b)
    loss = sum(v.square().mean() for v in res.values())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None)
