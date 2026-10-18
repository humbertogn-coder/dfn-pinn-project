"""Rate-parametric Li-SPAN PINN (src/dfn_pinn/lispan/rate_pinn.py): window fit, hard structure, residuals, training I/O."""

from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.lispan import LiSPANParams  # noqa: E402
from dfn_pinn.lispan.rate_pinn import (LiSPANRatePINN, RateSampler, RateTrainConfig, fit_window, load_rate_model,  # noqa: E402
                                       rate_residuals, train_rate, window_t_end)

# FV cut-off times of the nominal cell (Z_CC 0.025, reactions 2-3 irreversible, 30 s ramp), results/lispan/pinn_reference_*
T_CUT = {0.1: 34097.83, 0.2: 16669.42, 0.5: 6268.80, 1.0: 2837.83}
PARAMS = LiSPANParams(Z_CC=0.025, reversible=(True, False, False))


def small_model(dtype=torch.float64):
    coef, _ = fit_window(list(T_CUT), list(T_CUT.values()))
    torch.manual_seed(0)
    return LiSPANRatePINN(PARAMS, coef, 0.1, 1.0, width=16, depth=2, fourier_t=2, quad_order=8).to(dtype)


def random_points(n, seed=1, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    Y = torch.rand(n, 1, generator=g, dtype=dtype)
    T = torch.rand(n, 1, generator=g, dtype=dtype)
    r = torch.exp(math.log(0.1) + math.log(10.0) * torch.rand(n, 1, generator=g, dtype=dtype))
    return Y, T, r


def test_window_fit_and_t_end():
    coef, err = fit_window(list(T_CUT), list(T_CUT.values()))
    assert err < 0.01
    for rate, t in T_CUT.items():
        assert abs(window_t_end(coef, rate, 1.0) / t - 1) < 0.01
    m = small_model()
    r = torch.tensor([[0.3]], dtype=torch.float64)
    assert abs(float(m.t_end(r)) / window_t_end(coef, 0.3, 0.98) - 1) < 1e-12
    assert float(m.rate_feature(torch.tensor([[0.1]], dtype=torch.float64))) == pytest.approx(-1.0)
    assert float(m.rate_feature(torch.tensor([[1.0]], dtype=torch.float64))) == pytest.approx(1.0)


def test_hard_structure():
    m = small_model()
    Y, T, r = random_points(16)
    # extents: zero at T = 0, ordered xi1 >= xi2 >= xi3 >= 0 and xi1 <= 1
    with torch.no_grad():
        xi0 = m.extents(Y, torch.zeros_like(T), r)
    assert all(float(x.abs().max()) == 0.0 for x in xi0)
    xi1, xi2, xi3 = m.extents(Y, T, r)
    assert bool(((xi1 >= xi2) & (xi2 >= xi3) & (xi3 >= 0) & (xi1 <= 1)).all())
    # current projection: i_e = 0 at the collector and -I at the separator side (exact, same quadrature)
    with torch.no_grad():
        _, ie0 = m.current_fields(torch.zeros_like(Y), T, r)
        _, ie1 = m.current_fields(torch.ones_like(Y), T, r)
    assert float(ie0.abs().max()) < 1e-12
    torch.testing.assert_close(ie1, -m.I(T, r), rtol=1e-10, atol=1e-12)
    # charge passed: dQ/dT = t_end(r) I(T, r); ramp: I(0) = 0, I -> I0 = 10 r A/m2 at the end of the window
    Tg = T.clone().requires_grad_(True)
    dQ = torch.autograd.grad(m.charge_passed(Tg, r).sum(), Tg)[0]
    torch.testing.assert_close(dQ, m.t_end(r) * m.I(T, r), rtol=1e-8, atol=1e-8)
    assert float(m.I(torch.zeros_like(T), r).abs().max()) == 0.0
    torch.testing.assert_close(m.I(torch.ones_like(T), r), 10.0 * r, rtol=1e-6, atol=0.0)


def test_residuals_finite_and_shapes():
    m = small_model()
    m.salt_total_scale, m.charge_total_scale = 1.0, 1.0
    cfg = RateTrainConfig(n_cathode=32, n_separator=16, n_boundary=8)
    batch = RateSampler(cfg, torch.float64, torch.Generator().manual_seed(2), torch.device("cpu")).draw()
    rates = torch.cat([batch["c"][2], batch["s"][2], batch["b"][1]])
    assert float(rates.min()) >= 0.1 * (1 - 1e-12) and float(rates.max()) <= 1.0 * (1 + 1e-12)
    out = rate_residuals(m, batch)
    expected = {"ode1", "ode2", "ode3", "charge_s", "charge_e", "salt_c", "salt_s", "bc_c0", "bc_an", "bc_if",
                "salt_total", "charge_total"}
    assert set(out) == expected
    assert out["ode1"].shape == (32, 1) and out["charge_e"].shape == (48, 1) and out["bc_an"].shape == (8, 1)
    assert all(bool(torch.isfinite(v).all()) for v in out.values())


def test_discharge_curve_refuses_extrapolation():
    m = small_model()
    curve = m.discharge_curve(0.3, n=11)
    assert curve["t"][0] == 0.0 and abs(curve["t"][-1] - curve["t_end"]) < 1e-9 and np.isfinite(curve["V"]).all()
    assert curve["capacity_mAh_gS"][-1] > 0
    with pytest.raises(ValueError):
        m.discharge_curve(2.0)


def test_train_save_load_resume(tmp_path):
    cfg = RateTrainConfig(width=16, depth=2, fourier_t=2, quad_order=8, n_cathode=32, n_separator=16, n_boundary=8,
                          adam_steps=4, log_every=2, eval_every=1000, latest_every=2, checkpoint_every=0, threads=1,
                          charge_total_scale=1.0, charge_total_start=0, salt_sep_natural_scale=True)
    coef, _ = fit_window(list(T_CUT), list(T_CUT.values()))
    model, history, _ = train_rate(cfg, PARAMS, coef, tmp_path, refs=None, log=lambda msg: None)
    assert (tmp_path / "final.pt").exists() and history[-1]["step"] == 4
    loaded, cfg2, params2 = load_rate_model(tmp_path / "final.pt")
    assert cfg2.width == 16 and params2.Z_CC == PARAMS.Z_CC and params2.reversible == PARAMS.reversible
    np.testing.assert_allclose(model.discharge_curve(0.3, n=20)["V"], loaded.discharge_curve(0.3, n=20)["V"], rtol=0, atol=1e-6)
    # resume from latest.pt (step 4) and continue to step 6
    _, history2, _ = train_rate(replace(cfg, adam_steps=6), PARAMS, coef, tmp_path, refs=None, log=lambda msg: None,
                                resume=tmp_path / "latest.pt")
    assert history2[-1]["step"] == 6
    assert torch.load(tmp_path / "final.pt", weights_only=False)["step"] == 6


def test_matches_single_rate_model_at_fixed_rate():
    """With the rate-feature weights set to zero and the other weights copied from a single-rate LiSPANPINN built for
    the same current and window, every residual and the voltage equal the single-rate ones (same physics, per-point
    scales reduce to the protocol scales)."""
    from dfn_pinn.lispan import LiSPANProtocol
    from dfn_pinn.lispan.pinn import LiSPANPINN, residuals
    rm = LiSPANRatePINN(PARAMS, fit_window(list(T_CUT), list(T_CUT.values()))[0], 0.1, 1.0, width=16, depth=2,
                        fourier_t=2, quad_order=8, salt_sep_natural_scale=True).double()
    g = torch.Generator().manual_seed(3)

    def U(n):
        return torch.rand(n, 1, generator=g, dtype=torch.float64)
    for r0 in (0.1, 0.37, 1.0):
        te = float(rm.t_end(torch.tensor([[r0]], dtype=torch.float64)))
        bm = LiSPANPINN(PARAMS, LiSPANProtocol(current=10 * r0, ramp_s=30.0), te, width=16, depth=2, fourier_t=2,
                        quad_order=8).double()
        bm.salt_scale_sep = bm.salt_scale * PARAMS.L_cat / PARAMS.L_sep
        with torch.no_grad():
            for name in ("net_R", "net_xi", "net_ce", "net_phie"):
                for i, (a, b) in enumerate(zip(getattr(bm, name).net, getattr(rm, name).net)):
                    if isinstance(a, torch.nn.Linear):
                        if i == 0:
                            b.weight.zero_(); b.weight[:, :-1].copy_(a.weight)
                        else:
                            b.weight.copy_(a.weight)
                        b.bias.copy_(a.bias)
        for m in (rm, bm):
            m.salt_total_scale, m.charge_total_scale = 1.0, 1.0
        Yc, T, Ys, Ts, Tb = U(12), U(12), U(8), U(8), U(5)

        def full(x):
            return torch.full_like(x, r0)
        out_r = rate_residuals(rm, {"c": (Yc, T, full(T)), "s": (Ys, Ts, full(Ts)), "b": (Tb, full(Tb))})
        out_b = residuals(bm, {"c": (Yc, T), "s": (Ys, Ts), "b": Tb})
        for k in out_b:
            torch.testing.assert_close(out_r[k].detach(), out_b[k].detach(), rtol=1e-10, atol=1e-12)
        with torch.no_grad():
            torch.testing.assert_close(rm.voltage(T, full(T)), bm.voltage(T), rtol=1e-12, atol=1e-12)
