"""Chemistry-conditioned Li-SPAN PINN (src/dfn_pinn/lispan/chem_pinn.py, Phase 3b)."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.lispan import LiSPANParams  # noqa: E402
from dfn_pinn.lispan.chem_pinn import (CHEM_INPUTS_V1, ChemSampler, ChemTrainConfig, LiSPANChemPINN, apply_chem,  # noqa: E402
                                       build_chem_model, check_spec, chem_eval_summary, evaluate_chem, fit_chem_window,
                                       load_chem_model, n_window_features, normalize, window_features)
from dfn_pinn.lispan.rate_pinn import LiSPANRatePINN, fit_window, rate_residuals, train_rate  # noqa: E402

T_CUT = {0.1: 34097.83, 0.2: 16669.42, 0.5: 6268.80, 1.0: 2837.83}
PARAMS = LiSPANParams(Z_CC=0.025, reversible=(True, False, False))
SPEC = check_spec(CHEM_INPUTS_V1)
K = len(SPEC)


def embed_rate_window(coef3, k=K):
    """Chem window coefficients that reproduce the rate window (no chemistry dependence)."""
    c = np.zeros(n_window_features(k))
    c[0], c[1], c[2 + k] = coef3[0], coef3[1], coef3[2]
    return c.tolist()


def paired_models():
    """A rate model and a chem model with the same weights (chemistry inputs disconnected)."""
    coef3, _ = fit_window(list(T_CUT), list(T_CUT.values()))
    rm = LiSPANRatePINN(PARAMS, coef3, 0.1, 1.0, width=16, depth=2, fourier_t=2, quad_order=8,
                        salt_sep_natural_scale=True).double()
    cm = LiSPANChemPINN(PARAMS, embed_rate_window(coef3), CHEM_INPUTS_V1, 0.1, 1.0, width=16, depth=2, fourier_t=2,
                        quad_order=8, salt_sep_natural_scale=True).double()
    with torch.no_grad():
        for name in ("net_R", "net_xi", "net_ce", "net_phie"):
            for i, (a, b) in enumerate(zip(getattr(rm, name).net, getattr(cm, name).net)):
                if isinstance(a, torch.nn.Linear):
                    if i == 0:
                        b.weight.zero_(); b.weight[:, :a.weight.shape[1]].copy_(a.weight)
                    else:
                        b.weight.copy_(a.weight)
                    b.bias.copy_(a.bias)
    for m in (rm, cm):
        m.salt_total_scale, m.charge_total_scale = 1.0, 1.0
    return rm, cm


def test_spec_checks():
    with pytest.raises(ValueError):
        check_spec([{"name": "D_salt", "lo": -1, "hi": 1}])
    with pytest.raises(ValueError):
        check_spec([{"name": "dU0_2", "lo": 0.01, "hi": 0.1}])          # nominal outside
    with pytest.raises(ValueError):
        check_spec([{"name": "b", "lo": -1, "hi": 1}, {"name": "b_2", "lo": -1, "hi": 1}])
    assert np.allclose(normalize(SPEC, [s["lo"] for s in SPEC]), -1) and np.allclose(normalize(SPEC, [s["hi"] for s in SPEC]), 1)


def test_window_fit_recovers_a_quadratic():
    rng = np.random.default_rng(0)
    coef = rng.normal(0, 0.1, n_window_features(K)); coef[0] = 8.0
    rates = np.exp(rng.uniform(np.log(0.1), 0.0, 60))
    th = rng.uniform(-1, 1, (60, K))
    t_cut = np.exp(window_features(np.log(rates), th) @ coef) / rates
    fit, err_max, err_rms = fit_chem_window(rates, th, t_cut)
    assert np.allclose(fit, coef, atol=1e-8) and err_max < 1e-10


def test_apply_chem_maps_inputs_to_params():
    p = apply_chem(PARAMS, SPEC, [0.05, -0.03, math.log(1.2), math.log(2.0)])
    assert p.U0 == pytest.approx((PARAMS.U0[0], PARAMS.U0[1] + 0.05, PARAMS.U0[2] - 0.03))
    assert p.b == pytest.approx(tuple(1.2 * b for b in PARAMS.b))
    assert p.k0 == pytest.approx((2.0 * PARAMS.k0[0], PARAMS.k0[1], PARAMS.k0[2]))


def test_nominal_chemistry_equals_rate_model():
    """Chemistry inputs disconnected and at their nominal values: residuals and voltage equal the rate model's."""
    rm, cm = paired_models()
    g = torch.Generator().manual_seed(4)

    def U(n):
        return torch.rand(n, 1, generator=g, dtype=torch.float64)
    for r0 in (0.1, 0.37, 1.0):
        Yc, T, Ys, Ts, Tb = U(10), U(10), U(6), U(6), U(4)

        def rr(x):
            return torch.full_like(x, r0)

        def zz(x):
            return torch.cat([rr(x), torch.zeros(x.shape[0], K, dtype=x.dtype)], dim=1)
        out_r = rate_residuals(rm, {"c": (Yc, T, rr(T)), "s": (Ys, Ts, rr(Ts)), "b": (Tb, rr(Tb))})
        out_c = rate_residuals(cm, {"c": (Yc, T, zz(T)), "s": (Ys, Ts, zz(Ts)), "b": (Tb, zz(Tb))})
        for k in out_r:
            torch.testing.assert_close(out_c[k].detach(), out_r[k].detach(), rtol=1e-10, atol=1e-12)
        with torch.no_grad():
            torch.testing.assert_close(cm.voltage(T, zz(T)), rm.voltage(T, rr(T)), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(cm.t_end(zz(T)), rm.t_end(rr(T)), rtol=1e-12, atol=0)


def test_chemistry_inputs_act_like_parameter_multipliers():
    """Per-point chemistry (dU0_2, b, k0_1) gives the residuals of the rate model with the same parameter values set
    as its (scalar) multipliers: the plumbing of the per-point kinetics is exact."""
    rm, cm = paired_models()
    vals = [0.05, -0.04, math.log(1.3), math.log(3.0)]
    rm.set_parameter_values({"U0_2": 0.05, "U0_3": -0.04, "b_1": 1.3, "b_2": 1.3, "b_3": 1.3, "k0_1": 3.0})
    g = torch.Generator().manual_seed(5)
    Yc, T = torch.rand(10, 1, generator=g, dtype=torch.float64), torch.rand(10, 1, generator=g, dtype=torch.float64)
    Ys, Ts = torch.rand(6, 1, generator=g, dtype=torch.float64), torch.rand(6, 1, generator=g, dtype=torch.float64)
    Tb = torch.rand(4, 1, generator=g, dtype=torch.float64)

    def rr(x):
        return torch.full_like(x, 0.5)

    def zz(x):
        return torch.cat([rr(x), torch.tensor([vals], dtype=x.dtype).expand(x.shape[0], -1)], dim=1)
    out_r = rate_residuals(rm, {"c": (Yc, T, rr(T)), "s": (Ys, Ts, rr(Ts)), "b": (Tb, rr(Tb))})
    out_c = rate_residuals(cm, {"c": (Yc, T, zz(T)), "s": (Ys, Ts, zz(Ts)), "b": (Tb, zz(Tb))})
    for k in ("ode1", "ode2", "ode3", "charge_s"):
        torch.testing.assert_close(out_c[k].detach(), out_r[k].detach(), rtol=1e-9, atol=1e-11)
    with torch.no_grad():
        V = cm.voltage(T, zz(T))
        torch.testing.assert_close(V, rm.voltage(T, rr(T)), rtol=1e-12, atol=1e-12)
        # analytic parameters: common U0 shift adds, contact resistance subtracts Z I
        torch.testing.assert_close(cm.voltage(T, zz(T), U0_shift=0.07) - V, torch.full_like(V, 0.07), rtol=0, atol=1e-12)
        dZ = cm.voltage(T, zz(T), Z_CC=0.01) - V
        torch.testing.assert_close(dZ, (PARAMS.Z_CC - 0.01) * cm.I(T, zz(T)), rtol=1e-12, atol=1e-12)


def test_sampler_box_and_shapes():
    cfg = ChemTrainConfig(n_cathode=50, n_separator=20, n_boundary=10)
    s = ChemSampler(cfg, torch.float64, torch.Generator().manual_seed(1), "cpu")
    b = s.draw()
    z = b["c"][2]
    assert z.shape == (50, 1 + K) and b["b"][1].shape == (10, 1 + K)
    lo = torch.tensor([x["lo"] for x in SPEC], dtype=torch.float64); hi = torch.tensor([x["hi"] for x in SPEC], dtype=torch.float64)
    assert torch.all(z[:, 1:] >= lo) and torch.all(z[:, 1:] <= hi)
    assert torch.all(z[:, 0] >= cfg.rate_min * (1 - 1e-12)) and torch.all(z[:, 0] <= cfg.rate_max * (1 + 1e-12))


def test_train_save_load_and_evaluate(tmp_path):
    """Tiny run through the shared training loop (hooks), then load_chem_model and evaluate_chem on a synthetic case."""
    coef3, _ = fit_window(list(T_CUT), list(T_CUT.values()))
    cfg = ChemTrainConfig(width=8, depth=1, fourier_t=1, quad_order=4, n_cathode=8, n_separator=4, n_boundary=4,
                          adam_steps=6, log_every=1000, eval_every=3, latest_every=0, checkpoint_every=0, threads=1)
    coef = embed_rate_window(coef3)
    model = build_chem_model(cfg, PARAMS, coef)
    t = np.linspace(0, 2800, 50)
    cases = {"nominal_1C": {"rate": 1.0, "chem": {}, "ref": {"t": t, "V": np.full_like(t, 2.0)}, "select": True},
             "shifted_1C": {"rate": 1.0, "chem": {"dU0_2": 0.05}, "ref": {"t": t, "V": np.full_like(t, 2.0)}, "select": False}}
    train_rate(cfg, PARAMS, coef, tmp_path, refs=cases, log=lambda m: None, model=model, sampler_factory=ChemSampler,
               evaluate_fn=evaluate_chem, score_fn=lambda ev: ev["V_rmse_select_mV"], eval_summary_fn=chem_eval_summary,
               kind="lispan_chem", extra_save={"chem_inputs": [dict(s) for s in SPEC]})
    m2, cfg2, p2 = load_chem_model(tmp_path / "final.pt")
    assert m2.k == K and (tmp_path / "best.pt").exists()
    ev = evaluate_chem(m2, cases)
    assert np.isfinite(ev["V_rmse_select_mV"]) and np.isfinite(ev["V_rmse_test_mV"])
    curve = m2.discharge_curve(0.5, {"k0_1": 1.0}, n=20, V_cut=1.0)
    assert len(curve["t"]) >= 2 and np.all(np.isfinite(curve["V"]))
    with pytest.raises(ValueError):
        m2.discharge_curve(0.5, {"dU0_2": 0.5})


def test_warm_start_from_rate_checkpoint(tmp_path):
    """A Phase 3a network copied into a chem model reproduces it at the nominal chemistry (same window)."""
    from dfn_pinn.lispan.chem_pinn import warm_start_from_rate
    rm, _ = paired_models()
    torch.save({"model": rm.state_dict(), "step": 7}, tmp_path / "rate.pt")
    coef3, _ = fit_window(list(T_CUT), list(T_CUT.values()))
    cm = LiSPANChemPINN(PARAMS, embed_rate_window(coef3), CHEM_INPUTS_V1, 0.1, 1.0, width=16, depth=2, fourier_t=2,
                        quad_order=8, salt_sep_natural_scale=True).double()
    info = warm_start_from_rate(cm, tmp_path / "rate.pt")
    assert info["layers"] == 4 * 3 and info["rate_step"] == 7
    T = torch.rand(9, 1, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    z = torch.cat([torch.full_like(T, 0.4), torch.zeros(9, K, dtype=torch.float64)], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(cm.voltage(T, z), rm.voltage(T, torch.full_like(T, 0.4)), rtol=1e-12, atol=1e-12)


def test_window_cache_and_vectorized_features():
    """The degree-by-degree torch features equal the numpy ones, and the per-tensor t_end cache returns the uncached
    value, recomputes after an in-place change and never mixes two tensors."""
    from dfn_pinn.lispan.chem_pinn import window_features_torch
    rng = np.random.default_rng(3)
    x, th = rng.normal(size=(40, 1)), rng.uniform(-1, 1, (40, K))
    for d in (1, 2, 3, 4):
        assert np.array_equal(window_features(x, th, d), window_features_torch(torch.tensor(x), torch.tensor(th), d).numpy())
    coef = rng.normal(scale=0.01, size=n_window_features(K, 4))
    coef[0] = math.log(2838.0)
    m = LiSPANChemPINN(PARAMS, coef.tolist(), CHEM_INPUTS_V1, 0.1, 1.0, width=8, depth=1, fourier_t=1,
                       quad_order=4).double()
    s = ChemSampler(ChemTrainConfig(chem_inputs=SPEC, n_cathode=64, n_separator=32, n_boundary=16), torch.float64,
                    torch.Generator().manual_seed(0), "cpu")
    z1, z2 = s.draw()["c"][2], s.draw()["c"][2]
    a1 = m.t_end(z1)
    assert torch.equal(a1, m._t_end(z1)) and m.t_end(z1) is a1             # cached
    assert torch.equal(m.t_end(z2), m._t_end(z2))                          # another tensor, own entry
    z1[:, 0] *= 0.5                                                        # in-place change -> recomputed
    assert torch.equal(m.t_end(z1), m._t_end(z1)) and not torch.equal(m.t_end(z1), a1)
    zq = z2.reshape(-1, 1, z2.shape[1]).expand(-1, 3, -1).reshape(-1, z2.shape[1])
    assert torch.equal(m.t_end(zq), m._t_end(zq))
