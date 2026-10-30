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


def paired_models(spec=CHEM_INPUTS_V1):
    """A rate model and a chem model with the same weights (chemistry inputs disconnected)."""
    coef3, _ = fit_window(list(T_CUT), list(T_CUT.values()))
    rm = LiSPANRatePINN(PARAMS, coef3, 0.1, 1.0, width=16, depth=2, fourier_t=2, quad_order=8,
                        salt_sep_natural_scale=True).double()
    cm = LiSPANChemPINN(PARAMS, embed_rate_window(coef3, len(spec)), spec, 0.1, 1.0, width=16, depth=2, fourier_t=2,
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


# ---------------------------------------------------------------------------------------------------------------- v2
def test_v2_spec_effects_and_kinetics():
    """v2 inputs (dU0_3rel2, b_12, b_3): spec checks, the FV map apply_chem, and the per-point kinetics of the network
    equal the rate model with the same parameters as scalar multipliers."""
    from dfn_pinn.lispan.chem_pinn import CHEM_INPUTS_V2, chem_effects
    spec2 = check_spec(CHEM_INPUTS_V2)
    with pytest.raises(ValueError):
        check_spec([{"name": "b_12", "lo": -1, "hi": 1}, {"name": "b_1", "lo": -1, "hi": 1}])
    with pytest.raises(ValueError):
        check_spec([{"name": "dU0_3", "lo": -1, "hi": 1}, {"name": "dU0_3rel2", "lo": -1, "hi": 1}])
    vals = [0.05, -0.03, math.log(1.3), math.log(0.8), math.log(3.0)]
    dU, lb, lk = chem_effects(spec2, vals)
    assert np.allclose(dU, [0.0, 0.05, 0.02]) and np.allclose(np.exp(lb), [1.3, 1.3, 0.8]) and np.allclose(np.exp(lk), [3, 1, 1])
    p2 = apply_chem(PARAMS, spec2, vals)
    assert np.allclose(np.array(p2.U0) - np.array(PARAMS.U0), [0.0, 0.05, 0.02])
    assert np.allclose(np.array(p2.b) / np.array(PARAMS.b), [1.3, 1.3, 0.8])
    # the steps stay in order over the whole box: gap 1-2 >= 0.02 V, gap 2-3 >= 0.04 V
    g12 = PARAMS.U0[0] - PARAMS.U0[1]; g23 = PARAMS.U0[1] - PARAMS.U0[2]
    assert g12 - CHEM_INPUTS_V2[0]["hi"] >= 0.019 and g23 - CHEM_INPUTS_V2[1]["hi"] >= 0.039
    rm, cm = paired_models(spec2)
    rm.set_parameter_values({"U0_2": 0.05, "U0_3": 0.02, "b_1": 1.3, "b_2": 1.3, "b_3": 0.8, "k0_1": 3.0})
    g = torch.Generator().manual_seed(9)
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


def test_cliff_metrics():
    """A cliff moved by 0.3 % of the discharge: large rms, small rms outside the cliff, shift found."""
    from dfn_pinn.lispan.chem_pinn import cliff_metrics
    x = np.linspace(0.01, 0.97, 3000)

    def curve(x0):
        return 2.1 - 0.4 * x - 0.15 * (1 + np.tanh((x - x0) / 0.002)) / 2
    r_all, r_out, shifts, share = cliff_metrics(x, curve(0.5), curve(0.503))
    assert r_all > 3.0 and r_out < 1e-3 and len(shifts) == 1 and abs(shifts[0] - 0.3) < 0.02 and 0 < share < 0.05
    r_all, r_out, shifts, _ = cliff_metrics(x, 2.1 - 0.4 * x, 2.1 - 0.4 * x + 1e-3)
    assert shifts == [] and abs(r_all - 1.0) < 1e-9 and abs(r_out - 1.0) < 1e-9


def test_rar_sampler_pool_and_mix():
    """ChemRARSampler keeps the worst candidates and mixes rar_fraction of them into the cathode batch, inside the box;
    with rar off it draws exactly what ChemSampler draws."""
    from dfn_pinn.lispan.chem_pinn import CHEM_INPUTS_V2, ChemRARSampler
    spec2 = check_spec(CHEM_INPUTS_V2)
    cfg = ChemTrainConfig(chem_inputs=spec2, n_cathode=40, n_separator=8, n_boundary=4, rar=True, rar_every=10,
                          rar_candidates=500, rar_pool=20, rar_fraction=0.25, rar_chunk=128, rar_jitter_T=1e-3)

    def fake_residuals(model, batch):                  # "kinetics residual" large only where T > 0.9
        T = batch["c"][1]
        big = (T > 0.9).to(T.dtype)
        return {"ode1": big * 10.0 + 1e-3, "salt_c": torch.zeros_like(T)}
    s = ChemRARSampler(cfg, torch.float64, torch.Generator().manual_seed(3), "cpu")
    s.before_step(1, None, fake_residuals)
    assert s.pool[1].min() > 0.9 and s.pool[0].shape == (20, 1)
    b = s.draw()
    Y, T, Z = b["c"]
    assert Y.shape == (40, 1) and Z.shape == (40, 1 + len(spec2))
    assert (T[-10:] > 0.89).all()                                       # the last 25 % come from the pool
    Yp, Tp, _ = s._from_pool(2000)
    assert float(Tp.min()) > 0.0 and float(Yp.min()) > 0.0 and float(Tp.max()) <= 1.0   # reflected, never exactly 0
    lo = torch.tensor([x["lo"] for x in spec2], dtype=torch.float64); hi = torch.tensor([x["hi"] for x in spec2], dtype=torch.float64)
    assert torch.all(Z[:, 1:] >= lo) and torch.all(Z[:, 1:] <= hi) and torch.all(Z[:, 0] <= cfg.rate_max + 1e-12)
    pool_before = s.pool
    s.before_step(5, None, fake_residuals); assert s.pool is pool_before        # not yet rar_every steps later
    s.before_step(11, None, fake_residuals); assert s.pool is not pool_before
    cfg_off = ChemTrainConfig(chem_inputs=spec2, n_cathode=40, n_separator=8, n_boundary=4, rar=False)
    a = ChemRARSampler(cfg_off, torch.float64, torch.Generator().manual_seed(4), "cpu")
    c = ChemSampler(cfg_off, torch.float64, torch.Generator().manual_seed(4), "cpu")
    a.before_step(1, None, fake_residuals)
    for x_, y_ in zip(a.draw()["c"], c.draw()["c"]):
        assert torch.equal(x_, y_)


def test_train_with_rar_through_the_loop(tmp_path):
    """A tiny v2 run with residual-based sampling through train_rate (the before_step hook and the cliff metrics)."""
    from dfn_pinn.lispan.chem_pinn import CHEM_INPUTS_V2, ChemRARSampler
    spec2 = check_spec(CHEM_INPUTS_V2)
    coef3, _ = fit_window(list(T_CUT), list(T_CUT.values()))
    cfg = ChemTrainConfig(chem_inputs=spec2, width=8, depth=1, fourier_t=1, quad_order=4, n_cathode=8, n_separator=4,
                          n_boundary=4, adam_steps=6, log_every=1000, eval_every=3, latest_every=0, checkpoint_every=0,
                          threads=1, rar=True, rar_every=2, rar_candidates=64, rar_pool=8, rar_chunk=32)
    coef = embed_rate_window(coef3, len(spec2))
    model = build_chem_model(cfg, PARAMS, coef)
    t = np.linspace(0, 2800, 60)
    cases = {"nominal_1C": {"rate": 1.0, "chem": {}, "ref": {"t": t, "V": 2.0 - 0.1 * (t > 1400)}, "select": True},
             "other_1C": {"rate": 1.0, "chem": {"b_3": 0.5}, "ref": {"t": t, "V": np.full_like(t, 2.0)}, "select": False}}
    logs = []
    train_rate(cfg, PARAMS, coef, tmp_path, refs=cases, log=logs.append, model=model, sampler_factory=ChemRARSampler,
               evaluate_fn=evaluate_chem, score_fn=lambda ev: ev["V_rmse_select_mV"], eval_summary_fn=chem_eval_summary,
               kind="lispan_chem", extra_save={})
    assert (tmp_path / "final.pt").exists()
    assert any("outside cliffs" in m for m in logs) and any(m.startswith("  sampler:") for m in logs)


def test_warm_start_from_a_chem_checkpoint(tmp_path):
    """A chem checkpoint with the same inputs is copied as it is (e.g. v2 -> a wider rate range)."""
    from dfn_pinn.lispan.chem_pinn import warm_start_from_rate
    _, src = paired_models(CHEM_INPUTS_V1)
    torch.manual_seed(3)
    _, dst = paired_models(CHEM_INPUTS_V1)
    with torch.no_grad():
        for prm in dst.parameters():
            prm.add_(torch.randn_like(prm))
    torch.save({"model": src.state_dict(), "step": 7}, tmp_path / "ck.pt")
    info = warm_start_from_rate(dst, tmp_path / "ck.pt")
    assert info["layers"] > 0 and info["rate_step"] == 7
    for name in ("net_R", "net_xi", "net_ce", "net_phie"):
        for a, b in zip(getattr(src, name).net, getattr(dst, name).net):
            if isinstance(a, torch.nn.Linear):
                assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)


def test_window_on_quadrature_expanded_conditions():
    """_t_end evaluates the window once per row of a quadrature-expanded condition tensor; same values as row by row."""
    _, cm = paired_models(CHEM_INPUTS_V1)
    q = cm.q_nodes.shape[0]
    z0 = torch.cat([torch.rand(40, 1, dtype=torch.float64) * 0.9 + 0.1,
                    torch.rand(40, K, dtype=torch.float64) * 0.2 - 0.1], 1)
    z = z0.reshape(40, 1, -1).expand(40, q, z0.shape[1]).reshape(40 * q, -1)
    assert torch.equal(cm._t_end(z), cm._t_end_rows(z))
    zr = z.clone(); zr[1, 0] += 1e-3                                  # not a repetition any more: plain path
    assert torch.equal(cm._t_end(zr), cm._t_end_rows(zr))


def test_initial_weights_and_freeze_floor(tmp_path):
    """initial_weights set the group weights of a new run; freeze_weight_min bounds them at the freeze."""
    from dfn_pinn.lispan.chem_pinn import CHEM_INPUTS_V2, ChemSampler
    spec2 = check_spec(CHEM_INPUTS_V2)
    coef3, _ = fit_window(list(T_CUT), list(T_CUT.values()))
    coef = embed_rate_window(coef3, len(spec2))
    for init, floor, stop in (({"inventory": 3.0, "ode": 7.0}, {}, 1), ({}, {"inventory": 5.0}, 3)):
        cfg = ChemTrainConfig(chem_inputs=spec2, width=8, depth=1, fourier_t=1, quad_order=4, n_cathode=8, n_separator=4,
                              n_boundary=4, adam_steps=4, log_every=1000, eval_every=0, latest_every=0, checkpoint_every=0,
                              threads=1, initial_weights=init, freeze_weight_min=floor, adaptive_stop=stop,
                              charge_total_start=0)
        model = build_chem_model(cfg, PARAMS, coef)
        d = tmp_path / f"run{stop}"
        train_rate(cfg, PARAMS, coef, d, refs={}, log=lambda m: None, model=model, sampler_factory=ChemSampler,
                   evaluate_fn=evaluate_chem, score_fn=lambda ev: 0.0, eval_summary_fn=chem_eval_summary,
                   kind="lispan_chem", extra_save={})
        w = torch.load(d / "final.pt", map_location="cpu", weights_only=False)["group_weights"]
        if init:
            assert w["inventory"] == 3.0 and w["ode"] == 7.0          # frozen from step 1: never adapted
        else:
            assert w["inventory"] >= 5.0
