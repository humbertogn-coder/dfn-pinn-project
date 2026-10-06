"""Consistency tests of the Li-SPAN finite-volume reference (src/dfn_pinn/lispan)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfn_pinn.lispan import LiSPANParams, LiSPANProtocol, solve  # noqa: E402
from dfn_pinn.lispan.params import F  # noqa: E402


@pytest.fixture(scope="module")
def run_01C():
    p = LiSPANParams()
    m, res = solve(p, LiSPANProtocol.from_crate(0.1), N_c=10, N_s=5, n_out=200)
    return p, m, res


def sulfur_inventory(p, m, res, n):
    """Total sulfur per electrode area [mol/m2] at output index n."""
    g = m.g
    dy_c = g.dy[: g.N_c]
    span = ((4 * res["c_S4"][:, n] + 3 * res["c_S3"][:, n] + 2 * res["c_S2"][:, n] + res["c_S1"][:, n]) * dy_c).sum()
    li2s = (res["eps_L"][:, n] / p.V_m_L * dy_c).sum()
    eps_e = np.where(g.is_cathode, 1 - p.eps_SPAN - p.eps_CB - np.concatenate([res["eps_L"][:, n], np.zeros(g.N_s)]), p.eps_e_sep)
    dissolved = (eps_e * res["c_S"][:, n] * g.dy).sum()
    return span + li2s + dissolved


def test_sulfur_and_electron_balance(run_01C):
    p, m, res = run_01C
    S0 = sulfur_inventory(p, m, res, 0)
    assert abs(S0 - 4 * p.c_init[0] * p.L_cat) / (4 * p.c_init[0] * p.L_cat) < 2e-3
    for n in (len(res["t"]) // 3, len(res["t"]) - 1):
        assert abs(sulfur_inventory(p, m, res, n) - S0) / S0 < 2e-3
    # electrons from the reaction extents vs the charge passed (double layer negligible)
    n = len(res["t"]) - 1
    dy_c = m.g.dy[: m.g.N_c]
    xi1 = 2 * (p.c_init[0] - res["c_S4"][:, n])
    xi2 = xi1 - 2 * res["c_S3"][:, n]
    xi3 = xi2 - 2 * res["c_S2"][:, n]
    Q_species = F * ((xi1 + xi2 + xi3) * dy_c).sum()
    assert abs(Q_species - res["Q_C_m2"][n]) / res["Q_C_m2"][n] < 5e-3


def test_capacity_and_cutoff(run_01C):
    p, m, res = run_01C
    assert res["solver"]["success"]
    assert abs(res["V"][-1] - 1.0) < 1e-3                       # stopped at the cut-off
    assert 1150 < res["Q_mAh_gS"][-1] < 1260                    # paper: ~1230 mAh/g_S at 0.1 C
    # monotone decrease after the initial double-layer transient
    V = res["V"][res["t"] > 200.0]
    assert np.all(np.diff(V) < 2e-3)


def test_s2m_saturation_and_li2s(run_01C):
    p, m, res = run_01C
    c_sat = p.c_S_sat * p.K_sp / 10.0
    # pinned at saturation (a_Li a_S = K_sp, a_Li ~ 0.99) in the cathode once precipitation starts; the separator
    # has no Li2S sink and sits slightly above
    assert res["c_S"][: m.g.N_c].max() < 1.03 * c_sat
    assert res["c_S"].max() < 1.1 * c_sat
    assert res["eps_L_avg"][-1] > 0.02                          # ~0.03 at the end (Fig. 5b)
    onset = np.argmax(res["eps_L_avg"] > 1e-4)                  # after the seed has dissolved and nucleation restarted
    assert np.all(np.diff(res["eps_L_avg"][onset:]) > -1e-9)    # no dissolution during discharge


def test_rate_dependence_and_zcc():
    p0 = LiSPANParams(Z_CC=0.0)
    _, r01 = solve(p0, LiSPANProtocol.from_crate(0.1), N_c=10, N_s=5, n_out=150)
    _, r1 = solve(p0, LiSPANProtocol.from_crate(1.0), N_c=10, N_s=5, n_out=150)
    q = 300.0
    v01 = np.interp(q, r01["Q_mAh_gS"], r01["V"]); v1 = np.interp(q, r1["Q_mAh_gS"], r1["V"])
    assert 0.10 < v01 - v1 < 0.25                               # paper Fig. 6a: ~0.17 V between 0.1 C and 1 C
    _, r1z = solve(LiSPANParams(Z_CC=0.025), LiSPANProtocol.from_crate(1.0), N_c=10, N_s=5, n_out=150)
    v1z = np.interp(q, r1z["Q_mAh_gS"], r1z["V"])
    assert abs((v1 - v1z) - 0.25) < 0.02                        # Z_CC I = 0.025 * 10 A/m2


def test_tafel_slope_in_k0():
    """Phase-1 voltage drops by about (2RT/F) ln 10 = 0.118 V per decade of k0 (paper Fig. 6a: 0.12; here
    0.09 because the reverse term of the reversible reaction (1) is not negligible at the operating point)."""
    out = []
    for s in (1.0, 0.1):
        p = LiSPANParams(Z_CC=0.0, kin_scale=s)
        _, r = solve(p, LiSPANProtocol.from_crate(0.1), N_c=10, N_s=5, n_out=150, t_end=6000.0)
        out.append(np.interp(100.0, r["Q_mAh_gS"], r["V"]))
    assert 0.07 < out[0] - out[1] < 0.14


def test_grid_convergence():
    p = LiSPANParams()
    _, a = solve(p, LiSPANProtocol.from_crate(0.2), N_c=10, N_s=5, n_out=150)
    _, b = solve(p, LiSPANProtocol.from_crate(0.2), N_c=30, N_s=15, n_out=150)
    q = np.linspace(50, min(a["Q_mAh_gS"][-1], b["Q_mAh_gS"][-1]) - 20, 100)
    e = np.interp(q, a["Q_mAh_gS"], a["V"]) - np.interp(q, b["Q_mAh_gS"], b["V"])
    assert np.sqrt(np.mean(e ** 2)) < 5e-3


def test_reversible_reaction3_comproportionation():
    """With the full mass-action reverse term of reaction (3), PAN-SLi is consumed during phase 2
    (electrode-mediated S3 + S1 -> 2 S2); the default (irreversible (3)) keeps c_S1 at 598 mol/m3."""
    p_rev = LiSPANParams(reversible=(True, True, True))
    _, r = solve(p_rev, LiSPANProtocol.from_crate(0.1), N_c=10, N_s=5, n_out=150, t_end=22000.0)
    _, d = solve(LiSPANParams(), LiSPANProtocol.from_crate(0.1), N_c=10, N_s=5, n_out=150, t_end=22000.0)
    q = 600.0
    assert np.interp(q, r["Q_mAh_gS"], r["c_S1_avg"]) < 500.0
    assert abs(np.interp(q, d["Q_mAh_gS"], d["c_S1_avg"]) - 598.0) < 5.0
