"""Finite-volume / method-of-lines reference solver for the Li-SPAN model.

Conventions (our own, see docs/LI_SPAN_MODEL_FORMULATION.md section 4):

* y = 0 cathode current collector, y = L_cat cathode/separator interface, y = L_tot Li surface.
* I > 0 is discharge [A/m2].  i_e(y) is the ionic current density in the +y direction: i_e = 0 at
  y = 0, i_e = -I in the separator; i_s + i_e = -I in the cathode.
* Reaction rates r_m > 0 in the reduction (discharge) direction, per m2 of SPAN surface:
      r_m = kin_scale k0_m [ a_ed,m exp(-x_m) - a_prod,m exp(+x_m) ],
      x_m = F (dphi - U_m^ref) / (2RT),  U_m^ref = U0_m - b_m zeta_m,
  which is the paper's generalized Butler-Volmer expression (S1, main text (4)-(8)) written in
  mass-action form; the Nernst term RT ln(a_ed/a_prod) of (7) is included.
* Electrolyte: Nernst-Planck fluxes with the dilute ion diffusivities D+ = D_salt/(2(1-t+)),
  D- = D_salt/(2t+) (salt diffusion coefficient and transference number exact) and the *measured*
  conductivity kappa0 (c/c0) eps^beta for the migration current, so that the diffusion potential
  agrees with concentrated-solution theory to about 10 %.  Electroneutrality c_Li = c_PF6 + 2 c_S.
* Double layer: a c_DL d(dphi)/dt = d i_e/dy + F a sum_m r_m (dphi = phi_s - phi_e).
* Cell voltage E = phi_s(0) - Z_CC I with phi_s(Li) = 0.

State vector: cathode cells (v_Sx = ln(c_Sx/c_ref,x) for x = 4, 3, 2, 1, eps_L, dphi), then c_PF6 and
w = ln(c_S2-/c_S_ref) in all cells.  The SPAN species and S2- are integrated in log form: they span
many orders of magnitude, the sqrt(activity) factors of the generalized Butler-Volmer rates make the
plain form extremely stiff near zero, and positivity is guaranteed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .params import F, R, LiSPANParams, LiSPANProtocol

Z = {"Li": 1.0, "PF6": -1.0, "S": -2.0}


@dataclass
class Grid:
    N_c: int
    N_s: int
    L_cat: float
    L_sep: float

    def __post_init__(self):
        self.N = self.N_c + self.N_s
        self.dy = np.concatenate([np.full(self.N_c, self.L_cat / self.N_c), np.full(self.N_s, self.L_sep / self.N_s)])
        self.y_f = np.concatenate([[0.0], np.cumsum(self.dy)])          # faces (N+1)
        self.y_c = 0.5 * (self.y_f[1:] + self.y_f[:-1])                 # centres (N)
        self.d_int = np.diff(self.y_c)                                  # centre-to-centre, interior faces (N-1)
        self.is_cathode = np.arange(self.N) < self.N_c


class LiSPANModel:
    def __init__(self, params: LiSPANParams, protocol: LiSPANProtocol, N_c: int = 20, N_s: int = 10):
        self.p = params
        self.prot = protocol
        self.g = Grid(N_c, N_s, params.L_cat, params.L_sep)
        p, g = params, self.g
        self.RT = R * p.T
        self.f = F / self.RT
        # dilute ion diffusivities reproducing D_salt and t+
        self.D0 = {"Li": p.D_salt / (2.0 * (1.0 - p.t_plus)), "PF6": p.D_salt / (2.0 * p.t_plus), "S": p.D_S}
        self.beta = np.where(g.is_cathode, p.beta_cat, p.beta_sep)
        self.n_state = 6 * g.N_c + 2 * g.N
        # slices
        Nc, N = g.N_c, g.N
        self.sl = {"v_S4": slice(0, Nc), "v_S3": slice(Nc, 2 * Nc), "v_S2": slice(2 * Nc, 3 * Nc),
                   "v_S1": slice(3 * Nc, 4 * Nc), "l_L": slice(4 * Nc, 5 * Nc), "dphi": slice(5 * Nc, 6 * Nc),
                   "c_PF6": slice(6 * Nc, 6 * Nc + N), "w_S": slice(6 * Nc + N, 6 * Nc + 2 * N)}
        cell = np.concatenate([np.tile(np.arange(Nc), 6), np.arange(N), np.arange(N)])
        self.cell_of_state = cell
        self.jac_sparsity = (np.abs(cell[:, None] - cell[None, :]) <= 1).astype(float)

    # ------------------------------------------------------------------ pieces
    def unpack(self, u):
        st = {k: u[s] for k, s in self.sl.items()}
        for k, key in enumerate(("S4", "S3", "S2", "S1")):
            st["c_" + key] = self.p.c_ref[k] * np.exp(np.minimum(st["v_" + key], 60.0))
        st["c_S"] = self.p.c_S_ref * np.exp(np.minimum(st["w_S"], 60.0))
        # Li2S cannot exceed the initial pore volume; the cap only acts on Newton trial states
        st["eps_L"] = np.minimum(self.p.eps_L_seed * np.exp(np.minimum(st["l_L"], 15.0)), 0.9 * self.p.eps_e0)
        return st

    def rates(self, st, I=None):
        """Reaction rates (per m2 SPAN) and geometric quantities in the cathode cells."""
        p = self.p
        c_S4, c_S3, c_S2, c_S1 = (st[k] for k in ("c_S4", "c_S3", "c_S2", "c_S1"))
        eps_L = st["eps_L"]
        dphi = st["dphi"]
        Nc = self.g.N_c
        c_PF6 = np.maximum(st["c_PF6"][:Nc], 0.0)
        c_S = np.maximum(st["c_S"][:Nc], 0.0)
        c_Li = c_PF6 + 2.0 * c_S
        eps_e = 1.0 - p.eps_SPAN - p.eps_CB - eps_L
        a = p.a_SPAN0 * (eps_e / p.eps_e0) ** p.xi
        aS4, aS3, aS2, aS1 = c_S4 / p.c_ref[0], c_S3 / p.c_ref[1], c_S2 / p.c_ref[2], c_S1 / p.c_ref[3]
        aLi = c_Li / p.c_Li0
        aS = c_S / p.c_S_ref                                   # S2- activity in the SPAN kinetics
        if p.frozen_S2m:
            aS = np.full_like(aS, p.c_S_sat / p.c_S_ref)
        aS_L = c_S * p.K_sp / p.c_S_sat                        # S2- activity in the Li2S equilibrium (a_Li aS_L = K_sp at saturation)
        zeta = [np.clip(1.0 - aS4, 0.0, 1.0), np.clip(1.0 - aS3, 0.0, 1.0), np.clip(1.0 - aS2, 0.0, 1.0)]
        Uref = [p.U0[m] - p.b[m] * zeta[m] for m in range(3)]
        x = [np.clip(0.5 * self.f * (dphi - Uref[m]), -80.0, 80.0) for m in range(3)]
        ks = p.kin_scale
        rev = [1.0 if r else 0.0 for r in p.reversible]
        r1 = ks * p.k0[0] * (np.sqrt(aS4) * aLi * np.exp(-x[0]) - rev[0] * np.sqrt(aS3 * aS1) * np.exp(x[0]))
        r2 = ks * p.k0[1] * (np.sqrt(aS3) * np.exp(-x[1]) - rev[1] * np.sqrt(aS2 * aS) * np.exp(x[1]))
        r3 = ks * p.k0[2] * (np.sqrt(aS2) * np.exp(-x[2]) - rev[2] * np.sqrt(aS1 * aS) * np.exp(x[2]))
        rL = p.precip_scale * p.k0_L / np.sqrt(p.K_sp) * (aLi * aS_L - p.K_sp)
        # (S40) a_L = a_SPAN eps_L, with a permanent nucleation seed for precipitation; dissolution acts on
        # the existing precipitate only, so eps_L stays positive (log state) and nucleation can always restart
        a_L = a * np.where(rL > 0.0, eps_L + p.eps_L_seed, eps_L)
        return {"r1": r1, "r2": r2, "r3": r3, "rL": rL, "a": a, "a_L": a_L, "eps_e_c": eps_e,
                "Uref": Uref, "aLi_c": aLi}

    def electrolyte(self, st, I, rt):
        """Face fluxes, ionic current and potential gradients."""
        p, g = self.p, self.g
        Nc, N = g.N_c, g.N
        c_PF6 = st["c_PF6"]; c_S = st["c_S"]
        c_Li = c_PF6 + 2.0 * c_S
        conc = {"Li": c_Li, "PF6": c_PF6, "S": c_S}
        eps_e = np.where(g.is_cathode, np.concatenate([rt["eps_e_c"], np.zeros(g.N_s)]), p.eps_e_sep)
        eps_e[Nc:] = p.eps_e_sep
        brug = eps_e ** self.beta
        D_cell = {k: self.D0[k] * brug for k in conc}
        kap_cell = p.kappa0 * np.maximum(c_Li, 1e-9) / p.c_Li0 * brug
        # interior faces 1..N-1
        D_f = {k: 2.0 * D_cell[k][1:] * D_cell[k][:-1] / (D_cell[k][1:] + D_cell[k][:-1]) for k in conc}
        kap_f = 2.0 * kap_cell[1:] * kap_cell[:-1] / (kap_cell[1:] + kap_cell[:-1])
        c_f = {k: 0.5 * (conc[k][1:] + conc[k][:-1]) for k in conc}
        grad = {k: (conc[k][1:] - conc[k][:-1]) / g.d_int for k in conc}
        B_f = F * sum(Z[k] * D_f[k] * grad[k] for k in conc)
        S_f = sum(Z[k] ** 2 * D_f[k] * np.maximum(c_f[k], 0.0) for k in conc)
        t_f = {k: Z[k] ** 2 * D_f[k] * np.maximum(c_f[k], 0.0) / S_f for k in conc}
        # ionic current at all faces (N+1)
        i_e = np.full(N + 1, -I)
        i_e[0] = 0.0
        ks_ = p.kappa_s_eff
        dphi = st["dphi"]
        dphi_grad = (dphi[1:] - dphi[:-1]) / g.d_int[:Nc - 1]
        kf_c = kap_f[:Nc - 1]; Bf_c = B_f[:Nc - 1]
        i_e[1:Nc] = (dphi_grad - I / ks_ - Bf_c / kf_c) / (1.0 / ks_ + 1.0 / kf_c)
        # fluxes at interior faces
        i_mig = i_e[1:N] + B_f
        Nflux = {k: -D_f[k] * grad[k] + t_f[k] * i_mig / (Z[k] * F) for k in conc}
        phi_grad_int = -i_mig / kap_f
        # anode face (y = L_tot): N_PF6 = N_S = 0 -> BC-consistent gradients; i_e = -I
        D_an = {k: D_cell[k][-1] for k in conc}
        c_an = {k: conc[k][-1] for k in conc}
        S_an = sum(Z[k] ** 2 * D_an[k] * max(c_an[k], 0.0) for k in conc)
        t_an = {k: Z[k] ** 2 * D_an[k] * max(c_an[k], 0.0) / S_an for k in conc}
        # zero flux for PF6 and S: -D grad c + t (i_e+B)/(zF) = 0 with i_e = -I; solve for B self-consistently:
        # grad_k = t_k (i_e + B)/(z_k F D_k) for k in {PF6, S}; grad_Li = grad_PF6 + 2 grad_S;
        # B = F sum z_k D_k grad_k  ->  linear in B.
        def grads_for(Bval):
            gP = t_an["PF6"] * (-I + Bval) / (Z["PF6"] * F * D_an["PF6"])
            gS = t_an["S"] * (-I + Bval) / (Z["S"] * F * D_an["S"])
            return {"PF6": gP, "S": gS, "Li": gP + 2.0 * gS}
        g0 = grads_for(0.0); g1 = grads_for(1.0)
        B0 = F * sum(Z[k] * D_an[k] * g0[k] for k in conc)
        B1 = F * sum(Z[k] * D_an[k] * g1[k] for k in conc)
        slope = B1 - B0
        B_an = B0 / (1.0 - slope) if abs(1.0 - slope) > 1e-12 else B0
        phi_grad_an = -(-I + B_an) / kap_cell[-1]
        return {"Nflux": Nflux, "i_e": i_e, "eps_e": eps_e, "phi_grad_int": phi_grad_int,
                "phi_grad_an": phi_grad_an, "c_Li": c_Li, "kap_cell": kap_cell}

    # ------------------------------------------------------------------ rhs
    def rhs(self, t, u):
        p, g = self.p, self.g
        Nc, N = g.N_c, g.N
        I = self.prot.I(t)
        st = self.unpack(u)
        rt = self.rates(st)
        el = self.electrolyte(st, I, rt)
        a, a_L, r1, r2, r3, rL = rt["a"], rt["a_L"], rt["r1"], rt["r2"], rt["r3"], rt["rL"]
        du = np.zeros_like(u)
        fl = p.c_floor_SPAN
        du[self.sl["v_S4"]] = -0.5 * a * r1 / (st["c_S4"] + fl)
        du[self.sl["v_S3"]] = 0.5 * a * (r1 - r2) / (st["c_S3"] + fl)
        du[self.sl["v_S2"]] = 0.5 * a * (r2 - r3) / (st["c_S2"] + fl)
        du[self.sl["v_S1"]] = 0.5 * a * (r1 + r3) / (st["c_S1"] + fl)
        deps_L = p.V_m_L * a_L * rL
        du[self.sl["l_L"]] = deps_L / (st["eps_L"] + 1e-2 * p.eps_L_seed)
        # double layer
        i_e = el["i_e"]
        du[self.sl["dphi"]] = ((i_e[1:Nc + 1] - i_e[:Nc]) / g.dy[:Nc] + F * a * (r1 + r2 + r3)) / (a * p.c_DL)
        # electrolyte species: fluxes at all faces (0 at y=0; PF6, S zero at anode)
        eps_e = el["eps_e"]
        deps = np.concatenate([-deps_L, np.zeros(g.N_s)])
        for key, name in (("c_PF6", "PF6"), ("w_S", "S")):
            flux = np.zeros(N + 1)
            flux[1:N] = el["Nflux"][name]
            src = np.zeros(N)
            if name == "S":
                src[:Nc] = 0.5 * a * (r2 + r3) - a_L * rL
            c = st["c_S"] if name == "S" else st[key]
            dc = (-(flux[1:] - flux[:-1]) / g.dy + src - c * deps) / eps_e
            du[self.sl[key]] = dc / (c + p.c_floor_S) if name == "S" else dc      # d ln c / dt for S2-
        return du

    # ------------------------------------------------------------------ diagnostics
    def potentials(self, t, u):
        """phi_e at cell centres, phi_s at cathode centres, cell voltage."""
        p, g = self.p, self.g
        Nc, N = g.N_c, g.N
        I = self.prot.I(t)
        st = self.unpack(u)
        rt = self.rates(st)
        el = self.electrolyte(st, I, rt)
        aLi_an = max(el["c_Li"][-1], 1e-9) / p.c_Li0
        eta_an = (2.0 * self.RT / F) * np.arcsinh(I / (2.0 * F * p.k0_Li * np.sqrt(aLi_an)))
        phi_e_an = -(self.RT / F) * np.log(aLi_an) - eta_an
        phi_e = np.empty(N)
        phi_e[-1] = phi_e_an - el["phi_grad_an"] * 0.5 * g.dy[-1]
        for k in range(N - 2, -1, -1):
            phi_e[k] = phi_e[k + 1] - el["phi_grad_int"][k] * g.d_int[k]
        phi_s = st["dphi"] + phi_e[:Nc]
        phi_s0 = phi_s[0] - I / p.kappa_s_eff * 0.5 * g.dy[0]     # phi_s' = (I + i_e)/kappa_s, i_e(0) = 0
        E = phi_s0 - p.Z_CC * I
        return {"phi_e": phi_e, "phi_s": phi_s, "E": E, "i_e": el["i_e"], "eta_an": eta_an, "rt": rt, "I": I}

    def voltage(self, t, u):
        return self.potentials(t, u)["E"]

    # ------------------------------------------------------------------ initial state
    def initial_state(self, equilibrate_S2m: bool = True):
        p, g = self.p, self.g
        Nc, N = g.N_c, g.N
        u = np.zeros(self.n_state)
        for k, key in enumerate(("v_S4", "v_S3", "v_S2", "v_S1")):
            u[self.sl[key]] = np.log(p.c_init[k] / p.c_ref[k])
        u[self.sl["l_L"]] = np.log(p.eps_L0 / p.eps_L_seed)
        u[self.sl["c_PF6"]] = p.c_PF6_0
        u[self.sl["w_S"]] = np.log(p.c_S0 / p.c_S_ref)
        # initial potential difference: equilibrium of reaction (1), the only one with educts present
        st = self.unpack(u)

        def net(d):
            st["dphi"] = np.full(Nc, d)
            return float(self.rates(st)["r1"][0])
        dphi0 = brentq(net, 0.0, 4.0, xtol=1e-12)
        u[self.sl["dphi"]] = dphi0
        if equilibrate_S2m:
            # S2- in equilibrium with the reverse of reactions (2) and (3) at dphi0 (the paper's initial
            # 0.01 mol/m3 is oxidised within microseconds; its only effect is a violent initial transient)
            st["dphi"] = np.full(Nc, dphi0)
            rt = self.rates(st)
            x2 = 0.5 * self.f * (dphi0 - rt["Uref"][1][0]); x3 = 0.5 * self.f * (dphi0 - rt["Uref"][2][0])
            aS3, aS2, aS1 = p.c_init[1] / p.c_ref[1], p.c_init[2] / p.c_ref[2], p.c_init[3] / p.c_ref[3]
            a_eq = min(aS3 / aS2 * np.exp(-2.0 * x2), aS2 / aS1 * np.exp(-2.0 * x3), p.c_S0 / p.c_S_ref)
            u[self.sl["w_S"]] = np.log(max(a_eq, p.c_floor_S / p.c_S_ref))
        return u


def solve(params: LiSPANParams, protocol: LiSPANProtocol, *, N_c=20, N_s=10, n_out=600,
          rtol=1e-6, atol=None, method="BDF", t_end=None, max_step=np.inf, equilibrate_S2m=True):
    """Galvanostatic discharge until V_min or t_end.  Returns (model, result dict)."""
    m = LiSPANModel(params, protocol, N_c, N_s)
    u0 = m.initial_state(equilibrate_S2m)
    t_end = t_end if t_end is not None else protocol.t_end(params)
    if atol is None:
        atol = np.full(m.n_state, 1e-6)
        atol[m.sl["w_S"]] = 1e-6
        atol[m.sl["l_L"]] = 1e-6
        atol[m.sl["dphi"]] = 1e-8
        atol[m.sl["c_PF6"]] = 1e-6
        for key in ("v_S4", "v_S3", "v_S2", "v_S1"):
            atol[m.sl[key]] = 1e-7

    def cutoff(t, u):
        return m.voltage(t, u) - protocol.V_min
    cutoff.terminal = True
    cutoff.direction = -1
    early = np.linspace(0.0, min(20.0 * max(protocol.ramp_s, 1.0), t_end), 101)
    t_eval = np.unique(np.concatenate([early, np.linspace(0.0, t_end, n_out)]))
    sol = solve_ivp(m.rhs, (0.0, t_end), u0, method=method, t_eval=t_eval, events=cutoff,
                    rtol=rtol, atol=atol, jac_sparsity=m.jac_sparsity, max_step=max_step)
    t = sol.t; U = sol.y
    if sol.t_events[0].size:
        t = np.append(t, sol.t_events[0][0]); U = np.column_stack([U, sol.y_events[0][0]])
    res = assemble(m, t, U)
    res["solver"] = {"success": bool(sol.success), "message": sol.message, "nfev": int(sol.nfev),
                     "njev": int(sol.njev), "nlu": int(sol.nlu), "method": method, "rtol": rtol}
    return m, res


def assemble(m: LiSPANModel, t, U):
    p, g = m.p, m.g
    Nc, N = g.N_c, g.N
    nt = len(t)
    out = {"t": np.asarray(t), "y": g.y_c, "y_f": g.y_f, "y_c": g.y_c[:Nc]}
    for key, s in m.sl.items():
        out[key] = U[s, :]
    for k, key in enumerate(("S4", "S3", "S2", "S1")):
        out["c_" + key] = p.c_ref[k] * np.exp(np.minimum(out["v_" + key], 60.0))
    out["c_S"] = p.c_S_ref * np.exp(np.minimum(out["w_S"], 60.0))
    out["eps_L"] = np.minimum(p.eps_L_seed * np.exp(np.minimum(out["l_L"], 15.0)), 0.9 * p.eps_e0)
    out["c_Li"] = out["c_PF6"] + 2.0 * out["c_S"]
    V = np.empty(nt); I = np.empty(nt); phi_e = np.empty((N, nt)); phi_s = np.empty((Nc, nt))
    i_e = np.empty((N + 1, nt)); r = {k: np.empty((Nc, nt)) for k in ("r1", "r2", "r3", "rL", "a", "a_L")}
    for n in range(nt):
        d = m.potentials(t[n], U[:, n])
        V[n] = d["E"]; I[n] = d["I"]; phi_e[:, n] = d["phi_e"]; phi_s[:, n] = d["phi_s"]; i_e[:, n] = d["i_e"]
        for k in r:
            r[k][:, n] = d["rt"][k]
    out.update({"V": V, "I": I, "phi_e": phi_e, "phi_s": phi_s, "i_e": i_e, **r})
    ramp = m.prot.ramp_s
    tt = np.asarray(t, dtype=float)
    # integral of I0 tanh(t/ramp) = I0 ramp ln cosh(t/ramp), written overflow-free
    Q = m.prot.current * (tt + ramp * (np.log1p(np.exp(-2.0 * tt / ramp)) - np.log(2.0)) if ramp > 0 else tt)   # C/m2
    out["Q_C_m2"] = Q
    out["Q_Ah_m2"] = Q / 3600.0
    out["Q_mAh_gS"] = Q / 3600.0 * 1e3 / (p.m_S_model * 1e3)        # mAh per g sulfur (model sulfur content)
    out["Q_mAh_gS_nominal"] = Q / 3600.0 * 1e3 / (p.m_S_nominal * 1e3)
    # cathode averages
    for key in ("c_S4", "c_S3", "c_S2", "c_S1", "eps_L", "dphi"):
        out[key + "_avg"] = out[key].mean(axis=0)
    for key in ("c_Li", "c_PF6", "c_S"):
        out[key + "_cat_avg"] = out[key][:Nc].mean(axis=0)
    return out


def export(res: dict, path, params: LiSPANParams, protocol: LiSPANProtocol, extra_meta: dict | None = None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": "Li-SPAN (Simanjuntak 2024) finite-volume reference", "params": params.to_dict(),
            "protocol": protocol.to_dict(), "solver": res.get("solver", {})}
    if extra_meta:
        meta.update(extra_meta)
    data = {k: np.asarray(v) for k, v in res.items() if k != "solver"}
    np.savez_compressed(path, meta=json.dumps(meta), **data)
    return path


def load(path) -> dict:
    raw = np.load(path, allow_pickle=False)
    data = {k: raw[k] for k in raw.files if k != "meta"}
    data["meta"] = json.loads(str(raw["meta"]))
    return data
