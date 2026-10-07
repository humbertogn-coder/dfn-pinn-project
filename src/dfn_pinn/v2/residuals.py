"""Nondimensional DFN residuals for the v2 PINN.

All residuals are scaled so that an O(1) value corresponds to an error of the
size of the dominant physical term (see docs/V2_DESIGN.md, section 3).
"""

from __future__ import annotations

import torch

from . import constitutive as cv
from .params import FARADAY


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


class Residuals:
    def __init__(self, model, kinetics="inverse"):
        """kinetics: "inverse" (eta - asinh residual), "direct" (j - sinh residual) or
        "hard" (phi_e inside the electrodes is DEFINED by inverse BV,
        phi_e = phi_s - U(theta_s) - 2 V_T asinh(j / 2 j0); no kinetic residual)."""
        if kinetics not in ("inverse", "direct", "hard"):
            raise ValueError("kinetics must be inverse, direct or hard")
        self.m = model
        self.sc = model.sc
        self.c = model.sc.cell
        self.kinetics = kinetics
        c, sc = self.c, self.sc
        self.Lambda = sc.electrolyte_diffusion_number          # t_end D_ref / L^2
        self.S_e = sc.electrolyte_source                         # salt source scale
        self.K_fac = c.V_T / (c.L * sc.i_ref)                    # kappa -> K_hat factor
        self.eps = {"n": c.eps_n, "s": c.eps_s, "p": c.eps_p}
        self.salt_scale = None   # set by the trainer to enable the salt-inventory term
        self.soft_current = False  # set by the trainer when projection is off

    # ----------------------------------------------------------- electrolyte
    def electrolyte(self, region, X, t, side1, side2, aj_hat=None, phie=None, ce=None):
        """Return dict with c_hat, phi_e, i_hat, flux_hat and the two PDE residuals.

        X and t must require grad. ``aj_hat`` = a j L / i_ref (charge source),
        None in the separator. ``phie``/``ce`` may be supplied (as functions of
        X in the autograd graph) instead of the networks.
        """
        c, m = self.c, self.m
        eps = self.eps[region]
        ce = m.ce(X, t, side1, side2) if ce is None else ce
        phie = m.phie(X, t, side1, side2) if phie is None else phie
        ce_X, ce_t = grad(ce, X), grad(ce, t)
        phie_X = grad(phie, X)
        c_phys = ce * c.c_e0
        D_hat = cv.electrolyte_diffusivity(c_phys) / self.sc.D_e_ref * m.mult("D_e")
        flux_hat = eps ** c.brug * D_hat * ce_X                   # = -N L / (D_ref c0)
        K_hat = eps ** c.brug * cv.electrolyte_conductivity(c_phys) * self.K_fac * m.mult("kappa_e")
        ce_safe = torch.clamp(ce, min=1e-3)
        i_hat = -K_hat * (phie_X / c.V_T - 2 * (1 - c.t_plus) * c.tdf * ce_X / ce_safe)
        out = {"ce": ce, "phie": phie, "flux_hat": flux_hat, "i_hat": i_hat}
        if aj_hat is None:
            src_mass, src_charge = 0.0, 0.0
        else:
            src_charge = aj_hat
            src_mass = (1 - c.t_plus) * self.sc.t_end * aj_hat * self.sc.i_ref / (c.L * FARADAY * c.c_e0)
        out["r_mass"] = (eps * ce_t - self.Lambda * grad(flux_hat, X) - src_mass) / self.S_e
        out["r_charge"] = grad(i_hat, X) - src_charge
        return out

    def electrolyte_trace(self, X, t, side1, side2, phie=None):
        """Boundary/interface traces (no PDE): c, phi_e, flux_hat, i_hat."""
        c, m = self.c, self.m
        ce = m.ce(X, t, side1, side2)
        phie = m.phie(X, t, side1, side2) if phie is None else phie
        ce_X, phie_X = grad(ce, X), grad(phie, X)
        c_phys = ce * c.c_e0
        region_eps = torch.where(side2 > 0.5, torch.full_like(X, c.eps_p),
                                 torch.where(side1 > 0.5, torch.full_like(X, c.eps_s),
                                             torch.full_like(X, c.eps_n)))
        D_hat = cv.electrolyte_diffusivity(c_phys) / self.sc.D_e_ref * m.mult("D_e")
        flux_hat = region_eps ** c.brug * D_hat * ce_X
        K_hat = region_eps ** c.brug * cv.electrolyte_conductivity(c_phys) * self.K_fac * m.mult("kappa_e")
        i_hat = -K_hat * (phie_X / c.V_T - 2 * (1 - c.t_plus) * c.tdf * ce_X / torch.clamp(ce, min=1e-3))
        return {"ce": ce, "phie": phie, "flux_hat": flux_hat, "i_hat": i_hat}

    # ----------------------------------------------------------- electrodes
    def _constants(self, k):
        c = self.c
        if k == "n":   # a (specific area) carries the loss-of-active-material multiplier
            return dict(a=c.a_n * self.m.mult("eps_am_n"), Lk=c.L_n, sigma=c.sigma_n, R=c.R_n, D=c.D_n, cmax=c.cmax_n,
                        j_ref=self.sc.j_ref_n, X0=0.0, X1=c.X1)
        return dict(a=c.a_p * self.m.mult("eps_am_p"), Lk=c.L_p, sigma=c.sigma_p, R=c.R_p, D=c.D_p, cmax=c.cmax_p,
                    j_ref=self.sc.j_ref_p, X0=c.X2, X1=1.0)

    def electrode_fields(self, k, X, t):
        """All electrode fields as functions of the GLOBAL leaf X (requires grad) and t.

        Returns xk, j, theta_bar, theta_s, ce (normalized), phis, phie (network or,
        for kinetics="hard", defined by inverse Butler-Volmer), j0, U and the side indicators.
        """
        c, m = self.c, self.m
        K = self._constants(k)
        xk = (X - K["X0"]) / (K["X1"] - K["X0"])
        side1, side2 = m.sides(k, X)
        j, tb = m.current_and_mean(k, xk, t)
        ce = m.ce(X, t, side1, side2)
        phis = m.phis(k, xk, t)
        theta_s = m.theta(k, torch.ones_like(xk), xk, t, j=j, tb=tb)
        U = cv.ocp(theta_s, k)
        j0 = cv.exchange_current(ce * c.c_e0, theta_s, k, c) * m.mult("k_" + k)
        if self.kinetics == "hard":
            phie = phis - U - cv.bv_overpotential(j, j0, c.T)
        else:
            phie = m.phie(X, t, side1, side2)
        return {"xk": xk, "j": j, "tb": tb, "theta_s": theta_s, "ce": ce, "phis": phis, "phie": phie,
                "U": U, "j0": j0, "side1": side1, "side2": side2, "K": K}

    def electrode(self, k, xk, t):
        """Electrolyte PDEs inside electrode k plus solid charge, inventory and kinetics.

        ``xk`` (local coordinate, no grad needed) is converted to the global leaf X.
        """
        c, m, sc = self.c, self.m, self.sc
        X = m.electrode_X(k, xk.detach()).requires_grad_(True)
        f = self.electrode_fields(k, X, t)
        K, j, tb, theta_s, phis = f["K"], f["j"], f["tb"], f["theta_s"], f["phis"]
        aj_hat = K["a"] * j * c.L / sc.i_ref
        el = self.electrolyte(k, X, t, f["side1"], f["side2"], aj_hat=aj_hat, phie=f["phie"], ce=f["ce"])

        # solid phase, first-order form: dphi_s/dx = -(i_app - i_e)/sigma
        phis_x = grad(phis, f["xk"])
        sigma = K["sigma"] * (m.mult("sigma_p") if k == "p" else 1.0)
        r_solid = phis_x / (K["Lk"] * sc.i_ref / sigma) + (m.g(t) - el["i_hat"])

        # particle: surface flux is exact (model.theta); inventory residual unless derived
        if m.inventory == "derived":
            r_inv = None
        else:
            r_inv = (grad(tb, t) + 3 * j * sc.t_end / (FARADAY * K["R"] * K["cmax"])) / sc.theta_rate(k)

        eta = phis - el["phie"] - f["U"]
        if self.kinetics == "inverse":
            r_kin = (eta - cv.bv_overpotential(j, f["j0"], c.T)) / c.V_T
        elif self.kinetics == "direct":
            r_kin = (j - cv.bv_current(eta, f["j0"], c.T)) / K["j_ref"]
        else:
            r_kin = None          # exact by construction
        return {"r_mass": el["r_mass"], "r_charge": el["r_charge"], "r_solid": r_solid,
                "r_inv": r_inv, "r_kin": r_kin, "j": j, "eta": eta, "theta_s": theta_s, "phie": el["phie"]}

    def electrode_trace(self, k, X, t):
        """Electrolyte trace at an electrode boundary/interface, consistent with electrode()."""
        f = self.electrode_fields(k, X, t)
        return self.electrolyte_trace(X, t, f["side1"], f["side2"], phie=f["phie"])

    def separator(self, X, t):
        side1, side2 = self.m.sides("s", X)
        el = self.electrolyte("s", X, t, side1, side2, aj_hat=None)
        return {"r_mass": el["r_mass"], "r_charge": el["r_charge"]}

    # ----------------------------------------------------------- particles
    def particle(self, k, s, xk, t):
        """Reference implementation (one (x,t) pair per radial point)."""
        sc = self.sc
        theta = self.m.theta(k, s, xk, t)
        th_s = grad(theta, s)
        th_ss = grad(th_s, s)
        th_t = grad(theta, t)
        lap = 6 * th_s + 4 * s * th_ss
        # div(D(theta) grad theta) = D lap + D'(theta) |grad theta|^2, with |d theta/d rho|^2 = 4 s theta_s^2
        Dhat = self.m.D_factor(k, theta)
        dDhat = self._dD_factor(k)
        return (th_t - sc.particle_number(k) * self.m.mult("D_" + k) * (Dhat * lap + dDhat * 4 * s * th_s ** 2)) / sc.theta_rate(k)

    def _dD_factor(self, k):
        c = self.c
        return (c.theta_M_n * c.cmax_n) if k == "n" else (c.theta_M_p * c.cmax_p)

    def particle_grouped(self, k, s, xk, t):
        """Same residual, with n_r radial points sharing each (x,t) pair.

        s: (P, n_r), xk and t: (P, 1). Per-pair parts of model.theta (theta_bar,
        j, mean(h), h_s(1)) are evaluated once per pair and differentiated in t
        explicitly, which is ~n_r times cheaper than ``particle``.
        """
        m, sc = self.m, self.sc
        P, nr = s.shape
        j, tb = m.current_and_mean(k, xk, t)
        n, q = P, m.r_nodes.shape[0]
        sq = (m.r_nodes.to(t.dtype) ** 2).view(1, q).expand(n, q).reshape(-1, 1)
        hq = m._h(k, sq, xk.expand(n, q).reshape(-1, 1), t.expand(n, q).reshape(-1, 1)).view(n, q)
        hbar = (hq * m.r_weights.to(t.dtype).view(1, q)).sum(1, keepdim=True)
        s1 = torch.ones_like(t).requires_grad_(True)
        h1s = grad(m._h(k, s1, xk, t), s1)
        gW = m.g(t) * m.w_scale[k]
        Pk = -m.parabola[k] / (m.mult("D_" + k) * m.D_factor(k, tb)) * j  # coefficient of (s - 3/5)
        # time derivatives of per-pair quantities
        tb_t, P_t, hbar_t, h1s_t, gW_t = (grad(v, t) for v in (tb, Pk, hbar, h1s, gW))
        # per-point network evaluations
        sp = s.reshape(-1, 1).detach().requires_grad_(True)
        xp = xk.detach().expand(P, nr).reshape(-1, 1)
        tp = t.detach().expand(P, nr).reshape(-1, 1).requires_grad_(True)
        h = m._h(k, sp, xp, tp)
        h_s = grad(h, sp)
        h_ss = grad(h_s, sp)
        h_t = grad(h, tp)
        rs = lambda v: v.view(P, nr)  # noqa: E731
        sm = s - 0.6
        bracket = rs(h) - hbar - h1s * sm
        th_t = tb_t + P_t * sm + gW_t * bracket + gW * (rs(h_t) - hbar_t - h1s_t * sm)
        th_s = Pk + gW * (rs(h_s) - h1s)
        th_ss = gW * rs(h_ss)
        lap = 6 * th_s + 4 * s * th_ss
        Q = sc.particle_number(k) * m.mult("D_" + k)
        theta = tb + Pk * sm + gW * bracket
        Dhat = m.D_factor(k, theta)
        div = Dhat * lap + self._dD_factor(k) * 4 * s * th_s ** 2
        return ((th_t - Q * div) / sc.theta_rate(k)).reshape(-1, 1)

    # ----------------------------------------------------------- boundaries
    def boundaries(self, t):
        c = self.c
        out = {}
        one, zero = torch.ones_like(t), torch.zeros_like(t)
        X0 = torch.zeros_like(t).requires_grad_(True)
        X1 = torch.ones_like(t).requires_grad_(True)
        hard = self.kinetics == "hard"
        left = self.electrode_trace("n", X0, t) if hard else self.electrolyte_trace(X0, t, zero, zero)
        right = self.electrode_trace("p", X1, t) if hard else self.electrolyte_trace(X1, t, one, one)
        scale = self.Lambda / self.S_e
        out["bc_flux_0"] = left["flux_hat"] * scale
        out["bc_flux_L"] = right["flux_hat"] * scale
        out["bc_ie_0"] = left["i_hat"]
        out["bc_ie_L"] = right["i_hat"]
        for name, pos, sl, sr, k_el in (("1", c.X1, (zero, zero), (one, zero), "n"),
                                        ("2", c.X2, (one, zero), (one, one), "p")):
            Xi = torch.full_like(t, pos).requires_grad_(True)
            if hard and k_el == "n":
                L_ = self.electrode_trace("n", Xi, t)
                R_ = self.electrolyte_trace(Xi, t, *sr)
                out[f"if{name}_phie"] = (L_["phie"] - R_["phie"]) / c.V_T
            elif hard:
                L_ = self.electrolyte_trace(Xi, t, *sl)
                R_ = self.electrode_trace("p", Xi, t)
                out[f"if{name}_phie"] = (L_["phie"] - R_["phie"]) / c.V_T
            else:
                L_ = self.electrolyte_trace(Xi, t, *sl)
                R_ = self.electrolyte_trace(Xi, t, *sr)
            out[f"if{name}_flux"] = (L_["flux_hat"] - R_["flux_hat"]) * scale
            out[f"if{name}_ie"] = L_["i_hat"] - R_["i_hat"]
        return out

    def electrode_current(self, k, t):
        """Soft electrode-integrated current residual (integral a j dx - J_k)/i_app.

        Used only when the hard projection is off (variant C/D of the planned
        factorial). With the projection on this is identically zero.
        """
        c, m = self.c, self.m
        n, q = t.shape[0], m.q_nodes.shape[0]
        xq = m.q_nodes.to(t.dtype).view(1, q).expand(n, q).reshape(-1, 1)
        tq = t.expand(n, q).reshape(-1, 1)
        jq = m.j(k, xq, tq).view(n, q)
        a, Lk, sign = (c.a_n * m.mult("eps_am_n"), c.L_n, 1.0) if k == "n" else (c.a_p * m.mult("eps_am_p"), c.L_p, -1.0)
        integral = a * Lk * (jq * m.q_weights.to(t.dtype).view(1, q)).sum(1, keepdim=True)
        return integral / self.sc.i_ref - sign * m.g(t)

    def salt_inventory(self, t, scale_mol_m3=1.0):
        """Global salt conservation: integral of eps*c_e over the cell is constant.

        Exact for the continuous problem (no salt flux at the collectors and,
        with the hard current projection, zero net source). Per-region
        Gauss-Legendre quadrature (the fields have kinks at the interfaces).
        Returned in units of ``scale_mol_m3`` of cell-averaged concentration,
        so the local PDE-residual normalization (by the source scale S_e) no
        longer hides a slow drift of the mean.
        """
        c, m = self.c, self.m
        n, q = t.shape[0], m.q_nodes.shape[0]
        nodes, weights = m.q_nodes.to(t.dtype), m.q_weights.to(t.dtype)
        total, eps_total = 0.0, 0.0
        for (a, b), eps, sides in (((0.0, c.X1), c.eps_n, (0.0, 0.0)),
                                   ((c.X1, c.X2), c.eps_s, (1.0, 0.0)),
                                   ((c.X2, 1.0), c.eps_p, (1.0, 1.0))):
            X = (a + (b - a) * nodes).view(1, q).expand(n, q).reshape(-1, 1)
            tq = t.expand(n, q).reshape(-1, 1)
            s1, s2 = torch.full_like(X, sides[0]), torch.full_like(X, sides[1])
            ce = m.ce(X, tq, s1, s2).view(n, q)
            total = total + eps * (b - a) * (ce * weights.view(1, q)).sum(1, keepdim=True)
            eps_total += eps * (b - a)
        return (total / eps_total - 1.0) * c.c_e0 / scale_mol_m3
