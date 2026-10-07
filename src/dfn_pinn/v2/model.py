"""Field networks for the v2 DFN PINN.

Coordinates passed to every field are physical-but-normalized:

* ``t``   : t_hat = t / t_end in [0, 1]
* ``xk``  : local electrode coordinate in [0, 1] (n: 0 = collector, 1 = separator;
            p: 0 = separator, 1 = collector)
* ``X``   : global x / L in [0, 1] (electrolyte fields)
* ``s``   : rho^2 = (r / R)^2 in [0, 1] (particle fields; using rho^2 as the
            input imposes d theta / d rho = 0 at the centre by construction)

Every network output is multiplied by a physical scale (``Scales``) so that
the raw outputs are O(1) at the solution. Initial conditions are imposed
exactly by a time factor that vanishes at t = 0.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .constitutive import ocp_n, ocp_p
from .params import Scales


class MLP(nn.Module):
    def __init__(self, n_in, width=64, depth=4, n_out=1, act="tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}
        layers, d = [], n_in
        for _ in range(depth):
            lin = nn.Linear(d, width)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, acts[act]()]
            d = width
        out = nn.Linear(d, n_out)
        nn.init.xavier_normal_(out.weight, gain=0.1)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def _logcosh(x):
    """Numerically safe log(cosh(x)) for x >= 0."""
    return x + torch.log1p(torch.exp(-2 * x)) - math.log(2.0)


def gauss_legendre(n, dtype=torch.float64):
    """Nodes/weights on [0, 1] (weights sum to 1)."""
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor(0.5 * (x + 1), dtype=dtype), torch.tensor(0.5 * w, dtype=dtype)


class DFNPINN(nn.Module):
    """All DFN fields. ``variant`` selects the kinetics/current options.

    variant keys
      kinetics   : "inverse" (eta - asinh form, default) or "direct" (j - sinh form)
      projection : True -> hard electrode-integrated current projection of j
    """

    def __init__(self, scales: Scales, width=64, depth=4, act="tanh",
                 projection=True, quad_order=24, ic_tau_s=60.0, inventory="soft",
                 fourier_t=0, width_scalar=None, short_t=(), collector_bc="soft", fourier_period=1.0):
        super().__init__()
        self.sc = scales
        c = scales.cell
        self.projection = projection
        if collector_bc not in ("soft", "hard"):
            raise ValueError("collector_bc must be 'soft' or 'hard'")
        self.collector_bc = collector_bc       # "hard": electrolyte x-features with zero slope at X = 0, 1
        # time features: 2t-1, 2g-1 and, if fourier_t = K > 0, sin/cos(2 pi k t), k = 1..K
        self.fourier_t = fourier_t
        self.fourier_period = float(fourier_period)   # period of the k = 1 feature in units of t_end (1 = periodic on [0, 1])
        # short-time features 2 exp(-t / tau) - 1 for the early transients (tau in seconds)
        self.short_tau_hat = tuple(float(tau) / scales.t_end for tau in short_t)
        n_t = 2 + 2 * fourier_t + len(self.short_tau_hat)
        ws = width_scalar or width // 2       # width of the 1-D-in-space / scalar networks
        self.ic_tau_hat = ic_tau_s / scales.t_end
        self.ramp_hat = scales.protocol.ramp_s / scales.t_end
        # inventory="soft"   : theta_bar is a network; d theta_bar/dt = -3 j/(F R cmax) is a residual.
        # inventory="derived": theta_bar is the primary field and j := -(F R cmax / 3) d theta_bar/dt,
        #                      so the inventory, the electrode-integrated current and j(x,0) = 0 are
        #                      all exact by construction (see theta_bar / j).
        if inventory not in ("soft", "derived"):
            raise ValueError("inventory must be 'soft' or 'derived'")
        self.inventory = inventory
        self.inv_coef = {}
        for k in ("n", "p"):
            R, cm = (c.R_n, c.cmax_n) if k == "n" else (c.R_p, c.cmax_p)
            self.inv_coef[k] = 3 * scales.t_end / (96485.33212331001 * R * cm)   # d theta_bar/dt_hat = -coef * j
        # --- networks (inputs listed in the comment) -------------------------
        # particle: theta = theta_bar + parabolic(j) + zero-mean, zero-surface-slope correction
        self.net_theta_n = MLP(3 + n_t, width, depth, act=act)   # s, xk, sqrt(t), time features
        self.net_theta_p = MLP(3 + n_t, width, depth, act=act)
        self.net_thbar_n = MLP(1 + n_t, ws, depth, act=act)      # xk, time features (particle mean)
        self.net_thbar_p = MLP(1 + n_t, ws, depth, act=act)
        self.net_ce = MLP(3 + n_t, width, depth, act=act)        # X, kink1, kink2, time features
        self.net_phie = MLP(3 + n_t, width, depth, act=act)
        self.net_phisn = MLP(1 + n_t, ws, depth, act=act)        # xk, time features
        self.net_phisp = MLP(1 + n_t, ws, depth, act=act)
        self.net_V = MLP(n_t, ws, depth, act=act)                # time features
        self.net_jn = MLP(1 + n_t, width, depth, act=act)        # xk, time features
        self.net_jp = MLP(1 + n_t, width, depth, act=act)
        # --- output scales ---------------------------------------------------
        self.theta_scale = {"n": max(scales.theta_rate("n"), 0.05), "p": max(scales.theta_rate("p"), 0.05)}
        # surface-gradient scale of the particle correction; with stress-enhanced diffusion the gradients are
        # 1/D(theta0) smaller (NMC with OKane2022 mechanics: 1/113), so the scale follows D at the initial state
        self.w_scale = {k: 0.5 * scales.surface_gradient(k) / (1 + (c.theta_M_n * c.cmax_n * c.theta_n0 if k == "n"
                                                                    else c.theta_M_p * c.cmax_p * c.theta_p0))
                        for k in ("n", "p")}
        self.parabola = {}  # R / (2 F D cmax) [1/(A/m2)], see theta()
        for k in ("n", "p"):
            R, D, cm = ((c.R_n, c.D_n, c.cmax_n) if k == "n" else (c.R_p, c.D_p, c.cmax_p))
            self.parabola[k] = R / (2 * 96485.33212331001 * D * cm)
        qr, wr = gauss_legendre(quad_order)
        self.register_buffer("r_nodes", qr)
        self.register_buffer("r_weights", 3 * qr * qr * wr)   # volume weights, sum to 1
        self.ce_scale = 1.0                       # c_e / c_e0 deviation
        self.phie_scale = 0.2                     # V
        self.phisn_scale = scales.i_ref * c.L_n / c.sigma_n   # V (ohmic scale)
        self.phisp_scale = scales.i_ref * c.L_p / c.sigma_p
        self.V_scale = 0.5                        # V
        self.j_scale = {"n": scales.j_ref_n, "p": scales.j_ref_p}
        # V0 / phie0 (output offsets) are evaluated at the CURRENT initial stoichiometries, which are
        # learnable in the inverse problem (theta_n0, theta_p0 multipliers: loss of lithium inventory).
        q, w = gauss_legendre(quad_order)
        self.register_buffer("q_nodes", q)
        self.register_buffer("q_weights", w)
        # Log-multipliers of physical parameters (inverse problem). Frozen by
        # default; ``set_trainable_parameters`` unfreezes a chosen subset.
        self.log_mult = nn.ParameterDict({name: nn.Parameter(torch.zeros(()), requires_grad=False)
                                          for name in self.PARAMETERS})

    # D_e, kappa_e: scale factors of the Nyman fits. Aging parameters (2026-10-05): eps_am_n/p = active-material
    # volume fractions (loss of active material, enter through a = 3 eps_am / R), theta_n0/p0 = initial
    # stoichiometries (loss of lithium inventory), R0 = lumped series resistance [Ohm] applied to the measured
    # terminal voltage only (V_terminal = V_electrochemical - R0 * I). R0 is LINEAR: R0 = R0_SCALE * log_mult["R0"].
    PARAMETERS = ("D_n", "D_p", "k_n", "k_p", "sigma_p", "D_e", "kappa_e", "eps_am_n", "eps_am_p", "theta_n0", "theta_p0", "R0")
    R0_SCALE = 5e-3   # Ohm per unit of the "R0" parameter (param_log_bound 0.7 -> +/- 3.5 mOhm, ln 10 -> +/- 11.5 mOhm)

    def set_trainable_parameters(self, names, initial_multipliers=None):
        for name in self.PARAMETERS:
            self.log_mult[name].requires_grad_(name in names)
        for name, value in (initial_multipliers or {}).items():
            with torch.no_grad():
                self.log_mult[name].fill_(value / self.R0_SCALE if name == "R0" else math.log(value))

    def mult(self, name):
        """Multiplier of a physical parameter (exp of the log-multiplier); for "R0" the resistance in Ohm."""
        if name == "R0":
            return self.R0_SCALE * self.log_mult["R0"]
        return torch.exp(self.log_mult[name])

    def theta0(self, k):
        c = self.sc.cell
        return (c.theta_n0 if k == "n" else c.theta_p0) * self.mult("theta_n0" if k == "n" else "theta_p0")

    @property
    def V0(self):
        return ocp_p(self.theta0("p")) - ocp_n(self.theta0("n"))

    @property
    def phie0(self):
        return -ocp_n(self.theta0("n"))

    def terminal_voltage(self, t):
        """Measured voltage: electrochemical voltage minus the lumped ohmic drop R0 * I(t) (I > 0 discharge)."""
        return self.voltage(t) - self.mult("R0") * self.sc.protocol.current_A * self.g(t)

    def parameter_values(self):
        c = self.sc.cell
        base = {"D_n": c.D_n, "D_p": c.D_p, "k_n": c.k_n, "k_p": c.k_p, "sigma_p": c.sigma_p, "D_e": 1.0, "kappa_e": 1.0,
                "eps_am_n": c.eps_am_n, "eps_am_p": c.eps_am_p, "theta_n0": c.theta_n0, "theta_p0": c.theta_p0, "R0": 1.0}
        return {k: base[k] * float(self.mult(k).detach()) for k in self.PARAMETERS}

    # ------------------------------------------------------------- helpers
    def g(self, t):
        """Normalized applied current I(t)/I_plateau = tanh(t / ramp)."""
        return torch.tanh(t / self.ramp_hat)

    def ic_factor(self, t):
        """Vanishes at t = 0; ~1 after a few ic_tau. Imposes the IC exactly."""
        return 1 - torch.exp(-t / self.ic_tau_hat)

    @staticmethod
    def _c(t):
        return 2 * t - 1

    def tf(self, t):
        """Time features: [2t-1, 2g-1, sin(2 pi k t), cos(2 pi k t) (k = 1..fourier_t)]."""
        feats = [self._c(t), self._c(self.g(t))]
        for k in range(1, self.fourier_t + 1):
            w = 2 * math.pi * k / self.fourier_period
            feats += [torch.sin(w * t), torch.cos(w * t)]
        for tau in self.short_tau_hat:
            feats.append(2 * torch.exp(-t / tau) - 1)
        return torch.cat(feats, dim=1)

    # ------------------------------------------------------------- fields
    def theta_bar(self, k, xk, t):
        """Volume-averaged particle stoichiometry; IC exact.

        derived: theta_bar = G_k(t) + rate_k * t * g(t) * (eta(x,t) - mean_x eta), where
        G_k(t) = theta0 -/+ coef_k * j_ref_k * ramp * ln cosh(t / ramp) is the exact mean
        stoichiometry imposed by the applied current and the spatial part has zero
        electrode mean (fixed Gauss-Legendre quadrature in x).
        """
        theta0 = self.theta0(k)
        if self.inventory == "derived":
            sign = 1.0 if k == "n" else -1.0      # sign of j in discharge
            G = theta0 - sign * self.inv_coef[k] * self.j_scale[k] * self.ramp_hat * _logcosh(t / self.ramp_hat)
            eta = self.j_raw(k, xk, t)
            n, q = t.shape[0], self.q_nodes.shape[0]
            xq = self.q_nodes.to(t.dtype).view(1, q).expand(n, q).reshape(-1, 1)
            eta_q = self.j_raw(k, xq, t.expand(n, q).reshape(-1, 1)).view(n, q)
            eta_mean = (eta_q * self.q_weights.to(t.dtype).view(1, q)).sum(1, keepdim=True)
            return G + self.theta_scale[k] * t * self.g(t) * (eta - eta_mean)
        net = self.net_thbar_n if k == "n" else self.net_thbar_p
        z = torch.cat([2 * xk - 1, self.tf(t)], dim=1)
        sign = -1.0 if k == "n" else 1.0          # discharge: n delithiates, p lithiates
        return theta0 + t * self.theta_scale[k] * (sign + net(z))

    def _h(self, k, s, xk, t):
        net = self.net_theta_n if k == "n" else self.net_theta_p
        z = torch.cat([2 * s - 1, 2 * xk - 1, self._c(torch.sqrt(t + 1e-4)), self.tf(t)], dim=1)
        return net(z)

    def current_and_mean(self, k, xk, t):
        """(j, theta_bar) at (x, t) with one evaluation of the mean network."""
        if self.inventory == "derived":
            if not t.requires_grad:
                with torch.enable_grad():
                    t2 = t.detach().clone().requires_grad_(True)
                    tb = self.theta_bar(k, xk, t2)
                    dtb = torch.autograd.grad(tb, t2, torch.ones_like(tb), create_graph=torch.is_grad_enabled())[0]
                return -dtb / self.inv_coef[k], tb.detach() if not torch.is_grad_enabled() else tb
            tb = self.theta_bar(k, xk, t)
            dtb = torch.autograd.grad(tb, t, torch.ones_like(tb), create_graph=True)[0]
            return -dtb / self.inv_coef[k], tb
        return self.j(k, xk, t), self.theta_bar(k, xk, t)

    def D_factor(self, k, theta):
        """Stress-enhanced diffusivity factor D(theta)/D = 1 + theta_M cmax theta (1 when theta_M = 0)."""
        c = self.sc.cell
        thM, cmax = (c.theta_M_n, c.cmax_n) if k == "n" else (c.theta_M_p, c.cmax_p)
        if thM == 0.0:
            return torch.ones_like(theta)
        return 1 + thM * cmax * theta

    def theta(self, k, s, xk, t, j=None, tb=None):
        """Particle stoichiometry at s = rho^2.

        theta = theta_bar(x,t) - j R/(2 F D cmax) (s - 3/5) + g(t) W [h - mean(h) - h_s(1)(s - 3/5)]

        * the volume mean of theta is exactly theta_bar (inventory variable);
        * -D cmax/R d theta/d rho at rho = 1 equals j/F exactly (flux BC is hard);
        * d theta/d rho = 0 at the centre (s = rho^2 input);
        * theta(t = 0) = theta0 exactly (theta_bar(0) = theta0, j(0) = 0, g(0) = 0).
        """
        if j is None or tb is None:
            j, tb = self.current_and_mean(k, xk, t)
        # flux condition with the diffusivity at the particle mean (D(theta_surf) differs by < 1 % for graphite,
        # and NMC with theta_M_p > 0 is nearly uniform)
        parab = -self.parabola[k] / (self.mult("D_" + k) * self.D_factor(k, tb)) * j * (s - 0.6)
        n, q = s.shape[0], self.r_nodes.shape[0]
        sq = (self.r_nodes.to(s.dtype) ** 2).view(1, q).expand(n, q).reshape(-1, 1)
        hq = self._h(k, sq, xk.expand(n, q).reshape(-1, 1), t.expand(n, q).reshape(-1, 1)).view(n, q)
        hbar = (hq * self.r_weights.to(s.dtype).view(1, q)).sum(1, keepdim=True)
        s1 = torch.ones_like(s).requires_grad_(True)
        h1 = self._h(k, s1, xk, t)
        h1_s = torch.autograd.grad(h1, s1, torch.ones_like(h1), create_graph=True)[0]
        w = self._h(k, s, xk, t) - hbar - h1_s * (s - 0.6)
        return tb + parab + self.g(t) * self.w_scale[k] * w

    def _kinks(self, X, side1, side2):
        c = self.sc.cell
        return (X - c.X1) * side1 / (1 - c.X1), (X - c.X2) * side2 / (1 - c.X2)

    def _xe(self, X, side1, side2):
        """Electrolyte x-features: base coordinate plus one kink per interface (C0, slope jump).

        collector_bc = "hard": cos(pi X) and sin(pi/2 k) kinks, all with zero x-derivative at the
        two collectors, so d c_e/dx = d phi_e/dx = 0 at X = 0 and X = 1 hold for any network
        (zero salt flux and zero electrolyte current are then exact; the soft bc_* terms vanish).
        """
        k1, k2 = self._kinks(X, side1, side2)
        if self.collector_bc == "hard":
            return torch.cat([torch.cos(math.pi * X), torch.sin(0.5 * math.pi * k1), torch.sin(0.5 * math.pi * k2)], dim=1)
        return torch.cat([2 * X - 1, 2 * k1, 2 * k2], dim=1)

    def ce(self, X, t, side1, side2):
        z = torch.cat([self._xe(X, side1, side2), self.tf(t)], dim=1)
        return 1 + self.ic_factor(t) * self.ce_scale * self.net_ce(z)

    def phie(self, X, t, side1, side2):
        z = torch.cat([self._xe(X, side1, side2), self.tf(t)], dim=1)
        return self.phie0 + self.phie_scale * self.net_phie(z)

    def phis(self, k, xk, t):
        z = torch.cat([2 * xk - 1, self.tf(t)], dim=1)
        if k == "n":   # phi_s,n(0) = 0 (gauge) at the negative collector xk = 0
            return xk * self.phisn_scale * self.net_phisn(z)
        # phi_s,p = V(t) at the positive collector xk = 1
        return self.voltage(t) + (1 - xk) * self.phisp_scale * self.net_phisp(z)

    def voltage(self, t):
        z = self.tf(t)
        return self.V0 + self.V_scale * self.net_V(z)

    def j_raw(self, k, xk, t):
        net = self.net_jn if k == "n" else self.net_jp
        z = torch.cat([2 * xk - 1, self.tf(t)], dim=1)
        return net(z)

    def j(self, k, xk, t):
        """Interfacial current density [A/m2]; oxidation positive.

        With projection, the electrode integral of a*j equals +/- i_app(t)
        exactly (fixed Gauss-Legendre quadrature in x at each query time).
        With inventory="derived", j = -(d theta_bar/dt_hat) / coef (autograd in t).
        """
        if self.inventory == "derived":
            if t.requires_grad:
                tb = self.theta_bar(k, xk, t)
                dtb = torch.autograd.grad(tb, t, torch.ones_like(tb), create_graph=True)[0]
            else:
                with torch.enable_grad():
                    t2 = t.detach().clone().requires_grad_(True)
                    tb = self.theta_bar(k, xk, t2)
                    dtb = torch.autograd.grad(tb, t2, torch.ones_like(tb), create_graph=torch.is_grad_enabled())[0]
            return -dtb / self.inv_coef[k]
        sign = 1.0 if k == "n" else -1.0
        # the electrode integral of a_eff * j must equal the applied current: a_eff = a * mult(eps_am)
        mean_part = sign * self.j_scale[k] / self.mult("eps_am_" + k)
        raw = self.j_raw(k, xk, t)
        if self.projection:
            n, q = t.shape[0], self.q_nodes.shape[0]
            xq = self.q_nodes.to(t.dtype).view(1, q).expand(n, q).reshape(-1, 1)
            tq = t.expand(n, q).reshape(-1, 1)
            raw_q = self.j_raw(k, xq, tq).view(n, q)
            mean = (raw_q * self.q_weights.to(t.dtype).view(1, q)).sum(1, keepdim=True)
            raw = raw - mean
        return self.g(t) * (mean_part + self.j_scale[k] * raw)   # j(x, 0) = 0 exactly

    def electrode_X(self, k, xk):
        c = self.sc.cell
        if k == "n":
            return xk * c.X1
        return c.X2 + xk * (1 - c.X2)

    @staticmethod
    def sides(k, X):
        """Kink side indicators for points inside an electrode/separator."""
        one, zero = torch.ones_like(X), torch.zeros_like(X)
        if k == "n":
            return zero, zero
        if k == "s":
            return one, zero
        return one, one
