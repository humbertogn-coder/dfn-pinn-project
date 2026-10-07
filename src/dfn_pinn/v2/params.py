"""Cell parameters (Chen2020 / LG M50), protocol and derived scales for v2."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import math

FARADAY = 96485.33212331001  # C/mol
GAS_CONSTANT = 8.31446261815324  # J/(mol K)


@dataclass
class CellParams:
    """Isothermal DFN parameters. Defaults reproduce PyBaMM ``Chen2020``."""

    L_n: float = 85.2e-6
    L_s: float = 12e-6
    L_p: float = 75.6e-6
    eps_n: float = 0.25          # electrolyte volume fractions
    eps_s: float = 0.47
    eps_p: float = 0.335
    eps_am_n: float = 0.75       # active material volume fractions
    eps_am_p: float = 0.665
    R_n: float = 5.86e-6
    R_p: float = 5.22e-6
    D_n: float = 3.3e-14
    D_p: float = 4e-15
    # stress-enhanced ("stress-induced") diffusion, PyBaMM / Ai et al. 2019 eq. 12: D_k(c) = D_k (1 + theta_M_k c),
    # theta_M = Omega/(RT) 2 Omega E/(9(1-nu)). 0 = plain Fickian (Chen2020). OKane2022 mechanical parameters give
    # 1.846e-5 (graphite, factor 1.2-1.5) and 6.566e-3 m3/mol (NMC, factor 110-370); PyBaMM switches it on by
    # default whenever a particle-mechanics submodel is used (V2_RESULTS.md section 8).
    theta_M_n: float = 0.0
    theta_M_p: float = 0.0
    cmax_n: float = 33133.0
    cmax_p: float = 63104.0
    sigma_n: float = 215.0       # effective solid conductivity (Bruggeman 0 in Chen2020)
    sigma_p: float = 0.18
    k_n: float = 6.48e-7         # exchange-current prefactors, (A/m2)(m3/mol)^1.5
    k_p: float = 3.42e-6
    brug: float = 1.5            # electrolyte Bruggeman exponent in all regions
    t_plus: float = 0.2594
    tdf: float = 1.0             # thermodynamic factor
    T: float = 298.15
    c_e0: float = 1000.0
    c_n0: float = 29866.0        # initial solid concentrations (SOC = 1)
    c_p0: float = 17038.0
    area: float = 0.065 * 1.58   # electrode area [m2]
    capacity_Ah: float = 5.0

    # ------------------------------------------------------------------ derived
    @property
    def L(self) -> float:
        return self.L_n + self.L_s + self.L_p

    @property
    def X1(self) -> float:
        return self.L_n / self.L

    @property
    def X2(self) -> float:
        return (self.L_n + self.L_s) / self.L

    @property
    def a_n(self) -> float:
        return 3 * self.eps_am_n / self.R_n

    @property
    def a_p(self) -> float:
        return 3 * self.eps_am_p / self.R_p

    @property
    def theta_n0(self) -> float:
        return self.c_n0 / self.cmax_n

    @property
    def theta_p0(self) -> float:
        return self.c_p0 / self.cmax_p

    @property
    def V_T(self) -> float:
        return GAS_CONSTANT * self.T / FARADAY

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Protocol:
    """Galvanostatic discharge I(t) = I * tanh(t / ramp_s); I > 0 is discharge.

    The smooth ramp removes the incompatible t = 0 corner (uniform particle
    profile with an instantaneous nonzero surface flux). The same ramp is used
    in the PyBaMM reference so that the two problems are identical.
    """

    current_A: float = 5.0
    ramp_s: float = 30.0
    t_end_s: float = 3000.0

    def current(self, t_s):
        """Works with floats, numpy arrays and torch tensors."""
        try:
            import torch
            if isinstance(t_s, torch.Tensor):
                return self.current_A * torch.tanh(t_s / self.ramp_s)
        except ImportError:  # pragma: no cover
            pass
        import numpy as np
        return self.current_A * np.tanh(np.asarray(t_s) / self.ramp_s)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scales:
    """Characteristic scales used to nondimensionalize residuals and outputs."""

    cell: CellParams = field(default_factory=CellParams)
    protocol: Protocol = field(default_factory=Protocol)

    @property
    def t_end(self) -> float:
        return self.protocol.t_end_s

    @property
    def i_ref(self) -> float:
        """Geometric current density at the plateau of the ramp [A/m2]."""
        return self.protocol.current_A / self.cell.area

    @property
    def j_ref_n(self) -> float:
        c = self.cell
        return self.i_ref / (c.a_n * c.L_n)

    @property
    def j_ref_p(self) -> float:
        c = self.cell
        return self.i_ref / (c.a_p * c.L_p)

    def particle_number(self, k: str) -> float:
        """Q = D * t_end / R^2 (dimensionless diffusion number)."""
        c = self.cell
        return (c.D_n if k == "n" else c.D_p) * self.t_end / (c.R_n if k == "n" else c.R_p) ** 2

    def theta_rate(self, k: str) -> float:
        """Mean |d theta / d t_hat| at the plateau current: 3 j t_end / (F R cmax)."""
        c = self.cell
        j = self.j_ref_n if k == "n" else self.j_ref_p
        R = c.R_n if k == "n" else c.R_p
        cmax = c.cmax_n if k == "n" else c.cmax_p
        return 3 * j * self.t_end / (FARADAY * R * cmax)

    def surface_gradient(self, k: str) -> float:
        """|d theta / d rho| at the surface at the plateau current: j R / (F D cmax)."""
        c = self.cell
        j = self.j_ref_n if k == "n" else self.j_ref_p
        R, D, cmax = ((c.R_n, c.D_n, c.cmax_n) if k == "n" else (c.R_p, c.D_p, c.cmax_p))
        return j * R / (FARADAY * D * cmax)

    @property
    def electrolyte_source(self) -> float:
        """(1-t+) t_end a j_ref / (F c_e0), a typical salt source per unit t_hat."""
        c = self.cell
        return (1 - c.t_plus) * self.t_end * c.a_n * self.j_ref_n / (FARADAY * c.c_e0)

    @property
    def D_e_ref(self) -> float:
        x = self.cell.c_e0 / 1000.0
        return 8.794e-11 * x * x - 3.972e-10 * x + 4.862e-10

    @property
    def electrolyte_diffusion_number(self) -> float:
        return self.t_end * self.D_e_ref / self.cell.L ** 2

    def summary(self) -> dict:
        return {
            "t_end_s": self.t_end, "i_ref_A_m2": self.i_ref,
            "j_ref_n": self.j_ref_n, "j_ref_p": self.j_ref_p,
            "Q_n": self.particle_number("n"), "Q_p": self.particle_number("p"),
            "theta_rate_n": self.theta_rate("n"), "theta_rate_p": self.theta_rate("p"),
            "surface_gradient_n": self.surface_gradient("n"),
            "surface_gradient_p": self.surface_gradient("p"),
            "electrolyte_source": self.electrolyte_source,
            "electrolyte_diffusion_number": self.electrolyte_diffusion_number,
            "V_T": self.cell.V_T,
        }


def _finite(x: float) -> bool:  # pragma: no cover - trivial helper
    return math.isfinite(x)
