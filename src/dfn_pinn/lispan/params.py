"""Parameters of the Li-SPAN model (Simanjuntak et al. 2024, Tables 1-3; SI Table S1).

All quantities in SI units.  The defaults are the *validated cell* of the paper (Figs 4-7): 100 um
SPAN cathode (9.45 vol-% SPAN), two Whatman glass-fibre separators (618 um), Li foil anode, 1 M LiPF6
in EC:DEC.  Two quantities are not fixed unambiguously by paper + SI and are exposed as explicit
knobs (see docs/LI_SPAN_MODEL_FORMULATION.md section 8):

* ``c_S_ref``: reference concentration of S2- in the SPAN reaction kinetics (activities a = c/c_ref).
  The text says "initial condition" (0.01 mol/m3, Table 2); this value also places the model OCV
  break points of Fig. 3 (1.97 V and 1.73 V) correctly.
* ``c_S_sat``: S2- concentration in equilibrium with Li2S for the nominal K_sp = 10.  Figs 5b, 7e
  and 7f of the paper show 1e-5 mol/m3 (and a saturation proportional to K_sp), which is not
  K_sp * c_S_ref; the Li2S driving force is therefore written with its own reference
  c_S_sat / K_sp (1e-6 mol/m3).
* ``k0``: effective frequency factors.  With Table 3's values (1e-2, 1e-2, 1e-4 mol/m2/s) and
  a_SPAN = 1e7 1/m the exchange rates exceed the demand by ~1e6 and the SPAN reactions would sit at
  equilibrium at every rate, whereas the paper's Fig. 6a shows a Tafel slope of 0.118 V per decade of
  k0 and 0.17 V between 0.1 C and 1 C: the published simulations operate in the Tafel regime.  Only
  the products a_SPAN * k0 enter the kinetics, so the defaults below are *effective* values fitted to
  the paper's Figs 5a/6a (scripts/lispan_fit_paper.py); Table 3's values are kept in ``k0_table3``.
* ``reversible``: with the full mass-action form the reverse of reaction (3) (S2- + PAN-SLi ->
  PAN-S2Li + e-) runs strongly during discharge phases 1-2, where dphi is 0.5-0.9 V above U_3, and
  consumes PAN-SLi together with all S2- (an electrode-mediated comproportionation S3 + S1 -> 2 S2).
  The paper's Fig. 5a shows none of this (c_S1 stays at 598 mol/m3 until reaction (3) starts and S2-
  is in equilibrium with the reversible reaction (2), Fig. 7e), so by default reaction (3) is treated
  as irreversible in discharge (its reverse is the charging reaction, out of the paper's scope).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math

F = 96485.33212
R = 8.314462618


@dataclass
class LiSPANParams:
    # --- geometry (Table 1)
    L_cat: float = 100e-6
    L_sep: float = 618e-6
    # --- cathode phases (Table 1)
    eps_SPAN: float = 0.0945
    eps_CB: float = 0.032
    eps_L0: float = 1e-5                 # initial Li2S volume fraction
    a_SPAN0: float = 1e7                 # specific SPAN surface area [1/m]
    xi: float = 1.5                      # surface exponent (S41)
    beta_cat: float = 1.5                # Bruggeman, cathode
    kappa_SPAN: float = 1.0              # electronic conductivity of the SPAN/CB matrix [S/m]
    c_DL: float = 0.1                    # double-layer capacitance [F/m2]
    Z_CC: float = 0.025                  # contact resistance [Ohm m2] (0.035 fits Fig. 4b best)
    rho_SPAN: float = 1440.0
    rho_CB: float = 1810.0
    w_S: float = 0.436                   # sulfur mass fraction of SPAN
    # --- separator (Table 1)
    eps_sep: float = 0.11                # glass fibre volume fraction (porosity 0.89)
    beta_sep: float = 1.0
    # --- electrolyte (Table 2)
    D_salt: float = 2.52e-10             # LiPF6 salt diffusion coefficient [m2/s]
    D_S: float = 4.7e-10                 # S2- diffusion coefficient [m2/s]
    t_plus: float = 0.1625
    kappa0: float = 0.796                # conductivity at 1 M [S/m]
    tdf: float = 1.6                     # thermodynamic factor 1 + dln f/dln c
    c_Li0: float = 1000.02
    c_PF6_0: float = 1000.0
    c_S0: float = 0.01                   # initial S2- [mol/m3]
    c_S_ref: float = 0.01                # S2- reference concentration in the SPAN kinetics (see docstring)
    c_S_sat: float = 1e-5                # S2- saturation concentration at K_sp = 10 [mol/m3] (Figs 5b, 7e-f)
    # --- SPAN kinetics (Table 3): reactions (1) S4 -> S3 + S1, (2) S3 -> S2 + S2-, (3) S2 -> S1 + S2-
    k0: tuple = (2.95e-8, 2.39e-8, 2.34e-8)  # effective frequency factors [mol/m2/s], fit to Fig. 6a (results/lispan/fit_k0.json)
    k0_table3: tuple = (1e-2, 1e-2, 1e-4)   # values printed in Table 3
    U0: tuple = (2.2, 1.9, 1.66)         # U^eq,0 [V]
    b: tuple = (0.3, 0.28, 0.62)         # OCV slopes [V]
    alpha: float = 0.5
    c_ref: tuple = (598.0, 598.0, 598.0, 1196.0)   # reference concentrations S4, S3Li, S2Li, SLi
    c_init: tuple = (598.0, 1e-5, 1e-5, 1e-5)      # initial concentrations
    kin_scale: float = 1.0               # common multiplier of the three k0 (parameter studies)
    reversible: tuple = (True, True, False)  # keep the oxidation (reverse) term of each SPAN reaction (see docstring)
    # --- Li2S precipitation (Table 3)
    K_sp: float = 10.0
    k0_L: float = 2e2                    # [mol/m2/s]
    precip_scale: float = 1.0            # multiplier of k0_L
    M_L: float = 0.04594                 # kg/mol
    rho_L: float = 1659.0
    # --- numerical regularisation (our additions, see model.py)
    eps_L_seed: float = 1e-5             # minimum Li2S area a_L = a_SPAN max(eps_L, eps_L_seed) (nucleation seed)
    c_floor_SPAN: float = 1e-6           # [mol/m3] softening of the log-dynamics of the SPAN species near zero
    c_floor_S: float = 1e-14             # [mol/m3] idem for S2- (below ~1 ion per mm3 of electrolyte)
    # --- Li metal anode (Table 3)
    k0_Li: float = 3.94
    alpha_Li: float = 0.5
    # --- temperature (not stated in the paper)
    T: float = 298.15

    # derived
    @property
    def L_tot(self) -> float:
        return self.L_cat + self.L_sep

    @property
    def eps_e0(self) -> float:
        """Initial cathode porosity (0.87 in Table 1)."""
        return 1.0 - self.eps_SPAN - self.eps_CB - self.eps_L0

    @property
    def eps_e_sep(self) -> float:
        return 1.0 - self.eps_sep

    @property
    def kappa_s_eff(self) -> float:
        return self.kappa_SPAN * self.eps_SPAN ** self.beta_cat        # (S34)

    @property
    def V_m_L(self) -> float:
        return self.M_L / self.rho_L

    @property
    def m_S_model(self) -> float:
        """Sulfur mass per electrode area implied by the SPAN concentrations [kg/m2]: 4 S atoms per
        PAN-S4-PAN chain.  Specific capacities in the paper's figures are consistent with this value
        (theoretical 1254 mAh/g_S = 6 e- per chain), not with the 0.6 mg/cm2 quoted in the text."""
        return 4.0 * self.c_init[0] * 0.032065 * self.L_cat

    @property
    def m_S_nominal(self) -> float:
        """Sulfur loading from eps_SPAN rho_SPAN L_cat w_S (0.59 mg/cm2)."""
        return self.eps_SPAN * self.rho_SPAN * self.L_cat * self.w_S

    @property
    def Q_theo(self) -> float:
        """Theoretical areal capacity [C/m2] = 6 F c_S4,0 L_cat (reactions 1-3, 2 e- each per chain)."""
        return 6.0 * F * self.c_init[0] * self.L_cat

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({"L_tot": self.L_tot, "eps_e0": self.eps_e0, "kappa_s_eff": self.kappa_s_eff,
                  "m_S_model": self.m_S_model, "Q_theo_Ah_m2": self.Q_theo / 3600.0})
        return d


@dataclass
class LiSPANProtocol:
    """Galvanostatic discharge.  ``current`` in A/m2 (positive = discharge); the paper's 1 C is
    1 mA/cm2 = 10 A/m2 (theoretical capacity 0.96 mAh/cm2)."""
    current: float = 1.0                 # A/m2  (0.1 C)
    ramp_s: float = 1.0                  # smooth switch-on I(t) = current * tanh(t / ramp_s)
    t_end_s: float | None = None         # default: 1.3 x nominal discharge time
    V_min: float = 1.0
    V_max: float = 3.0

    @classmethod
    def from_crate(cls, crate: float, **kw) -> "LiSPANProtocol":
        return cls(current=10.0 * crate, **kw)

    @property
    def crate(self) -> float:
        return self.current / 10.0

    def t_end(self, params: LiSPANParams) -> float:
        if self.t_end_s is not None:
            return self.t_end_s
        return 1.3 * params.Q_theo / self.current

    def I(self, t):
        return self.current * math.tanh(t / self.ramp_s) if self.ramp_s > 0 else self.current

    def to_dict(self) -> dict:
        return {"current_A_m2": self.current, "crate": self.crate, "ramp_s": self.ramp_s,
                "t_end_s": self.t_end_s, "V_min": self.V_min, "V_max": self.V_max}
