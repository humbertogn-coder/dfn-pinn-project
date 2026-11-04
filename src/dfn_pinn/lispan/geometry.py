"""Cell geometry of the Li-SPAN model (geometry v5; docs/GEOMETRY.md).

The chemistry networks up to v4 are trained for one cell: the Simanjuntak test cell (100 um cathode with a low SPAN
fraction, porosity 0.87, and a 618 um glass-fibre separator).  Network v5 adds two cathode inputs; the separator is
fixed to a polyolefin (Celgard-type) separator, the one used by almost every SPAN-DB coin cell:

  L_cat  cathode thickness [um] (network input ln L_cat)
  f      SPAN scale factor: eps_SPAN = f x 0.0945 with the carbon fraction, the chain concentrations (c_init, c_ref)
         and the specific surface a_SPAN0 scaled by the same f, so the chemistry per PAN-S4-PAN chain is unchanged
         (activities c / c_ref, a / c_S4,0 and the extent equations do not depend on f); porosity 1 - 0.1265 f.

Sulfur loading of the model cell: m_S = m_S,ref f L_cat / 100 um (m_S,ref = 0.767 mg S/cm2).  A C-rate is per gram of
sulfur, so the current density at rate r is I = ONE_C r f L_cat / 100 um (ONE_C = 10 A/m2 of the reference cell).

Step 1 (scripts/lispan_geometry_scaling.py, lispan_geometry_reduction.py): the separator acts almost as a series
resistance (a 20-25 um polyolefin separator in place of the fixed one changes the fitted parameters by <= 1 mV / 2 %);
the cathode density f is the dominant geometry effect (dU0_2 up to -30 mV, k0_1 up to -47 % at porosity 0.37 with the
v4 geometry), the thickness at fixed f mostly enters the fitted resistance.
"""

from dataclasses import replace
import math

from .params import LiSPANParams

L_REF_UM = 100.0                       # thickness of the reference cathode
# Celgard-type separator: 25 um, porosity 0.41, MacMullin number ~8 -> porosity^beta = 1/8 -> beta = 2.33 (nominal
# catalogue values; scripts/lispan_spandb_geometry.py lists the separator classes of SPAN-DB)
SEP_POLYOLEFIN = {"L_sep_um": 25.0, "porosity": 0.41, "beta": 2.33}
SEP_GLASS_FIBRE_REF = {"L_sep_um": 618.0, "porosity": 0.89, "beta": 1.0}

GEO_INPUTS_V5 = (
    {"name": "lnL_cat", "lo": math.log(20.0), "hi": math.log(200.0)},     # ln(thickness / um)
    {"name": "f", "lo": 1.0, "hi": 5.0},                                  # SPAN scale factor (porosity 0.87 - 0.37)
)


def solid_fraction(p: LiSPANParams, f: float = 1.0) -> float:
    return f * (p.eps_SPAN + p.eps_CB)


def porosity(p: LiSPANParams, f: float) -> float:
    """Initial cathode porosity of the scaled cell (Li2S seed included)."""
    return 1.0 - solid_fraction(p, f) - p.eps_L0


def f_from_porosity(p: LiSPANParams, por: float) -> float:
    return (1.0 - p.eps_L0 - por) / (p.eps_SPAN + p.eps_CB)


def cell_params(p: LiSPANParams, L_cat_um: float, f: float = 1.0, L_sep_um: float | None = None,
                sep_porosity: float | None = None, beta_sep: float | None = None) -> LiSPANParams:
    """LiSPANParams of a cell with cathode thickness L_cat_um, SPAN scale factor f and (optionally) another separator.
    p must be the reference cell (L_cat 100 um, f = 1); the chemistry (U0, b, k0, ...) and Z_CC are kept."""
    out = replace(p, L_cat=L_cat_um * 1e-6, eps_SPAN=f * p.eps_SPAN, eps_CB=f * p.eps_CB,
                  c_init=(f * p.c_init[0],) + tuple(p.c_init[1:]), c_ref=tuple(f * c for c in p.c_ref),
                  a_SPAN0=f * p.a_SPAN0)
    if L_sep_um is not None:
        out = replace(out, L_sep=L_sep_um * 1e-6)
    if sep_porosity is not None:
        out = replace(out, eps_sep=1.0 - sep_porosity)
    if beta_sep is not None:
        out = replace(out, beta_sep=beta_sep)
    return out


def polyolefin_cell(p: LiSPANParams, L_cat_um: float, f: float) -> LiSPANParams:
    """The v5 network cell: scaled cathode + the fixed polyolefin separator."""
    s = SEP_POLYOLEFIN
    return cell_params(p, L_cat_um, f, s["L_sep_um"], s["porosity"], s["beta"])


def loading_mg_cm2(p_ref: LiSPANParams, L_cat_um: float, f: float) -> float:
    """Sulfur loading of the scaled cell [mg S/cm2] (m_S_model of the reference cell x f L / L_ref)."""
    return p_ref.m_S_model * 1e2 * f * L_cat_um / L_REF_UM        # kg/m2 -> mg/cm2 is x 100


def mass_ratio(L_cat_um: float, f: float) -> float:
    """Sulfur mass of the scaled cell / that of the reference cell (= current-density ratio at equal C-rate)."""
    return f * L_cat_um / L_REF_UM


def thickness_from_loading(p_ref: LiSPANParams, loading_mg_S_cm2: float, f: float) -> float:
    """Cathode thickness [um] of the model cell that holds the given sulfur loading at SPAN scale factor f."""
    return loading_mg_S_cm2 / loading_mg_cm2(p_ref, L_REF_UM, f) * L_REF_UM


# salt-transport load of the cathode: Theta = I L_cat / porosity^1.5 [A/m] at the full current.  FV corners of the
# v5 box (nominal chemistry, polyolefin separator, 2026-11-03): Theta 0.015-0.026 -> minimum salt 650-880 mol/m3;
# 0.078 -> 220; 0.090 -> 59 (depleted, the voltage then collapses on transport).  The v5 domain keeps Theta <= 0.03.
THETA_MAX = 0.03
LOADING_MAX = 6.0                       # mg S/cm2 (SPAN-DB liquid cells: median 0.95, p90 2.0, max 5.7)


def transport_load(p_ref: LiSPANParams, rate: float, L_cat_um: float, f: float) -> float:
    I = 10.0 * rate * mass_ratio(L_cat_um, f)
    return I * L_cat_um * 1e-6 / porosity(p_ref, f) ** 1.5


def in_domain(p_ref: LiSPANParams, rate: float, L_cat_um: float, f: float) -> bool:
    return (transport_load(p_ref, rate, L_cat_um, f) <= THETA_MAX * (1 + 1e-9)
            and loading_mg_cm2(p_ref, L_cat_um, f) <= LOADING_MAX * (1 + 1e-9))
