"""Li-SPAN continuum model (Simanjuntak et al., Electrochim. Acta 497 (2024) 144571).

Numerical reference (1-D finite volumes, method of lines) for the Li-SPAN PINN work; see
docs/LI_SPAN_MODEL_FORMULATION.md for the equations and the conventions used here.
"""

from .params import LiSPANParams, LiSPANProtocol  # noqa: F401
from .model import LiSPANModel, solve, export, load  # noqa: F401
