"""Small project metadata helpers used by smoke tests and scripts."""

from __future__ import annotations


def project_summary() -> dict[str, str]:
    """Return a compact description of the research project."""
    return {
        "name": "dfn-pinn",
        "model": "isothermal full DFN/P2D",
        "reference_solver": "PyBaMM DFN + IDAKLU",
        "core_method": "mixed-variable inverse Butler-Volmer PINN",
        "first_goal": "export reproducible PyBaMM reference data",
    }
