"""PyBaMM reference for the v2 benchmark (same cell, same tanh-ramp protocol).

The reference is ONLY used for evaluation (and, optionally, for the inverse
problem's synthetic voltage data). It never enters the forward PINN loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .params import CellParams, Protocol

FIELDS = {
    "V": "Voltage [V]",
    "I": "Current [A]",
    "c_e": "Electrolyte concentration [mol.m-3]",
    "phi_e": "Electrolyte potential [V]",
    "phi_s_n": "Negative electrode potential [V]",
    "phi_s_p": "Positive electrode potential [V]",
    "j_n": "Negative electrode interfacial current density [A.m-2]",
    "j_p": "Positive electrode interfacial current density [A.m-2]",
    "theta_surf_n": "Negative particle surface stoichiometry",
    "theta_surf_p": "Positive particle surface stoichiometry",
    "theta_n": "Negative particle stoichiometry",
    "theta_p": "Positive particle stoichiometry",
    "eta_n": "Negative electrode reaction overpotential [V]",
    "eta_p": "Positive electrode reaction overpotential [V]",
    "j0_n": "Negative electrode exchange current density [A.m-2]",
    "j0_p": "Positive electrode exchange current density [A.m-2]",
    "U_n": "Negative electrode open-circuit potential [V]",
    "U_p": "Positive electrode open-circuit potential [V]",
    "i_e": "Electrolyte current density [A.m-2]",
}


def parameter_values(cell: CellParams, protocol: Protocol):
    import pybamm

    p = pybamm.ParameterValues("Chen2020")
    current, ramp = protocol.current_A, protocol.ramp_s
    p.update({
        "Negative electrode thickness [m]": cell.L_n,
        "Separator thickness [m]": cell.L_s,
        "Positive electrode thickness [m]": cell.L_p,
        "Negative electrode porosity": cell.eps_n,
        "Separator porosity": cell.eps_s,
        "Positive electrode porosity": cell.eps_p,
        "Negative electrode active material volume fraction": cell.eps_am_n,
        "Positive electrode active material volume fraction": cell.eps_am_p,
        "Negative particle radius [m]": cell.R_n,
        "Positive particle radius [m]": cell.R_p,
        "Negative particle diffusivity [m2.s-1]": cell.D_n,
        "Positive particle diffusivity [m2.s-1]": cell.D_p,
        "Maximum concentration in negative electrode [mol.m-3]": cell.cmax_n,
        "Maximum concentration in positive electrode [mol.m-3]": cell.cmax_p,
        "Negative electrode conductivity [S.m-1]": cell.sigma_n,
        "Positive electrode conductivity [S.m-1]": cell.sigma_p,
        "Cation transference number": cell.t_plus,
        "Thermodynamic factor": cell.tdf,
        "Initial concentration in electrolyte [mol.m-3]": cell.c_e0,
        "Initial concentration in negative electrode [mol.m-3]": cell.c_n0,
        "Initial concentration in positive electrode [mol.m-3]": cell.c_p0,
        "Ambient temperature [K]": cell.T,
        "Initial temperature [K]": cell.T,
        "Reference temperature [K]": cell.T,
        "Lower voltage cut-off [V]": 2.0,
        "Current function [A]": lambda t: current * pybamm.tanh(t / ramp),
    }, check_already_exists=False)
    return p


def solve(cell: CellParams, protocol: Protocol, *, mesh=None, times=None,
          rtol=1e-8, atol=1e-10):
    import pybamm

    mesh = mesh or {"x_n": 40, "x_s": 20, "x_p": 40, "r_n": 60, "r_p": 60}
    if times is None:
        times = default_times(protocol)
    model = pybamm.lithium_ion.DFN()
    sim = pybamm.Simulation(model, parameter_values=parameter_values(cell, protocol),
                            var_pts=mesh,
                            solver=pybamm.CasadiSolver(mode="safe", rtol=rtol, atol=atol))
    # IDAKLU (PyBaMM 26.9) fails at t = 0 with the time-dependent ramp; the
    # Casadi DAE solver is used instead with tight tolerances.
    solution = sim.solve(times)
    return sim, solution, mesh


def default_times(protocol: Protocol):
    t_end = protocol.t_end_s
    early = np.linspace(0.0, min(5 * protocol.ramp_s, t_end), 151)
    late = np.linspace(0.0, t_end, 301)
    return np.unique(np.concatenate([early, late]))


def export(solution, path: Path, cell: CellParams, protocol: Protocol, mesh: dict):
    data = {"t": np.asarray(solution.t)}
    for key, name in FIELDS.items():
        data[key] = np.asarray(solution[name].entries)
    # Spatial coordinates (cell centres) from the solution variables.
    data["x"] = np.asarray(solution["x [m]"].entries[:, 0])
    data["x_n"] = np.asarray(solution["x_n [m]"].entries[:, 0])
    data["x_p"] = np.asarray(solution["x_p [m]"].entries[:, 0])
    data["r_n"] = np.asarray(solution["r_n [m]"].entries[:, 0, 0])
    data["r_p"] = np.asarray(solution["r_p [m]"].entries[:, 0, 0])
    meta = {"cell": cell.to_dict(), "protocol": protocol.to_dict(), "mesh": mesh,
            "fields": FIELDS, "solver": "pybamm.CasadiSolver(safe, rtol=1e-8, atol=1e-10)"}
    try:
        import pybamm
        meta["pybamm_version"] = pybamm.__version__
    except ImportError:  # pragma: no cover
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **data)
    return data


def load(path) -> dict:
    raw = np.load(path, allow_pickle=False)
    data = {k: raw[k] for k in raw.files if k != "meta"}
    data["meta"] = json.loads(str(raw["meta"]))
    return data
