import json
from pathlib import Path

import torch

from dfn_pinn.constitutive import FARADAY_CONSTANT, ocp, exchange_current
from dfn_pinn.kinetics import direct_bv, inverse_bv


def specification():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/coupled_particle_v1.json").read_text())


def test_reference_is_admissible_and_compatible():
    c = specification()
    r, d, cref = c["radius_m"], c["diffusivity_m2_s"], c["concentration_scale_mol_m3"]
    j, tref = c["reference_current_A_m2"], c["time_reference_s"]
    b = -j*r/(2*FARADAY_CONSTANT*d*cref)
    k = -3*tref*j/(FARADAY_CONSTANT*r*cref)
    assert abs(k-6*b*d*tref/r**2) < 1e-12
    rho = torch.linspace(0, 1, 201, dtype=torch.float64)
    tau = torch.linspace(0, c["duration_s"]/tref, 401, dtype=torch.float64)
    profile = c["initial_mean"]+k*tau[None, :]+b*(rho[:, None]**2-.6)
    assert bool(((profile > 0) & (profile < 1)).all())
    theta = profile[-1]
    ce = torch.full_like(theta, c["electrolyte_concentration_mol_m3"])
    temp = torch.full_like(theta, c["temperature_K"])
    current = torch.full_like(theta, j)
    j0 = exchange_current(ce, theta*cref, temp, c["electrode"], mode=c["exchange_current_mode"])
    potential = ocp(theta, c["electrode"])+inverse_bv(current, j0, temp)
    torch.testing.assert_close(direct_bv(potential-ocp(theta, c["electrode"]), j0, temp), current)


def test_specification_is_bounded_and_not_a_training_result():
    c = specification()
    assert c["status"] == "specified_not_trained"
    assert c["adam_steps"] == 2000 and c["lbfgs_max_eval"] == 375
    assert c["device"] == "cpu" and c["dtype"] == "float64"
    assert all(value > 0 for value in c["acceptance"].values())
    assert c["acceptance"]["max_current_error_A_m2"] < c["reference_current_A_m2"]
