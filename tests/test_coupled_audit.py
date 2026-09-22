import json
from pathlib import Path
import pytest
import torch
from torch import nn

from dfn_pinn.coupled_training import CoupledParticle, reference
from dfn_pinn.coupled_audit import audit, assess


def test_exact_manufactured_solution_and_smoke_guard():
    c = json.loads((Path(__file__).resolve().parents[1]/"configs/coupled_particle_v1.json").read_text())
    class ExactConcentration(nn.Module):
        def forward(self, p):
            return reference(p, c)
    class ExactCurrent(nn.Module):
        def forward(self, p):
            return p[:, :1]*0+c["reference_current_A_m2"]
    model = CoupledParticle(c)
    model.concentration = ExactConcentration()
    model.current.integral.current_model = ExactCurrent()
    metrics, _ = audit(model, nr=9, nt=11, orders=(8, 16))
    result = assess(metrics, c["acceptance"], False)
    assert result["all_sampled_targets_pass"]
    for key in c["acceptance"]:
        assert metrics[key] < 1e-11
    assert assess(metrics, c["acceptance"], True)["status"] == "SMOKE_DIAGNOSTIC_ONLY"
    metrics["minimum_outward_flux_A_m2"] = -.01
    assert not assess(metrics, c["acceptance"], False)["all_sampled_targets_pass"]
    metrics["max_current_error_A_m2"] = float("nan")
    with pytest.raises(FloatingPointError):
        assess(metrics, c["acceptance"], False)


def test_threshold_failure_is_not_accepted():
    metrics = {"error": .2, "minimum_current_A_m2": .1,
               "minimum_outward_flux_A_m2": .1, "minimum_concentration": .4,
               "maximum_concentration": .6, "pde_quadrature_gap": 0.}
    result = assess(metrics, {"error": .1}, False)
    assert result["status"] == "SAMPLED_TARGETS_FAIL"
    assert result["checks"]["error"] is False
