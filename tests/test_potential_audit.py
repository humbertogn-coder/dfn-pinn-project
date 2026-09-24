import pytest
from dfn_pinn.potential_audit import assess


def test_equilibrium_sign_tolerance_and_smoke_guard():
    metrics = {"error": 0., "minimum_current_A_m2": 0., "minimum_outward_flux_A_m2": -1e-8,
               "minimum_concentration": .49, "maximum_concentration": .5, "pde_quadrature_gap": 0.}
    limits = {"error": .001, "sign_absolute_tolerance_A_m2": 1e-6}
    assert assess(metrics, limits, False)["status"] == "SAMPLED_TARGETS_PASS"
    assert assess(metrics, limits, True)["status"] == "SMOKE_DIAGNOSTIC_ONLY"
    metrics["minimum_outward_flux_A_m2"] = -2e-6
    assert not assess(metrics, limits, False)["checks"]["flux_sign"]


def test_missing_or_nonfinite_metric_cannot_pass():
    with pytest.raises(KeyError):
        assess({}, {"error": .001}, False)
    with pytest.raises(FloatingPointError):
        assess({"error": float("nan")}, {"error": .001}, False)
