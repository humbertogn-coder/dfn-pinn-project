import torch
from dfn_pinn.dfn_smoke import DFNSmoke


def test_three_region_sampling_and_boundary_inventory():
    torch.manual_seed(42)
    model = DFNSmoke()
    points = model.sample()
    for i, (left, right) in enumerate(model.bounds):
        assert ((points[f"region_{i}"][:, 0] > left) & (points[f"region_{i}"][:, 0] < right)).all()
        assert (points[f"region_{i}"][:, 1] > 0).all()
    residuals, diagnostics = model.residuals(points)
    assert len(residuals) == 34 and len(diagnostics) == 4
    assert "collector_negative_solid_potential_gauge" in residuals
    assert "collector_negative_solid_current" not in residuals
    assert "collector_positive_solid_current" in residuals
    assert all(v.ndim == 2 and v.shape[1] == 1 and torch.isfinite(v).all() for v in residuals.values())
    assert not any("diagnostic" in key or "total_current" in key for key in residuals)
