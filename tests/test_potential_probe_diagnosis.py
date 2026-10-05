"""Analytic check of the added value/slope Jacobian diagnostic."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import torch


def test_affine_value_slope_jacobians(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    diagnostic = importlib.import_module("diagnose_potential_probe")

    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.coefficients = torch.nn.Parameter(torch.tensor([2., 3.], dtype=torch.float64))

        def forward(self, p):
            return self.coefficients[0] + self.coefficients[1]*p[:, :1]

    field = Affine()
    model = SimpleNamespace(phis=[None, field], settings={"time_reference_s": 3600.})
    result = diagnostic.value_slope_sensitivity(model, torch.tensor([.1, 1.], dtype=torch.float64))
    assert abs(result["value_jacobian_row_norm_rms"] - 2**.5) < 1e-12
    assert abs(result["slope_jacobian_row_norm_rms"] - 1.) < 1e-12
    assert all(abs(v - 1/2**.5) < 1e-12 for v in result["row_cosines"])
    assert field.coefficients.grad is None
