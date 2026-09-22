from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_ramp_pilot import global_grid, local_grid, peak_record, nested_records, flux_peak_record


def test_nested_global_domains():
    r, t = global_grid()
    for stride in (4, 2, 1):
        assert r[::stride][0] == 0 and r[::stride][-1] == 1
        assert t[::stride][0] == 1e-5 and t[::stride][-1] == .02


def test_peak_keeps_sign_and_axes():
    result = peak_record(np.array([[1., -4.], [2., 3.]]), [0, 1], [.01, .02], True)
    assert result == {"value": 4., "signed_value": -4., "rho": 0., "tau": .02}


def test_local_grid_clips_boundaries():
    r, t = local_grid({"rho": 1., "tau": 1e-5})
    assert r.min() >= 0 and r.max() == 1
    assert t.min() == 1e-5 and t.max() <= .02


def test_signed_overshoot_not_absolute_error():
    r, t = np.linspace(0, 1, 5), np.linspace(1e-5, .02, 5)
    rows = nested_records(np.full((5, 5), .9), np.ones((5, 5)), r, t)
    assert rows[-1]["overshoot_above_one"]["value"] < 0
    assert rows[-1]["concentration_error"]["value"] > 0


def test_nonfinite_field_rejected():
    with pytest.raises(ValueError):
        peak_record([[np.nan]], [0], [.01])


def test_flux_sign_and_instantaneous_normalization():
    peak = flux_peak_record([.0035, .001], [1e-5, .02])
    assert peak["imposed_outward_flux"] == .0005
    assert peak["predicted_outward_flux"] == -.003
    assert peak["error_over_instantaneous_flux"] == 7
