"""Exercise native axis fidelity and deliberate HDF5 corruption detection."""

from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_internal_fields import verify_reference, write_reference


@pytest.fixture
def reference(tmp_path):
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    domains = {"primary": ["negative particle"], "secondary": ["negative electrode"]}
    record = {
        "id": "c_s_n", "pybamm_name": "particle", "unit": "mol/m3", "role": "primary",
        "axis_order": ["r", "x", "t"], "domains": domains,
        "coordinates": [
            {"axis": "r", "values": [1e-6, 2e-6], "unit": "m", "location": "nodes",
             "domains": domains["primary"]},
            {"axis": "x", "values": [1e-5, 2e-5, 3e-5], "unit": "m", "location": "nodes",
             "domains": domains["secondary"]},
        ],
    }
    report = {"time": {"values": [0., 1., 2., 3.]}, "variables": [record]}
    solution = {"particle": SimpleNamespace(entries=values),
                "Voltage [V]": SimpleNamespace(entries=np.array([4., 3.9, 3.8, 3.7])),
                "Current [A]": SimpleNamespace(entries=np.full(4, 5.))}
    path = tmp_path / "reference.h5"
    write_reference(path, solution, report)
    return path, solution, report


def test_round_trip_preserves_distinct_axes(reference):
    verify_reference(*reference)


@pytest.mark.parametrize("change", ["value", "coordinate", "unit", "axis"])
def test_verifier_rejects_corruption(reference, change):
    path, solution, report = reference
    with h5py.File(path, "r+") as handle:
        data = handle["fields/c_s_n/values"]
        if change == "value":
            data[0, 1, 2] += 1
        elif change == "coordinate":
            handle["fields/c_s_n/coordinates/r"][0] += 1e-6
        elif change == "unit":
            data.attrs["unit"] = "V"
        else:
            data.dims[0].label = "x"
    with pytest.raises(ValueError):
        verify_reference(path, solution, report)
