"""Preparation contract only; no model optimization or reference simulation."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_dfn", ROOT / "scripts/prepare_dfn_baseline.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def configuration():
    return json.loads((ROOT / "configs/dfn_baseline_v1.json").read_text())


def test_reproducibility_coordinates_and_endpoints():
    config = configuration()
    samples = module.make_samples(config, module.SETTINGS)
    again = module.make_samples(config, module.SETTINGS)
    edges = np.r_[0., np.cumsum(module.SETTINGS["lengths_m"])] / sum(module.SETTINGS["lengths_m"])
    for name, value in samples.items():
        np.testing.assert_array_equal(value, again[name])
        assert value.dtype == np.float64
        assert np.isfinite(value).all()
        assert (value[:, -1] > 0).all()
        assert (value[:, -1] <= 1/3600).all()
    for i in range(3):
        points = samples[f"region_{i}"]
        assert points.shape == (128, 2)
        assert ((points[:, 0] >= edges[i]) & (points[:, 0] <= edges[i+1])).all()
        assert (points[:, 1]*3600 < .001).any()
        if i != 1:
            particle = samples[f"particle_{i}"]
            np.testing.assert_array_equal(particle[:, 1:], points)
            assert ((particle[:, 0] >= 0) & (particle[:, 0] <= 1)).all()
    assert samples["times"][0, 0] == 1e-6/3600
    assert samples["times"][-1, 0] == 1/3600


def test_invalid_sampling_rejected():
    config = configuration()
    config["training"]["interior_points_per_region"] = 3
    with pytest.raises(ValueError):
        module.make_samples(config, module.SETTINGS)
    config = configuration()
    config["training"]["minimum_positive_time_s"] = 0
    with pytest.raises(ValueError):
        module.make_samples(config, module.SETTINGS)


def test_reference_hash_and_settings_are_not_optional(tmp_path):
    directory = tmp_path / "reference"
    directory.mkdir()
    (directory / "reference.h5").write_bytes(b"fixture")
    digest = module.sha256(directory / "reference.h5")
    config = {"reference": "reference", "reference_sha256": digest}
    report = {"settings": copy.deepcopy(module.SETTINGS), "reference_sha256": digest, "source_hashes": {}}
    path = directory / "report.json"
    path.write_text(json.dumps(report))
    module.verify_reference(tmp_path, config)
    (directory / "reference.h5").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        module.verify_reference(tmp_path, config)
    report["settings"]["applied_current_A"] = 6
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="settings"):
        module.verify_reference(tmp_path, config)
