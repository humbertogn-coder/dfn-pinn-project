import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from check_inventory_seeds import (
    ARTIFACTS, CRITERIA, atomic_json, build_summary, fingerprints,
    recover_training, study_lock, validate_training,
)


def completed(seed, factor=1):
    metrics = {key: value*factor for key, value in CRITERIA.items()}
    metrics.update(positive_time_rms_error=.001, positive_time_pde_rms=.1, physical_pde_rms=.01)
    return {"seed": seed, "status": "completed", "comparison": {"runs": {
        "projected": {"metrics": metrics}, "control": {"metrics": metrics}}}}


def test_missing_seeds_prevent_acceptance():
    summary = build_summary({"runs": {"42": completed(42)}})
    assert summary["completed"] == 1
    assert summary["exploratory_status"] == "INCOMPLETE"
    assert summary["targets"]["initial_max_error"]["passing"] == 1


def test_all_seeds_required_and_worst_case_retained():
    runs = {str(seed): completed(seed) for seed in (0, 1, 2, 3, 42)}
    assert build_summary({"runs": runs})["exploratory_status"] == "PASS"
    runs["2"] = completed(2, 2)
    result = build_summary({"runs": runs})
    assert result["exploratory_status"] == "FAIL"
    assert result["targets"]["positive_time_max_error"]["passing"] == 4


def test_partial_run_is_not_recovered(tmp_path):
    partial = tmp_path/"partial"
    partial.mkdir()
    (partial/"report.json").write_text("{}", encoding="utf-8")
    assert recover_training([{"run": str(partial)}]) is None
    full = tmp_path/"full"
    full.mkdir()
    for name in ARTIFACTS:
        (full/name).write_bytes(b"test")
    assert recover_training([{"run": str(full)}, {"run": str(partial)}]) == full
    original = fingerprints(full)
    (full/"model.pt").write_bytes(b"modified")
    assert fingerprints(full) != original


def test_atomic_progress_round_trip(tmp_path):
    path = tmp_path/"progress.json"
    atomic_json(path, {"completed": 1})
    atomic_json(path, {"completed": 2})
    assert json.loads(path.read_text()) == {"completed": 2}
    assert not path.with_suffix(".tmp").exists()


def test_exclusive_lock_released(tmp_path):
    with study_lock(tmp_path):
        with pytest.raises(OSError):
            with study_lock(tmp_path):
                pytest.fail("Second writer acquired the study lock")
    with study_lock(tmp_path):
        pass


def test_wrong_seed_or_budget_rejected(tmp_path):
    report = {"config": {"seed": 42, "variant": "inventory", "sampling": "full_radius",
                          "adam_steps": 2000, "lbfgs_steps": 300, "mass_weight": 0},
              "inventory_projection": {"radial_order": 64}}
    for name in ARTIFACTS:
        (tmp_path/name).write_bytes(b"test")
    (tmp_path/"report.json").write_text(json.dumps(report), encoding="utf-8")
    assert validate_training(tmp_path, 42) == fingerprints(tmp_path)
    with pytest.raises(ValueError, match="configuration"):
        validate_training(tmp_path, 0)
    report["config"]["adam_steps"] = 10
    (tmp_path/"report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        validate_training(tmp_path, 42)
