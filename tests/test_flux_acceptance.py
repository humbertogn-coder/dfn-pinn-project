import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from assess_flux_pilot import CRITERIA, SAMPLERS, SEEDS, evaluate


def study():
    return {"runs": [{"seed": seed, "sampling": sampling, "status": "completed", **CRITERIA}
                     for seed in SEEDS for sampling in SAMPLERS]}


def test_inclusive_limits():
    assert all(result["status"] == "PASS" for result in evaluate(study()).values())


def test_one_bad_seed_fails_without_averaging():
    data = study()
    data["runs"][0]["positive_time_max_error"] *= 2
    result = evaluate(data)["legacy"]
    assert result["status"] == "FAIL"
    assert result["metrics"]["positive_time_max_error"]["passed"] == 4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, None, True])
def test_invalid_metric_cannot_pass(value):
    data = study()
    data["runs"][0]["max_surface_flux_residual"] = value
    assert evaluate(data)["legacy"]["status"] == "INCOMPLETE"


def test_missing_and_failed_runs_are_incomplete():
    data = study()
    data["runs"].pop(0)
    data["runs"][0]["status"] = "failed"
    assert all(result["status"] == "INCOMPLETE" for result in evaluate(data).values())


def test_duplicate_rejected():
    data = study()
    data["runs"].append(copy.deepcopy(data["runs"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(data)
