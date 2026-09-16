import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("seed_study", Path(__file__).resolve().parents[1]/"scripts/check_single_particle_seeds.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def row(value):
    return {"status": "completed", **{key: value for key in study.METRICS},
            "reported_elapsed_s": value, "process_wall_s": value}


def test_statistics_and_failure_count():
    result = study.summarize([row(1.), row(3.), {"status": "failed"}])
    assert result["completed"] == 2 and result["failed"] == 1
    metric = result["statistics"]["max_abs_c_hat_error"]
    assert metric["mean"] == metric["median"] == 2.
    assert metric["minimum"] == 1. and metric["maximum"] == 3.
    assert abs(metric["sample_std"]-2**.5) < 1e-14


def test_all_failed_and_single_completed():
    assert study.summarize([{"status": "failed"}])["statistics"] == {}
    assert study.summarize([row(1.)])["statistics"]["max_abs_c_hat_error"]["sample_std"] is None
