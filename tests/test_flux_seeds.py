import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from check_flux_particle_seeds import OBJECTIVES, summarize


def row(seed, sampling, value):
    return {"seed": seed, "sampling": sampling, "status": "completed",
            "initial_weights_sha256": str(seed), **dict.fromkeys(OBJECTIVES, value)}


def test_pair_direction_and_failure_accounting():
    summary = summarize([row(0, "legacy", 3), row(0, "full_radius", 1),
                         row(1, "legacy", 5), {"seed": 1, "sampling": "full_radius", "status": "failed"}])
    assert summary["complete_pairs"] == 1
    assert summary["failed"] == 1
    assert summary["per_sampler"]["legacy"]["completed"] == 2
    for stats in summary["paired_statistics"].values():
        assert stats["median"] == -2
        assert stats["full_radius_lower"] == 1
        assert stats["sample_std"] is None


def test_empty_study():
    assert summarize([])["paired_statistics"] == {}
    assert summarize([])["attempted"] == 0


def test_hash_mismatch():
    with pytest.raises(ValueError, match="weights"):
        summarize([row(0, "legacy", 1), {**row(0, "full_radius", 1), "initial_weights_sha256": "bad"}])


def test_ties():
    summary = summarize([row(42, "legacy", 1), row(42, "full_radius", 1)])
    assert all(stats["ties"] == 1 for stats in summary["paired_statistics"].values())
