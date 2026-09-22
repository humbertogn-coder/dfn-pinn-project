import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_flux_candidates import nondominated, require_comparable


def test_tradeoffs_and_dominated_candidate():
    assert nondominated([[1, 3], [2, 2], [3, 1], [3, 3]]) == [True, True, True, False]


def test_ties_are_not_dominance():
    assert nondominated([[1, 1], [1, 1], [1, 2]]) == [True, True, False]


@pytest.mark.parametrize("values", [[], [[np.nan]], [[np.inf]], [1, 2], [[]]])
def test_invalid_objectives(values):
    with pytest.raises(ValueError):
        nondominated(values)


def test_mismatched_seed_is_rejected():
    report = {"config": {"seed": 42, "adam_steps": 2000, "lbfgs_steps": 300},
              "loss_weights": {"pde": 1}}
    require_comparable([report, report])
    with pytest.raises(ValueError, match="seed"):
        require_comparable([report, {**report, "config": {**report["config"], "seed": 0}}])
