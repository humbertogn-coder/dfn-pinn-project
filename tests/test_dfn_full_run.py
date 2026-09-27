"""Targeted full-run completion and reference compatibility guards."""

import copy
import json
from pathlib import Path

import pytest

from dfn_pinn.dfn_full_run import SCHEMA, completion_gate, verify_checkpoint
from dfn_pinn.dfn_smoke import SETTINGS


def setup():
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/dfn_baseline_v1.json').read_text())
    report = {'schema': SCHEMA, 'status': 'FULL_ATTEMPT_COMPLETE_NOT_PHYSICAL_ACCEPTANCE',
        'history': [{'step': i, 'total_loss': 1.} for i in range(1, 2001)],
        'completion': {'adam_completed': 2000, 'lbfgs_iterations': 200, 'lbfgs_evaluations': 240,
                       'training_wall_s': 600., 'stop_reason': 'optimizer_returned'}}
    return config, report


def test_bounded_normal_return_and_early_return():
    config, report = setup()
    assert completion_gate(report, config)
    report['completion']['lbfgs_iterations'] = 12
    assert completion_gate(report, config)


@pytest.mark.parametrize('change', ['missing_step', 'smoke', 'wall', 'evals', 'failure'])
def test_incomplete_or_excessive_runs_cannot_pass(change):
    config, report = setup()
    if change == 'missing_step':
        report['history'].pop()
    elif change == 'smoke':
        report['schema'] = 'dfn_run_v1'
    elif change == 'wall':
        report['completion']['training_wall_s'] = 7201
    elif change == 'evals':
        report['completion']['lbfgs_evaluations'] = 251
    else:
        report['status'] = 'STOPPED_NOT_COMPLETED'
    assert not completion_gate(report, config)


def test_checkpoint_hash_completion_and_physics():
    config, report = setup()
    report.update(settings=dict(SETTINGS, inventory_quadrature=32), checkpoint_sha256='c',
                  reference_sha256='r', config_sha256='s')
    saved = copy.deepcopy(report)
    reference = {'settings': SETTINGS, 'reference_sha256': 'r'}
    verify_checkpoint(saved, report, reference, 'c', 's')
    with pytest.raises(ValueError):
        verify_checkpoint(saved, report, reference, 'wrong', 's')
    saved['completion']['adam_completed'] = 3
    with pytest.raises(ValueError):
        verify_checkpoint(saved, report, reference, 'c', 's')
