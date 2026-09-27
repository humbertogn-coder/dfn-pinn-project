"""Necessary identity and completion guards for the bounded dry run."""

import pytest

from dfn_pinn.dfn_smoke import SETTINGS
from dfn_pinn.dfn_run_contract import completion, verify_physics


def test_training_changes_do_not_change_physics():
    changed = dict(SETTINGS, adam_steps=20, inventory_quadrature=32, hidden_widths=[16, 16])
    verify_physics(changed, SETTINGS)


@pytest.mark.parametrize('key,value', [('applied_current_A', 6.), ('temperature_K', 310.), ('new_physics', True)])
def test_physical_and_unknown_changes_are_rejected(key, value):
    with pytest.raises(ValueError, match='Physical'):
        verify_physics(dict(SETTINGS, **{key: value}), SETTINGS)


def test_dry_run_never_implies_full_completion():
    assert completion('dry_run', 20, 20, 'requested_steps_completed')['dry_run_completed']
    assert not completion('dry_run', 20, 20, 'requested_steps_completed')['full_budget_completed']
    assert not completion('dry_run', 19, 20, 'error_or_wall_cap')['dry_run_completed']
    with pytest.raises(ValueError):
        completion('full', 20, 20, 'requested_steps_completed')
