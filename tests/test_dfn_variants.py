"""Variant identity, fixed projection buffers and legacy replay compatibility."""

import copy
import pytest
import torch

from dfn_pinn.dfn_smoke import SETTINGS
from dfn_pinn.dfn_variants import HARD_CURRENT, build_model, restore_model, verify_checkpoint


@pytest.mark.parametrize('variant', [None, HARD_CURRENT])
def test_exact_restore_and_mismatched_architecture_rejection(variant):
    torch.manual_seed(42)
    model = build_model(SETTINGS, variant)
    saved = {'settings': SETTINGS, 'variant': variant, 'model': model.state_dict()}
    restored = restore_model(saved)
    samples = model.sample()
    for name, value in model.snapshot(samples).items():
        torch.testing.assert_close(value, restored.snapshot(samples)[name], rtol=0, atol=0)
    bad = dict(saved, variant=HARD_CURRENT if variant is None else None)
    with pytest.raises((RuntimeError, KeyError)):
        restore_model(bad)


def test_unknown_variant_and_tampered_fixed_target_rejected():
    with pytest.raises(ValueError):
        build_model(SETTINGS, dict(HARD_CURRENT, projection_order=16))
    model = build_model(SETTINGS, HARD_CURRENT)
    saved = {'settings': SETTINGS, 'variant': HARD_CURRENT, 'model': copy.deepcopy(model.state_dict())}
    saved['model']['reactions.0.integral.current_model.target'] *= 2
    with pytest.raises(AssertionError):
        restore_model(saved)


def test_report_cannot_relabel_checkpoint():
    saved = {'settings': SETTINGS, 'variant': HARD_CURRENT}
    with pytest.raises(ValueError, match='variant'):
        verify_checkpoint(saved, {'settings': SETTINGS}, {'settings': SETTINGS}, 'unused')


def test_fresh_raw_parameters_match_control_seed():
    torch.manual_seed(42)
    control = build_model(SETTINGS)
    torch.manual_seed(42)
    projected = build_model(SETTINGS, HARD_CURRENT)
    for old, new in zip(control.parameters(), projected.parameters(), strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
