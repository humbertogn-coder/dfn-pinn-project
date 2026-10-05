"""Explicit architecture identity for historical and projected DFN runs."""

import torch

from .dfn_full_run import verify_checkpoint as verify_base_checkpoint
from .dfn_projected_current import DFNProjectedCurrent
from .dfn_smoke import DFNSmoke


HARD_CURRENT = {'name': 'direct_bv_hard_current', 'projection_order': 32,
                'projection': 'uniform_additive_fixed_physical_quadrature'}


def build_model(settings, variant=None):
    if variant is None:
        return DFNSmoke(settings)
    if variant != HARD_CURRENT:
        raise ValueError('Unknown or changed DFN variant specification')
    return DFNProjectedCurrent(settings, projection_order=variant['projection_order'])


def verify_checkpoint(saved, training, reference, checkpoint_hash, config_hash=None):
    verify_base_checkpoint(saved, training, reference, checkpoint_hash, config_hash)
    variant = saved.get('variant')
    if variant != training.get('variant'):
        raise ValueError('Checkpoint/report variant mismatch')
    if variant is not None and variant != HARD_CURRENT:
        raise ValueError('Unsupported DFN variant')


def restore_model(saved):
    model = build_model(saved['settings'], saved.get('variant'))
    if saved.get('variant') is not None:
        # Quadrature and targets are fixed by the specification, not learned.
        for key, value in model.state_dict().items():
            if any(key.endswith('.current_model.'+suffix) for suffix in
                   ('nodes', 'weights', 'area', 'target')):
                torch.testing.assert_close(saved['model'][key], value, rtol=0, atol=0)
    model.load_state_dict(saved['model'], strict=True)
    return model
