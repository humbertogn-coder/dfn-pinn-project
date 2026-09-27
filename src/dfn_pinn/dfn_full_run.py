"""Full-attempt metadata and checkpoint compatibility, separate from physics acceptance."""

import hashlib
import math

from .dfn_run_contract import verify_physics


SCHEMA = 'dfn_full_run_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checkpoint(saved, training, reference, checkpoint_hash, config_hash=None):
    if saved['settings'] != training['settings']:
        raise ValueError('Checkpoint/report settings mismatch')
    if saved.get('schema') == SCHEMA:
        verify_physics(saved['settings'], reference['settings'])
        if training.get('schema') != SCHEMA or saved.get('completion') != training.get('completion'):
            raise ValueError('Completion record mismatch')
        if training.get('checkpoint_sha256') != checkpoint_hash:
            raise ValueError('Training checkpoint hash mismatch')
        if saved.get('reference_sha256') != reference['reference_sha256'] or training.get('reference_sha256') != reference['reference_sha256']:
            raise ValueError('Training reference hash mismatch')
        if saved.get('config_sha256') != training.get('config_sha256'):
            raise ValueError('Training config hash mismatch')
        if config_hash is not None and training.get('config_sha256') != config_hash:
            raise ValueError('Pinned config hash mismatch')
    elif saved['settings'] != reference['settings']:
        raise ValueError('Historical checkpoint/reference settings mismatch')


def completion_gate(training, config):
    """A returned bounded attempt is not optimizer convergence or physical success."""
    if training.get('schema') != SCHEMA:
        return False
    c, t = training.get('completion', {}), config['training']
    history = training.get('history', [])
    if (training.get('status') != 'FULL_ATTEMPT_COMPLETE_NOT_PHYSICAL_ACCEPTANCE'
            or c.get('stop_reason') != 'optimizer_returned'
            or c.get('adam_completed') != t['adam_steps']
            or len(history) != t['adam_steps']
            or [e.get('step') for e in history] != list(range(1, t['adam_steps']+1))):
        return False
    for entry in history:
        if not math.isfinite(entry.get('total_loss', float('nan'))):
            return False
    iterations, evaluations = c.get('lbfgs_iterations'), c.get('lbfgs_evaluations')
    elapsed = c.get('training_wall_s', float('nan'))
    return (type(iterations) is int and 0 <= iterations <= t['lbfgs_max_iter']
            and type(evaluations) is int and 1 <= evaluations <= t['lbfgs_max_eval']
            and math.isfinite(elapsed) and 0 <= elapsed <= t['wall_time_cap_s'])
