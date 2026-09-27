"""Versioned run identity and bounded dry-run training helpers."""

import torch


SCHEMA = 'dfn_run_v1'
NONPHYSICAL = {'scope', 'seed', 'hidden_widths', 'interior_points_per_region',
               'boundary_times', 'inventory_quadrature', 'adam_steps',
               'learning_rate', 'dtype', 'threads'}


def physical_identity(settings):
    # Unknown settings remain in the identity: new physics cannot be ignored.
    return {k: v for k, v in settings.items() if k not in NONPHYSICAL}


def verify_physics(settings, reference):
    if physical_identity(settings) != physical_identity(reference):
        raise ValueError('Physical reference identity mismatch')


def completion(mode, adam_completed, requested_adam, stop_reason):
    if mode != 'dry_run' or requested_adam != 20 or not 0 <= adam_completed <= 20:
        raise ValueError('Only the bounded 20-step dry-run contract is implemented')
    return {'schema': SCHEMA, 'mode': mode, 'adam_completed': adam_completed,
            'adam_requested': requested_adam, 'lbfgs_iterations': 0,
            'stop_reason': stop_reason, 'full_budget_completed': False,
            'dry_run_completed': adam_completed == 20 and stop_reason == 'requested_steps_completed'}


def adam_step(model, optimizer, samples):
    optimizer.zero_grad(set_to_none=True)
    residuals, _ = model.residuals(samples)
    if len(residuals) != 34:
        raise ValueError('Unexpected residual inventory')
    losses = {key: value.square().mean() for key, value in residuals.items()}
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite loss')
    total.backward()
    gradients = {}
    for name in ('ce', 'phie', 'phis', 'cs', 'reactions'):
        for i, branch in enumerate(getattr(model, name)):
            parameters = list(branch.parameters())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError(f'Missing/nonfinite gradient: {name}_{i}')
            norm = torch.sqrt(sum(p.grad.square().sum() for p in parameters))
            if not torch.isfinite(norm) or norm == 0:
                raise FloatingPointError(f'Invalid gradient norm: {name}_{i}')
            gradients[f'{name}_{i}'] = float(norm)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite post-step parameter')
    return {'total_loss': float(total.detach()),
            'losses': {k: float(v.detach()) for k, v in losses.items()},
            'branch_gradient_norms': gradients}
