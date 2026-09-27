"""Twenty Adam steps at baseline sample sizes; no physical acceptance or L-BFGS."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from dfn_pinn.dfn_smoke import DFNSmoke, SETTINGS
from dfn_pinn.dfn_run_contract import SCHEMA, adam_step, completion, physical_identity, verify_physics
from prepare_dfn_baseline import make_samples, sha256, verify_reference


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root/'configs/dfn_baseline_v1.json'
    config = json.loads(config_path.read_text())
    reference = verify_reference(root, config)
    train = config['training']
    settings = dict(SETTINGS)
    settings.update({key: train[key] for key in ('interior_points_per_region', 'boundary_times', 'inventory_quadrature')})
    settings.update(seed=config['seed'], adam_steps=20, learning_rate=train['adam_learning_rate'])
    verify_physics(settings, reference['settings'])
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    model = DFNSmoke(settings)
    samples = {k: torch.from_numpy(v) for k, v in make_samples(config, settings).items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    output = root/'results'/('dfn_dry_run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    sources = [Path(__file__), root/'scripts/prepare_dfn_baseline.py', *sorted((root/'src/dfn_pinn').glob('*.py'))]
    report = {'schema': SCHEMA, 'status': 'RUNNING', 'settings': settings,
        'physical_identity': physical_identity(settings), 'config_sha256': sha256(config_path),
        'reference_sha256': reference['reference_sha256'], 'history': [],
        'source_hashes': {str(p.relative_to(root)): sha256(p) for p in sources},
        'torch_version': str(torch.__version__), 'wall_cap_s': 600,
        'scope': 'Cost and replay only; fresh initialization, fixed samples, no reference labels, no L-BFGS or physical acceptance.'}
    path = output/'report.json'
    def write():
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    write()
    started = time.perf_counter()
    print('DFN dry run: 20 Adam steps, baseline samples, float64 CPU, one thread.', flush=True)
    try:
        for step in range(1, 21):
            if time.perf_counter()-started >= 600:
                raise TimeoutError('Dry-run wall cap reached at optimizer step boundary')
            tick = time.perf_counter()
            entry = adam_step(model, optimizer, samples)
            entry.update(step=step, wall_s=time.perf_counter()-tick)
            report['history'].append(entry)
            write()
            print(f'Adam {step}: loss={entry["total_loss"]:.6e}, wall={entry["wall_s"]:.3f}s', flush=True)
        with torch.no_grad():
            predictions = {k: v.detach().clone() for k, v in model.snapshot(samples).items()}
        record = completion('dry_run', 20, 20, 'requested_steps_completed')
        checkpoint = output/'checkpoint.pt'
        torch.save({'schema': SCHEMA, 'settings': settings, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'samples': samples, 'predictions': predictions,
                    'completion': record, 'rng_state': torch.get_rng_state()}, checkpoint)
        digest = sha256(checkpoint)
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        replay = DFNSmoke(saved['settings'])
        replay.load_state_dict(saved['model'])
        restored_optimizer = torch.optim.Adam(replay.parameters(), lr=settings['learning_rate'])
        restored_optimizer.load_state_dict(saved['optimizer'])
        with torch.no_grad():
            for name, value in replay.snapshot(saved['samples']).items():
                torch.testing.assert_close(value, saved['predictions'][name], rtol=0, atol=0)
        # Compare one further step on disposable copies; saved model remains at 20.
        control = copy.deepcopy(model)
        control_optimizer = torch.optim.Adam(control.parameters(), lr=settings['learning_rate'])
        control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        torch.set_rng_state(saved['rng_state'])
        a = adam_step(control, control_optimizer, samples)
        torch.set_rng_state(saved['rng_state'])
        b = adam_step(replay, restored_optimizer, saved['samples'])
        if a != b:
            raise AssertionError('Restored next-step diagnostics differ')
        for name, value in control.state_dict().items():
            torch.testing.assert_close(value, replay.state_dict()[name], rtol=0, atol=0)
        if sha256(checkpoint) != digest:
            raise AssertionError('Checkpoint changed during replay')
        costs = [entry['wall_s'] for entry in report['history']]
        report.update(status='DRY_RUN_COMPLETE_NOT_PHYSICAL_ACCEPTANCE', completion=record,
            replay_exact=True, next_adam_step_exact=True, checkpoint_sha256=digest,
            median_adam_step_s=float(np.median(costs)),
            estimated_adam_2000_s=float(np.median(costs)*2000),
            estimate_scope='Adam only; excludes L-BFGS, audits, I/O and runtime variability.',
            wall_s=time.perf_counter()-started)
        write()
    except Exception as exc:
        report.update(status='DRY_RUN_FAILED', error=f'{type(exc).__name__}: {exc}',
                      completion=completion('dry_run', len(report['history']), 20, 'error_or_wall_cap'))
        write()
        raise
    print('Exact prediction and next-Adam-step replay verified. No physical acceptance.', flush=True)
    print(f'Report: {path}')


if __name__ == '__main__':
    main()
