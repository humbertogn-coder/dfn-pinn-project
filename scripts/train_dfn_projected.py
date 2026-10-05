"""One matched hard-current/direct-kinetics attempt; preserves historical C0 trainer."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from dfn_pinn.dfn_smoke import SETTINGS
from dfn_pinn.dfn_variants import HARD_CURRENT, build_model, restore_model
from dfn_pinn.dfn_run_contract import adam_step, verify_physics
from dfn_pinn.dfn_full_run import SCHEMA, completion_gate
from prepare_dfn_baseline import make_samples, sha256, verify_reference


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root/'configs/dfn_baseline_v1.json'
    config = json.loads(config_path.read_text())
    reference = verify_reference(root, config)
    t = config['training']
    if t['residual_mse_weight'] != 1 or t['reference_labels_in_loss']:
        raise ValueError('Only the frozen baseline loss protocol is implemented')
    settings = dict(SETTINGS)
    settings.update({k: t[k] for k in ('interior_points_per_region', 'boundary_times', 'inventory_quadrature', 'adam_steps')})
    settings.update(seed=config['seed'], learning_rate=t['adam_learning_rate'])
    verify_physics(settings, reference['settings'])
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    model = build_model(settings, HARD_CURRENT)
    samples = {k: torch.from_numpy(v) for k, v in make_samples(config, settings).items()}
    adam = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    output = root/'results'/('dfn_projected_full_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    sources = [Path(__file__), root/'scripts/prepare_dfn_baseline.py', *sorted((root/'src/dfn_pinn').glob('*.py'))]
    report = {'schema': SCHEMA, 'status': 'RUNNING', 'settings': settings, 'variant': dict(HARD_CURRENT),
        'config_sha256': sha256(config_path), 'reference_sha256': reference['reference_sha256'],
        'source_hashes': {str(p.relative_to(root)): sha256(p) for p in sources},
        'history': [], 'lbfgs_history': [], 'torch_version': str(torch.__version__),
        'scope': 'Matched C0/F contrast: hard-current projection only; direct kinetics, fresh seed, unchanged 34 terms. No acceptance from loss.'}
    control = root/'results/dfn_baseline_full_20260927T023312902451Z'
    control_report = json.loads((control/'report.json').read_text())
    if control_report['config_sha256'] != report['config_sha256'] or control_report['settings'] != settings:
        raise ValueError('Control protocol/settings differ from the projected attempt')
    if sha256(control/'checkpoint.pt') != control_report['checkpoint_sha256']:
        raise ValueError('Control checkpoint changed')
    control_saved = torch.load(control/'checkpoint.pt', weights_only=True, map_location='cpu')
    for name, value in samples.items():
        torch.testing.assert_close(value, control_saved['samples'][name], rtol=0, atol=0)
    report['control'] = {'run': str(control), 'report_sha256': sha256(control/'report.json'),
                         'checkpoint_sha256': control_report['checkpoint_sha256'],
                         'training_samples_exact_match': True}
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    def write_report():
        temp = output/'report.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(output/'report.json')
    started = time.perf_counter()
    def guard():
        if time.perf_counter()-started >= t['wall_time_cap_s']:
            raise TimeoutError('Wall cap reached at step/closure boundary')
    def save(name, phase, optimizer, completion=None):
        with torch.no_grad():
            predictions = {k: v.detach().clone() for k, v in model.snapshot(samples).items()}
        payload = {'schema': SCHEMA, 'settings': settings, 'variant': dict(HARD_CURRENT), 'model': model.state_dict(),
            'samples': samples, 'predictions': predictions, 'optimizer': optimizer.state_dict(),
            'phase': phase, 'adam_completed': len(report['history']), 'rng_state': torch.get_rng_state(),
            'config_sha256': report['config_sha256'], 'reference_sha256': report['reference_sha256'],
            'completion': completion}
        temp = output/(name+'.tmp')
        torch.save(payload, temp)
        temp.replace(output/name)
    write_report()
    print(f'Projected DFN: Adam {t["adam_steps"]}, L-BFGS up to {t["lbfgs_max_iter"]}; float64 CPU.', flush=True)
    print(f'Output directory: {output}', flush=True)
    stage = 'adam'
    try:
        for step in range(1, t['adam_steps']+1):
            guard()
            entry = adam_step(model, adam, samples)
            entry['step'] = step
            report['history'].append(entry)
            if step % t['checkpoint_every_adam_steps'] == 0:
                save(f'adam_{step:04d}.pt', 'adam', adam)
                write_report()
                print(f'Adam {step}: loss={entry["total_loss"]:.6e}', flush=True)
        save('before_lbfgs.pt', 'adam', adam)
        accepted_adam = copy.deepcopy(model.state_dict())
        stage = 'lbfgs'
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=t['lbfgs_learning_rate'],
            max_iter=t['lbfgs_max_iter'], max_eval=t['lbfgs_max_eval'],
            history_size=t['lbfgs_history_size'], line_search_fn=t['lbfgs_line_search'])
        evaluations = 0
        def closure():
            nonlocal evaluations
            guard()
            if evaluations >= t['lbfgs_max_eval']:
                raise RuntimeError('Hard L-BFGS closure budget exhausted during line search')
            evaluations += 1
            lbfgs.zero_grad(set_to_none=True)
            residuals, _ = model.residuals(samples)
            loss = sum(v.square().mean() for v in residuals.values())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite L-BFGS loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Invalid L-BFGS gradient')
            report['lbfgs_history'].append({'evaluation': evaluations, 'loss': float(loss.detach())})
            if evaluations % 25 == 0:
                print(f'L-BFGS evaluation {evaluations}: loss={float(loss.detach()):.6e}', flush=True)
            return loss
        lbfgs.step(closure)
        guard()
        residuals, _ = model.residuals(samples)
        final_loss = sum(v.square().mean() for v in residuals.values())
        if not torch.isfinite(final_loss):
            raise FloatingPointError('Nonfinite final loss')
        state = lbfgs.state[next(iter(model.parameters()))]
        completed = {'adam_completed': len(report['history']), 'lbfgs_iterations': int(state.get('n_iter', 0)),
            'lbfgs_evaluations': evaluations, 'stop_reason': 'optimizer_returned',
            'training_wall_s': time.perf_counter()-started,
            'note': 'Normal bounded optimizer return, possibly early tolerance stopping; not a convergence certificate.'}
        report.update(status='FULL_ATTEMPT_COMPLETE_NOT_PHYSICAL_ACCEPTANCE', completion=completed,
                      final_training_loss=float(final_loss.detach()))
        if not completion_gate(report, config):
            raise AssertionError('Completion contract failed')
        save('checkpoint.pt', 'after_lbfgs', lbfgs, completed)
        saved = torch.load(output/'checkpoint.pt', weights_only=True, map_location='cpu')
        replay = restore_model(saved)
        with torch.no_grad():
            for name, value in replay.snapshot(saved['samples']).items():
                torch.testing.assert_close(value, saved['predictions'][name], rtol=0, atol=0)
        report.update(replay_exact=True, checkpoint_sha256=sha256(output/'checkpoint.pt'))
        write_report()
    except Exception as exc:
        # A failed line search can leave trial parameters in memory. Preserve
        # the last accepted Adam state instead of declaring that trial final.
        if stage == 'lbfgs':
            model.load_state_dict(accepted_adam)
        report.update(status='STOPPED_NOT_COMPLETED', error=f'{type(exc).__name__}: {exc}',
                      stopped_stage=stage, training_wall_s=time.perf_counter()-started)
        write_report()
        raise
    print('Bounded attempt completed and replay verified. Independent audit still required.', flush=True)
    print(f'Report: {output / "report.json"}')


if __name__ == '__main__':
    main()
