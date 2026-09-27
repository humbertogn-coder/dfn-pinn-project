"""Twenty-step technical probe of hard current projection, not an ablation result."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from dfn_pinn.dfn_projected_current import DFNProjectedCurrent
from dfn_pinn.dfn_smoke import SETTINGS
from dfn_pinn.dfn_run_contract import adam_step
from dfn_pinn.projection import gauss_legendre
from prepare_dfn_baseline import make_samples, sha256, verify_reference


def integral_errors(model):
    errors = {}
    with torch.no_grad():
        for k, i in enumerate((0, 2)):
            x, w = gauss_legendre(64, *model.bounds[i])
            worst = 0.
            for time_s in (0., 1e-6, .01, .1, .5, 1.):
                p = torch.stack((x, torch.full_like(x, time_s/model.settings['time_reference_s'])), dim=1)
                j = model.reactions[k].integral.current_model(p).squeeze(1)
                current = (j*w*sum(model.settings['lengths_m'])*model.active_area[k]).sum()
                target = (1 if k == 0 else -1)*model.charge_scales.current_A_m2
                worst = max(worst, abs(float(current)-target)/abs(target))
            errors[str(i)] = worst
    return errors


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root/'configs/dfn_baseline_v1.json'
    config = json.loads(config_path.read_text())
    reference = verify_reference(root, config)
    settings = dict(SETTINGS)
    settings.update({k: config['training'][k] for k in
                     ('interior_points_per_region', 'boundary_times', 'inventory_quadrature')})
    settings.update(seed=config['seed'], adam_steps=20, learning_rate=config['training']['adam_learning_rate'])
    torch.set_num_threads(1)
    torch.manual_seed(settings['seed'])
    model = DFNProjectedCurrent(settings, projection_order=32)
    samples = {k: torch.from_numpy(v) for k, v in make_samples(config, settings).items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    output = root/'results'/('dfn_projected_current_probe_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    sources = [Path(__file__), root/'scripts/prepare_dfn_baseline.py', *sorted((root/'src/dfn_pinn').glob('*.py'))]
    report = {'schema': 'dfn_projected_probe_v1', 'status': 'RUNNING', 'settings': settings,
        'variant': model.variant_spec, 'config_sha256': sha256(config_path),
        'reference_sha256': reference['reference_sha256'], 'history': [],
        'source_hashes': {str(p.relative_to(root)): sha256(p) for p in sources},
        'torch_version': str(torch.__version__), 'wall_cap_s': 600,
        'scope': 'Technical probe only. Direct kinetics unchanged; no field-accuracy or local-conservation acceptance.'}
    def write():
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    started = time.perf_counter()
    write()
    try:
        for step in range(1, 21):
            if time.perf_counter()-started >= 600:
                raise TimeoutError('Probe wall cap reached at step boundary')
            tick = time.perf_counter()
            entry = adam_step(model, optimizer, samples)
            entry.update(step=step, wall_s=time.perf_counter()-tick)
            report['history'].append(entry)
            write()
            print(f'Projected Adam {step}: loss={entry["total_loss"]:.6e}, wall={entry["wall_s"]:.3f}s', flush=True)
        errors = integral_errors(model)
        if max(errors.values()) > 1e-10:
            raise ValueError('Independent current integral failed the probe tolerance')
        with torch.no_grad():
            predictions = model.snapshot(samples)
        torch.save({'schema': report['schema'], 'settings': settings, 'variant': model.variant_spec,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'samples': samples,
                    'predictions': predictions, 'adam_completed': 20, 'rng_state': torch.get_rng_state()}, output/'checkpoint.pt')
        saved = torch.load(output/'checkpoint.pt', weights_only=True, map_location='cpu')
        replay = DFNProjectedCurrent(saved['settings'], saved['variant']['projection_order'])
        replay.load_state_dict(saved['model'])
        with torch.no_grad():
            for name, value in replay.snapshot(saved['samples']).items():
                torch.testing.assert_close(value, saved['predictions'][name], rtol=0, atol=0)
        report.update(status='PROJECTION_PROBE_COMPLETE_NOT_PHYSICAL_ACCEPTANCE',
            independent_current_relative_errors=errors, independent_quadrature_order=64,
            audited_times_s=[0., 1e-6, .01, .1, .5, 1.], replay_exact=True,
            checkpoint_sha256=sha256(output/'checkpoint.pt'),
            median_step_s=float(np.median([e['wall_s'] for e in report['history']])),
            wall_s=time.perf_counter()-started)
        write()
    except Exception as exc:
        report.update(status='PROJECTION_PROBE_FAILED', error=f'{type(exc).__name__}: {exc}')
        write()
        raise
    print('Independent current integrals and prediction replay verified; no physical acceptance.', flush=True)
    print(f'Report: {output / "report.json"}')


if __name__ == '__main__':
    main()
