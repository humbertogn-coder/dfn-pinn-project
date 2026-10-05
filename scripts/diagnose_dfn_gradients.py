"""Diagnose both frozen full attempts without optimizer steps or model changes."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from dfn_pinn.dfn_gradient_diagnosis import collector_sensitivity, loss_gradients
from dfn_pinn.dfn_variants import restore_model, verify_checkpoint
from prepare_dfn_baseline import make_samples, sha256, verify_reference


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root/'configs/dfn_baseline_v1.json'
    config = json.loads(config_path.read_text())
    reference = verify_reference(root, config)
    alternate = copy.deepcopy(config)
    alternate['sampling_seed'] = 20260928
    runs = {'C0': 'dfn_baseline_full_20260927T023312902451Z',
            'F': 'dfn_projected_full_20260927T232900582653Z'}
    report = {'status': 'FROZEN_GRADIENT_DIAGNOSTIC_ONLY', 'runs': {},
        'diagnostic_sampling_seed': alternate['sampling_seed'],
        'config_sha256': sha256(config_path), 'reference_sha256': reference['reference_sha256'],
        'scope': 'No training, parameter perturbation, weight change or acceptance. New samples are diagnostic, not a physical-measure integral.'}
    torch.set_num_threads(1)
    for label, directory in runs.items():
        run = root/'results'/directory
        training = json.loads((run/'report.json').read_text())
        for name, expected in training['source_hashes'].items():
            if sha256(root/name) != expected:
                raise ValueError(f'Training source mismatch: {name}')
        path = run/'checkpoint.pt'
        before = sha256(path)
        saved = torch.load(path, weights_only=True, map_location='cpu')
        verify_checkpoint(saved, training, reference, before, sha256(config_path))
        model = restore_model(saved).eval()
        with torch.no_grad():
            for name, value in model.snapshot(saved['samples']).items():
                torch.testing.assert_close(value, saved['predictions'][name], rtol=0, atol=0)
        fresh = {k: torch.from_numpy(v) for k, v in make_samples(alternate, saved['settings']).items()}
        record = {'run': str(run), 'variant': saved.get('variant'), 'checkpoint_sha256': before,
                  'training_report_sha256': sha256(run/'report.json')}
        for tag, samples in (('training_points', saved['samples']), ('diagnostic_points', fresh)):
            print(f'{label}: per-term gradients on {tag}...', flush=True)
            record[tag] = loss_gradients(model, samples)
        times = torch.tensor(np.unique(np.r_[np.linspace(1e-6, 1, 17), np.geomspace(1e-6, 1, 17)]), dtype=torch.float64)
        record['collector_sensitivity'] = collector_sensitivity(model, times)
        record['collector_times_s'] = times.tolist()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved['model'][name], rtol=0, atol=0)
        if sha256(path) != before:
            raise ValueError('Checkpoint modified')
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError('Unexpected accumulated parameter gradients')
        record.update(checkpoint_unmodified=True, replay_exact=True)
        report['runs'][label] = record
        terms = record['training_points']['terms']
        for name in ('collector_positive_solid_current', 'charge_s_2', 'kinetics_2', 'particle_flux_2'):
            value = terms[name]
            print(f'{label} {name}: loss={value["loss"]:.6e}, gradient={value["gradient_norm"]:.6e}', flush=True)
    output = root/'results'/('dfn_gradient_diagnosis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    np.savez(output/'diagnostic_samples.npz', **{k: v.numpy() for k, v in fresh.items()})
    report['diagnostic_samples_sha256'] = sha256(output/'diagnostic_samples.npz')
    report['source_hashes'] = {str(p.relative_to(root)): sha256(p) for p in
        [Path(__file__), root/'scripts/prepare_dfn_baseline.py', *sorted((root/'src/dfn_pinn').glob('*.py'))]}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'No training; both checkpoints unchanged. Report: {output / "report.json"}')


if __name__ == '__main__':
    main()
