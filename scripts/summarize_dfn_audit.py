"""Combine explicitly pinned existing audits without training or recomputation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from dfn_pinn.dfn_audit_summary import combine
from dfn_pinn.dfn_smoke import DFNSmoke
from dfn_pinn.dfn_full_run import verify_checkpoint, completion_gate
from prepare_dfn_baseline import sha256, verify_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/dfn_smoke_20260924T054745712381Z'))
    parser.add_argument('--fields', type=Path, default=Path('reference_comparison_20260924T211656586255Z/report.json'))
    parser.add_argument('--balances', type=Path, default=Path('balance_audit_20260925T052740252744Z/report.json'))
    parser.add_argument('--pdes', type=Path, default=Path('pde_audit_20260925T171946440055Z/report.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root/args.run
    config_path = root/'configs/dfn_baseline_v1.json'
    config = json.loads(config_path.read_text())
    reference = verify_reference(root, config)
    training_path = run/'report.json'
    training = json.loads(training_path.read_text())
    for name, expected in training['source_hashes'].items():
        if sha256(root/name) != expected:
            raise ValueError(f'Training source mismatch: {name}')
    checkpoint = run/'checkpoint.pt'
    checkpoint_hash = sha256(checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    verify_checkpoint(saved, training, reference, checkpoint_hash, sha256(config_path))
    torch.set_num_threads(1)
    model = DFNSmoke(saved['settings']).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    with torch.no_grad():
        for name, value in model.snapshot(saved['samples']).items():
            torch.testing.assert_close(value, saved['predictions'][name], rtol=0, atol=0)
    paths = {key: run/getattr(args, key) for key in ('fields', 'balances', 'pdes')}
    reports = {key: json.loads(path.read_text()) for key, path in paths.items()}
    result = combine(reports, config, saved['settings'], checkpoint_hash,
                     reference['reference_sha256'], sha256(config_path), training)
    if completion_gate(training, config):
        result['completion'] = training['completion']
        result['status'] = {'FAIL': 'FULL_ATTEMPT_AUDITED_FAIL',
                            'PASS': 'SAMPLED_CRITERIA_PASS_PROVISIONAL',
                            'INCOMPLETE': 'INCOMPLETE_NOT_ACCEPTED'}[result['sampled_metric_status']]
        result['remaining_gates'] = ['Interpret audited results; no automatic retraining or budget extension']
        result['limitations'] = 'Completed bounded attempt, sampled criteria only. Provisional reference; no continuous-time guarantee, new protocol generalization or experimental validation.'
    if sha256(checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed during integration')
    result['evidence'] = {key: {'path': str(path), 'sha256': sha256(path)} for key, path in paths.items()}
    result['training_report_sha256'] = sha256(training_path)
    result['replay_exact'] = True
    result['source_hashes'] = {str(p.relative_to(root)): sha256(p) for p in
        (Path(__file__), root/'src/dfn_pinn/dfn_audit_summary.py')}
    output = run/('combined_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for section, metrics in result['metrics'].items():
        print(f'{section}: {sum(m["pass"] for m in metrics.values())}/{len(metrics)} metrics pass')
    print(f'Sampled metric status: {result["sampled_metric_status"]}')
    print(f'Overall: {result["status"]}; no training or simulation was run.')
    print(f'Report: {output / "report.json"}')


if __name__ == '__main__':
    main()
