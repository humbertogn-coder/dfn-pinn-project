"""Audit a frozen DFN checkpoint without calling its training residual assembly."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch

from dfn_pinn.dfn_balance_audit import audit_balances
from dfn_pinn.dfn_pde_audit import audit_pdes
from dfn_pinn.dfn_variants import verify_checkpoint, restore_model
from prepare_dfn_baseline import sha256, verify_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/dfn_smoke_20260924T054745712381Z'))
    parser.add_argument('--pde-only', action='store_true', help='Audit the ten local equations instead of repeating balances')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root/args.run
    config_path = root/'configs/dfn_baseline_v1.json'
    config = json.loads(config_path.read_text())
    reference = verify_reference(root, config)
    training = json.loads((run/'report.json').read_text())
    for name, expected in training['source_hashes'].items():
        if sha256(root/name) != expected:
            raise ValueError(f'Checkpoint source mismatch: {name}')
    checkpoint = run/'checkpoint.pt'
    before = sha256(checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    verify_checkpoint(saved, training, reference, before, sha256(config_path))
    torch.set_num_threads(1)
    model = restore_model(saved)
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for name, value in model.snapshot(saved['samples']).items():
            torch.testing.assert_close(value, saved['predictions'][name], rtol=0, atol=0)
    print('Source identity and exact replay verified. Auditing frozen DFN fields...', flush=True)
    result = audit_pdes(model, config) if args.pde_only else audit_balances(model, config)
    if sha256(checkpoint) != before:
        raise ValueError('Checkpoint changed during audit')
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved['model'][name], rtol=0, atol=0)
    result.update({'variant': saved.get('variant'), 'checkpoint_sha256': before, 'config_sha256': sha256(config_path),
                   'reference_sha256': reference['reference_sha256'], 'replay_exact': True,
                   'checkpoint_unmodified': True, 'source_hashes': {str(p.relative_to(root)): sha256(p)
                    for p in [Path(__file__), root/'scripts/prepare_dfn_baseline.py', *sorted((root/'src/dfn_pinn').glob('*.py'))]}})
    prefix = 'pde_audit_' if args.pde_only else 'balance_audit_'
    output = run/(prefix+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for name, m in result['metrics'].items():
        print(f"{name}: {'PASS' if m['pass'] else 'FAIL'} value={m['value']:.6e}, limit={m['limit']:.6e}")
    print(result['status'])
    print(f'Report: {output / "report.json"}')


if __name__ == '__main__':
    main()
