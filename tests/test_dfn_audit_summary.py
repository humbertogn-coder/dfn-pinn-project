"""Fail-closed report integration without simulations or checkpoints."""

import copy
import json
from pathlib import Path

import pytest

from dfn_pinn.dfn_audit_summary import combine, required_metrics
from dfn_pinn.dfn_smoke import SETTINGS


def fixture():
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/dfn_baseline_v1.json').read_text())
    reports = {}
    for section, names in required_metrics().items():
        report = {'checkpoint_sha256': 'c', 'reference_sha256': 'r', 'config_sha256': 's',
                  'replay_exact': True, 'checkpoint_unmodified': True, 'metrics': {}}
        for name, criterion in names.items():
            limit = config['audit']['criteria'][criterion]
            scale = 1.
            if section == 'fields':
                if name.startswith(('surface_', 'c_s_')):
                    scale = SETTINGS['cmax_mol_m3'][0 if name.endswith('_n') else 1]
                if name.startswith('j_'):
                    k, i = (0, 0) if name.endswith('_n') else (1, 2)
                    scale = 5/.1027/(3*SETTINGS['solid_fractions'][k]/SETTINGS['radii_m'][k]*SETTINGS['lengths_m'][i])
            report['metrics'][name] = {'value': 0., 'max_abs_error': 0., 'limit': limit*scale,
                                      'criterion': criterion, 'pass': True}
        reports[section] = report
    return reports, config


def run(reports, config):
    return combine(reports, config, SETTINGS, 'c', 'r', 's', {'status': 'SMOKE', 'history': [1, 2, 3]})


def test_complete_metrics_never_promote_smoke_to_acceptance():
    reports, config = fixture()
    result = run(reports, config)
    assert result['metric_count'] == 83
    assert result['sampled_metric_status'] == 'PASS'
    assert result['status'] == 'INCOMPLETE_NOT_ACCEPTED'


def test_missing_metric_and_unresolved_quadrature_are_incomplete():
    reports, config = fixture()
    del reports['pdes']['metrics']['salt_0_physical_rms']
    assert run(reports, config)['sampled_metric_status'] == 'INCOMPLETE'
    reports, config = fixture()
    item = reports['pdes']['metrics']['salt_0_quadrature_gap']
    item.update(value=1., **{'pass': False})
    assert run(reports, config)['sampled_metric_status'] == 'INCOMPLETE'


def test_failing_physics_is_not_hidden_by_passes():
    reports, config = fixture()
    reports['fields']['metrics']['voltage_V'].update(max_abs_error=.1, **{'pass': False})
    assert run(reports, config)['sampled_metric_status'] == 'FAIL'


@pytest.mark.parametrize('mutation', ['checkpoint', 'threshold', 'flag', 'nonfinite'])
def test_inconsistent_evidence_is_rejected(mutation):
    reports, config = fixture()
    reports = copy.deepcopy(reports)
    item = reports['pdes']['metrics']['salt_0_physical_rms']
    if mutation == 'checkpoint':
        reports['fields']['checkpoint_sha256'] = 'different'
    elif mutation == 'threshold':
        item['limit'] *= 2
    elif mutation == 'flag':
        item['pass'] = False
    else:
        item['value'] = float('nan')
    with pytest.raises(ValueError):
        run(reports, config)
