"""Fail-closed integration of existing, explicitly selected DFN audit evidence."""

import math


def required_metrics():
    balance = {}
    def add(names, criterion):
        balance.update({name: criterion for name in names})
    add([f'total_current_region_{i}' for i in range(3)]+
        [f'collector_{side}_total_current' for side in ('negative', 'positive')],
        'max_total_current_relative_error')
    add([f'initial_ce_{i}' for i in range(3)]+[f'initial_cs_{i}' for i in (0, 2)],
        'max_initial_normalized_concentration_error')
    add([f'center_{i}' for i in (0, 2)], 'max_center_normalized_gradient')
    add([f'{kind}_residual_{i}' for kind in ('flux', 'kinetics') for i in (0, 2)],
        'max_normalized_flux_or_kinetics_residual')
    add([f'interface_{i}_{kind}' for i in (0, 1) for kind in
         ('concentration', 'potential', 'current', 'diffusive_flux')]+
        [f'solid_insulating_{i}' for i in (0, 1)]+
        [f'collector_{side}_{kind}' for side in ('negative', 'positive')
         for kind in ('electrolyte_current', 'diffusive_flux')]+['collector_positive_solid_current'],
        'max_normalized_boundary_or_interface_residual')
    add(['collector_negative_solid_potential_gauge'], 'max_normalized_gauge_residual')
    integrals = {'total_lithium_drift': 'max_total_lithium_drift_over_initial',
        'electrolyte_lithium_drift': 'max_electrolyte_lithium_drift_over_initial',
        'reaction_current_integrals': 'max_electrode_integrated_reaction_current_relative_error',
        'local_inventory_0': 'max_local_particle_inventory_error',
        'local_inventory_2': 'max_local_particle_inventory_error'}
    balance.update(integrals)
    add(['quadrature_gap_'+name for name in integrals], 'quadrature_metric_gap_over_metric_limit')
    pde = {}
    for i in range(3):
        for kind in (('salt', 'charge_e') if i == 1 else ('salt', 'charge_e', 'charge_s', 'particle')):
            for suffix, criterion in (('sampled_max', 'per_equation_normalized_pde_sampled_max'),
                                     ('physical_rms', 'per_equation_normalized_pde_rms'),
                                     ('quadrature_gap', 'quadrature_metric_gap_over_metric_limit')):
                pde[f'{kind}_{i}_{suffix}'] = criterion
    fields = {'voltage_V': 'max_voltage_error_V', 'c_e': 'max_ce_error_mol_m3',
              'phi_e': 'max_potential_error_V'}
    for suffix in ('n', 'p'):
        fields.update({f'phi_s_{suffix}': 'max_potential_error_V',
                       f'c_s_{suffix}': 'max_cs_error_over_cmax',
                       f'surface_{suffix}': 'max_surface_error_over_cmax',
                       f'j_{suffix}': 'max_reaction_current_error_over_jref'})
    return {'fields': fields, 'balances': balance, 'pdes': pde}


def combine(reports, config, settings, checkpoint_hash, reference_hash, config_hash, training):
    """Integrate historical smoke evidence; never infer full-budget completion."""
    limits = config['audit']['criteria']
    required = required_metrics()
    missing, failures, checked = [], [], {}
    for section, expected in required.items():
        report = reports.get(section)
        if report is None:
            missing.append(section)
            continue
        if report.get('checkpoint_sha256') != checkpoint_hash or report.get('reference_sha256') != reference_hash:
            raise ValueError(f'{section}: checkpoint/reference identity mismatch')
        if report.get('replay_exact') is not True:
            raise ValueError(f'{section}: exact replay evidence missing')
        if section != 'fields':
            if report.get('config_sha256') != config_hash or report.get('checkpoint_unmodified') is not True:
                raise ValueError(f'{section}: configuration or checkpoint integrity mismatch')
        checked[section] = {}
        for name, criterion in expected.items():
            entry = report.get('metrics', {}).get(name)
            if entry is None:
                missing.append(f'{section}/{name}')
                continue
            limit = limits[criterion]
            if section == 'fields':
                scale = 1.
                if name.startswith(('c_s_', 'surface_')):
                    scale = settings['cmax_mol_m3'][0 if name.endswith('_n') else 1]
                elif name.startswith('j_'):
                    k, i = (0, 0) if name.endswith('_n') else (1, 2)
                    area = 3*settings['solid_fractions'][k]/settings['radii_m'][k]
                    scale = settings['applied_current_A']/settings['area_m2']/(area*settings['lengths_m'][i])
                value = entry['max_abs_error']/scale
                expected_limit = limit*scale
            else:
                if entry.get('criterion') != criterion:
                    raise ValueError(f'{section}/{name}: criterion mismatch')
                value, expected_limit = entry['value'], limit
            if not math.isfinite(value) or value < 0 or not math.isfinite(entry['limit']):
                raise ValueError(f'{section}/{name}: invalid metric')
            if not math.isclose(entry['limit'], expected_limit, rel_tol=1e-12, abs_tol=0):
                raise ValueError(f'{section}/{name}: threshold changed')
            passed = value <= limit
            if entry.get('pass') is not passed:
                raise ValueError(f'{section}/{name}: inconsistent PASS flag')
            checked[section][name] = {'value': value, 'limit': limit, 'pass': passed, 'criterion': criterion}
            if not passed:
                failures.append(f'{section}/{name}')
    # Recorded optimizer settings or a low loss cannot prove execution completion.
    completion = {'status': 'NOT_FULL_BUDGET_EVIDENCE',
        'recorded_status': training.get('status'),
        'recorded_history_entries': len(training.get('history', [])),
        'required_adam_steps': config['training']['adam_steps'],
        'required_lbfgs_max_iter': config['training']['lbfgs_max_iter'],
        'reason': 'Historical smoke schema only; a future trainer must provide versioned completion and stopping evidence.'}
    quadrature_failed = any('quadrature_gap' in name for name in failures)
    metric_status = 'INCOMPLETE' if missing or quadrature_failed else ('FAIL' if failures else 'PASS')
    return {'status': 'INCOMPLETE_NOT_ACCEPTED', 'sampled_metric_status': metric_status,
        'metrics': checked, 'missing_metrics': missing, 'failed_metrics': failures,
        'completion': completion, 'metric_count': sum(map(len, checked.values())),
        'checkpoint_sha256': checkpoint_hash, 'reference_sha256': reference_hash, 'config_sha256': config_hash,
        'limitations': 'Integration of recorded evidence, not a new field evaluation. Historical smoke only. Reference and sampling uncertainty remain provisional; no continuous-time bounds or full DFN acceptance.',
        'remaining_gates': ['Versioned future-trainer completion schema and physical compatibility',
                            'Bounded cost/gradient/replay dry run', 'Full-budget attempt and matching audit']}
