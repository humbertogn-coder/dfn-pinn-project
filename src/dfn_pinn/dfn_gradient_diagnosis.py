"""Read-only loss-gradient diagnostics; not a physics acceptance audit."""

import math

import torch

from .charge_conservation import solid_charge_terms


def vector_gradient(loss, parameters, retain_graph=True):
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite diagnostic loss')
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    result = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                        for p, g in zip(parameters, gradients)])
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite diagnostic gradient')
    return result


def cosine(a, b):
    norm = float(a.norm()*b.norm())
    return None if norm == 0 else max(-1., min(1., float(torch.dot(a, b))/norm))


def loss_gradients(model, samples):
    parameters = list(model.parameters())
    slices, offset = {}, 0
    ordered = []
    for name in ('ce', 'phie', 'phis', 'cs', 'reactions'):
        for i, branch in enumerate(getattr(model, name)):
            branch_parameters = list(branch.parameters())
            ordered.extend(branch_parameters)
            size = sum(p.numel() for p in branch_parameters)
            slices[f'{name}_{i}'] = slice(offset, offset+size)
            offset += size
    if [id(p) for p in ordered] != [id(p) for p in parameters]:
        raise ValueError('Unexpected parameter/group order')
    residuals, _ = model.residuals(samples)
    if len(residuals) != 34:
        raise ValueError('Expected the unchanged 34-term loss')
    losses = {name: r.square().mean() for name, r in residuals.items()}
    gradients = {name: vector_gradient(loss, parameters) for name, loss in losses.items()}
    total = vector_gradient(sum(losses.values()), parameters)
    summed = sum(gradients.values())
    torch.testing.assert_close(summed, total, rtol=1e-8, atol=1e-10)
    terms = {}
    for name, gradient in gradients.items():
        terms[name] = {'loss': float(losses[name].detach()),
            'residual_sampled_max': float(residuals[name].detach().abs().max()),
            'gradient_norm': float(gradient.norm()), 'cosine_to_total': cosine(gradient, total),
            'branch_gradient_norms': {key: float(gradient[index].norm()) for key, index in slices.items()}}
    pairs = [('charge_s_2', 'kinetics_2'), ('collector_positive_solid_current', 'kinetics_2'),
             ('collector_positive_solid_current', 'charge_s_2'), ('particle_flux_2', 'particle_2'),
             ('particle_flux_2', 'kinetics_2'), ('particle_flux_0', 'particle_0')]
    alignment = {}
    for left, right in pairs:
        a, b = gradients[left], gradients[right]
        alignment[left+'__'+right] = {'global_cosine': cosine(a, b),
            'branch_cosines': {key: cosine(a[index], b[index]) for key, index in slices.items()}}
    denominator = sum(float(g.norm()) for g in gradients.values())
    return {'total_loss': float(sum(losses.values()).detach()), 'terms': terms,
        'total_gradient_norm': float(total.norm()),
        'norm_total_over_sum_term_norms': float(total.norm())/denominator if denominator else None,
        'gradient_sum_max_difference': float((summed-total).abs().max()),
        'selected_alignments': alignment,
        'scope': 'Euclidean parameter gradients of unchanged normalized MSE losses. Null cosine means a zero gradient. Parameterization dependent; not optimizer trajectory or physical acceptance.'}


def collector_sensitivity(model, times):
    result = {}
    c, scale = model.settings, model.charge_scales
    for k, x in enumerate((0., 1.)):
        points = torch.stack((torch.full_like(times, x), times/c['time_reference_s']), dim=1).requires_grad_()
        current = solid_charge_terms(model.phis[k], points, points.new_tensor(0.), scale,
            conductivity_S_m=c['solid_conductivities_S_m'][k])['current_A_m2']/scale.current_A_m2
        parameters = list(model.phis[k].parameters())
        jacobian = torch.stack([vector_gradient(v, parameters) for v in current[:, 0]])
        coefficient = c['solid_conductivities_S_m'][k]*scale.potential_V/(scale.length_m*scale.current_A_m2)
        result[str(k)] = {'normalized_current_min': float(current.detach().min()),
            'normalized_current_max': float(current.detach().max()),
            'current_jacobian_row_norm_rms': float(jacobian.square().sum(1).mean().sqrt()),
            'current_coefficient_per_normalized_potential_slope': coefficient,
            'required_normalized_potential_slope': -1/coefficient,
            'potential_output_amplitude': model.phis[k].amplitude}
        features = torch.stack((points[:, 0], points[:, 1]*model.phis[k].time_factor), dim=1)
        layers = []
        with torch.no_grad():
            for layer in model.phis[k].net:
                features = layer(features)
                if isinstance(layer, torch.nn.Tanh):
                    derivative = 1-features.square()
                    layers.append({'mean_tanh_derivative': float(derivative.mean()),
                                   'fraction_abs_activation_gt_0_99': float((features.abs() > .99).double().mean())})
        result[str(k)]['hidden_layers_at_collector'] = layers
    return result
