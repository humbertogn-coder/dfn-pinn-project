"""Frozen-field DFN inventory/current and boundary audit, independent of losses.

This stage does not certify PDE residuals or native reference-field errors.
Never promote its diagnostics to full DFN acceptance.
"""

import numpy as np
import torch

from .constitutive import FARADAY_CONSTANT
from .charge_conservation import electrolyte_charge_terms, solid_charge_terms
from .dfn_boundaries import collector_conditions, electrolyte_interface, solid_separator_condition
from .spherical_diffusion import center_residual
from .particle_interface import particle_interface


def grid(*axes):
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


def quadrature(order, left=0., right=1.):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return left+(nodes+1)*(right-left)/2, weights*(right-left)/2


def evaluate(function, points, derivatives=False, chunk=512):
    values = []
    for start in range(0, len(points), chunk):
        p = torch.tensor(points[start:start+chunk], dtype=torch.float64, requires_grad=derivatives)
        with torch.set_grad_enabled(derivatives):
            v = function(p)
        if v.shape != (len(p), 1) or not torch.isfinite(v).all():
            raise ValueError("Audit requires finite (N,1) field values")
        values.append(v.detach().numpy().ravel())
    return np.concatenate(values)


def times_for(config, settings):
    a = config["audit"]
    return np.unique(np.r_[np.linspace(a["minimum_positive_time_s"], settings["duration_s"], a["uniform_time_points"]),
                           np.geomspace(a["minimum_positive_time_s"], settings["duration_s"], a["log_time_points"])])


def inventory_curves(model, physical_times, order):
    """Direct physical integrals of predicted fields; no prescribed mass substitution."""
    c = model.settings
    tau = np.asarray(physical_times)/c["time_reference_s"]
    r, wr = quadrature(order)
    u, wu = quadrature(order)
    total_e = np.zeros(len(tau))
    total_s = np.zeros(len(tau))
    currents, local_errors = {}, {}
    for i, (left, right) in enumerate(model.bounds):
        x, wx = quadrature(order, left, right)
        dx = wx*sum(c["lengths_m"])
        ce = evaluate(model.ce[i], grid(x, tau)).reshape(len(x), len(tau))*c["initial_ce_mol_m3"]
        total_e += c["area_m2"]*c["porosities"][i]*np.einsum('x,xt->t', dx, ce)
        if i == 1:
            continue
        k = 0 if i == 0 else 1
        cs = evaluate(model.cs[k], grid(r, x, tau)).reshape(order, order, len(tau))
        mean = np.einsum('r,rxt->xt', 3*r*r*wr, cs)
        total_s += c["area_m2"]*c["solid_fractions"][k]*c["cmax_mol_m3"][k]*np.einsum('x,xt->t', dx, mean)
        current = model.reactions[k].integral.current_model
        j = evaluate(current, grid(x, tau)).reshape(order, len(tau))
        currents[str(i)] = c["area_m2"]*model.active_area[k]*np.einsum('x,xt->t', dx, j)
        # Use a separate time quadrature, not ParticleCurrent.mean_target.
        xx, tt, uu = np.meshgrid(x, tau, u, indexing="ij")
        queries = np.column_stack((xx.ravel(), (tt*uu).ravel()))
        jt = evaluate(current, queries).reshape(order, len(tau), order)
        charge = c["time_reference_s"]*tau[None, :]*np.einsum('xtu,u->xt', jt, wu)
        target = c["initial_stoichiometries"][k]-3*charge/(FARADAY_CONSTANT*c["radii_m"][k]*c["cmax_mol_m3"][k])
        local_errors[str(i)] = float(np.abs(mean-target).max())
    return {"total_mol": total_s+total_e, "electrolyte_mol": total_e,
            "electrode_current_A": currents, "local_inventory_errors": local_errors}


def local_inventory_error(model, k, xt, order):
    """Fixed audit locations prevent changing quadrature nodes from hiding maxima."""
    c = model.settings
    r, w = quadrature(order)
    values = []
    for start in range(0, len(xt), 128):
        p = xt[start:start+128]
        points = np.column_stack((np.tile(r, len(p)), np.repeat(p[:, 0], order), np.repeat(p[:, 1], order)))
        mean = evaluate(model.cs[k], points).reshape(len(p), order)@(3*r*r*w)
        queries = np.column_stack((np.repeat(p[:, 0], order), (p[:, 1:]*r).ravel()))
        j = evaluate(model.reactions[k].integral.current_model, queries).reshape(len(p), order)
        charge = c["time_reference_s"]*p[:, 1]*(j@w)
        target = c["initial_stoichiometries"][k]-3*charge/(FARADAY_CONSTANT*c["radii_m"][k]*c["cmax_mol_m3"][k])
        values.append(mean-target)
    return np.concatenate(values)


def reaction_error(currents, applied):
    return max(float(np.max(np.abs(currents['0']-applied))),
               float(np.max(np.abs(currents['2']+applied))))/abs(applied)


def audit_balances(model, config):
    """Return implemented criteria and explicit missing gates; no optimization."""
    c, a = model.settings, config["audit"]
    limits = a["criteria"]
    metrics, diagnostics = {}, {}
    def record(name, value, criterion):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite audit metric: {name}")
        limit = limits[criterion]
        metrics[name] = {"value": value, "limit": limit, "pass": value <= limit, "criterion": criterion}
    def maximum(name, function, points, criterion):
        value = float(np.abs(evaluate(function, points, derivatives=True)).max())
        record(name, value, criterion)
    time = times_for(config, c)
    tau = time/c["time_reference_s"]
    t = torch.tensor(tau[:, None], dtype=torch.float64)
    fixed = {}
    for i, (left, right) in enumerate(model.bounds):
        x = np.linspace(left, right, a["region_x_points"])
        xt = grid(x, tau)
        fixed[i] = xt
        k = 0 if i == 0 else 1
        def total_current(p):
            zero = p.new_tensor(0.)
            ie = electrolyte_charge_terms(model.ce[i], model.phie[i], p, zero, model.charge_scales,
                                           porosity=c["porosities"][i])["current_A_m2"]
            solid = 0 if i == 1 else solid_charge_terms(model.phis[k], p, zero, model.charge_scales,
                           conductivity_S_m=c["solid_conductivities_S_m"][k])["current_A_m2"]
            return (ie+solid)/model.charge_scales.current_A_m2-1
        maximum(f"total_current_region_{i}", total_current, xt, "max_total_current_relative_error")
        ce = evaluate(model.ce[i], xt)
        if np.min(ce) <= 0:
            raise ValueError("Nonpositive electrolyte concentration")
        diagnostics[f"ce_{i}_range_normalized"] = [float(ce.min()), float(ce.max())]
        initial = grid(x, np.array([0.]))
        record(f"initial_ce_{i}", np.abs(evaluate(model.ce[i], initial)-1).max(), "max_initial_normalized_concentration_error")
        if i == 1:
            continue
        rho = np.linspace(0, 1, a["particle_r_points"])
        theta = evaluate(model.cs[k], grid(rho, x, tau))
        if theta.min() <= 0 or theta.max() >= 1:
            raise ValueError("Solid stoichiometry outside open (0,1)")
        diagnostics[f"theta_{i}_range"] = [float(theta.min()), float(theta.max())]
        diagnostics[f"j_{i}_range_A_m2"] = [float(v) for v in (evaluate(model.reactions[k].integral.current_model, xt).min(), evaluate(model.reactions[k].integral.current_model, xt).max())]
        initial_cs = evaluate(model.cs[k], grid(rho, x, np.array([0.])))
        record(f"initial_cs_{i}", np.abs(initial_cs-c["initial_stoichiometries"][k]).max(), "max_initial_normalized_concentration_error")
        center = np.column_stack((np.zeros(len(xt)), xt))
        maximum(f"center_{i}", lambda p: center_residual(model.cs[k], p), center, "max_center_normalized_gradient")
        def interface(p):
            def fields(z):
                return {"c_e": c["initial_ce_mol_m3"]*model.ce[i](z),
                        "phi_e": model.charge_scales.potential_V*model.phie[i](z),
                        "phi_s": model.charge_scales.potential_V*model.phis[k](z),
                        "T": torch.full_like(z[:, :1], c["temperature_K"])}
            return particle_interface(model.reactions[k], model.cs[k], fields, p, 'n' if k == 0 else 'p',
                                      model.jref[k], mode=c["kinetics_mode"])
        for key in ("flux_residual", "kinetics_residual"):
            maximum(f"{key}_{i}", lambda p: interface(p)[key], xt, "max_normalized_flux_or_kinetics_residual")
    for k, position in enumerate((model.bounds[0][1], model.bounds[1][1])):
        for name, v in electrolyte_interface(model.branch(k), model.branch(k+1), t, position, model.mass_scales, model.charge_scales).items():
            record(f"interface_{k}_{name}", v.detach().abs().max(), "max_normalized_boundary_or_interface_residual")
        v = solid_separator_condition(model.phis[k], t, position, model.charge_scales, conductivity_S_m=c["solid_conductivities_S_m"][k])
        record(f"solid_insulating_{k}", v.detach().abs().max(), "max_normalized_boundary_or_interface_residual")
    for side, i, k in (("negative", 0, 0), ("positive", 2, 1)):
        bc = collector_conditions(side, model.branch(i), model.phis[k], t, model.mass_scales, model.charge_scales,
                                 conductivity_S_m=c["solid_conductivities_S_m"][k], applied_current=t.new_tensor(model.charge_scales.current_A_m2))
        for name, v in bc["conditions"].items():
            criterion = "max_normalized_gauge_residual" if name == "solid_potential_gauge" else "max_normalized_boundary_or_interface_residual"
            record(f"collector_{side}_{name}", v.detach().abs().max(), criterion)
        record(f"collector_{side}_total_current", bc["total_current_diagnostic"].detach().abs().max(), "max_total_current_relative_error")

    previous, gaps, curves = None, {}, None
    orders = list(a["quadrature_orders"])+[a["quadrature_refinement_order"]]
    for order in orders:
        print(f"Independent inventory quadrature order {order}...", flush=True)
        curves = inventory_curves(model, np.r_[0., time], order)
        values = {
            "total_lithium_drift": (float(np.abs(curves["total_mol"]-curves["total_mol"][0]).max()/abs(curves["total_mol"][0])), "max_total_lithium_drift_over_initial"),
            "electrolyte_lithium_drift": (float(np.abs(curves["electrolyte_mol"]-curves["electrolyte_mol"][0]).max()/abs(curves["electrolyte_mol"][0])), "max_electrolyte_lithium_drift_over_initial"),
            "reaction_current_integrals": (reaction_error(curves["electrode_current_A"], c["applied_current_A"]), "max_electrode_integrated_reaction_current_relative_error")}
        for k, i in enumerate((0, 2)):
            values[f"local_inventory_{i}"] = (float(np.abs(local_inventory_error(model, k, fixed[i], order)).max()), "max_local_particle_inventory_error")
        if previous is not None:
            gaps = {name: abs(v-previous[name][0])/limits[criterion] for name, (v, criterion) in values.items()}
            if max(gaps.values()) <= limits["quadrature_metric_gap_over_metric_limit"]:
                break
        previous = values
    for name, (value, criterion) in values.items():
        record(name, value, criterion)
    for name, gap in gaps.items():
        record(f"quadrature_gap_{name}", gap, "quadrature_metric_gap_over_metric_limit")
    transfer = abs(c["applied_current_A"])*c["duration_s"]/FARADAY_CONSTANT
    drift = float(np.abs(curves["total_mol"]-curves["total_mol"][0]).max())
    diagnostics.update({"time_points": len(time), "quadrature_final_order": order,
                        "max_total_lithium_drift_mol": drift, "drift_over_transferred_lithium": drift/transfer,
                        "initial_total_lithium_mol": float(curves["total_mol"][0])})
    return {"status": "PARTIAL_PHYSICAL_AUDIT_NOT_ACCEPTANCE", "metrics": metrics, "diagnostics": diagnostics,
            "missing_gates": ["Independent per-equation PDE RMS and sampled maxima", "Combined native-field reference audit and full-budget completion checks"],
            "scope": "Frozen-field balances, boundaries, kinetics, admissibility and initial conditions. No training. Shared verified local operators; independent samples and inventory integration."}
