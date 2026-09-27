"""Independent sampling and aggregation of the ten local DFN equations.

Local differential operators are shared with training, not independently
reimplemented. No training loss or training collocation points are used.
"""

import numpy as np

from .dfn_balance_audit import evaluate, grid, quadrature, times_for
from .electrolyte_mass import electrolyte_mass_terms
from .charge_conservation import electrolyte_charge_terms, solid_charge_terms
from .spherical_diffusion import diffusion_residual


def equation(model, region, kind):
    """Build one normalized equation directly from frozen physical fields."""
    c = model.settings
    k = 0 if region == 0 else 1
    if kind == "particle":
        return lambda p: diffusion_residual(model.cs[k], p, model.reactions[k].scales)

    def residual(p):
        aj = p[:, :1]*0 if region == 1 else (
            model.active_area[k]*model.reactions[k].integral.current_model(p))
        if kind == "salt":
            terms = electrolyte_mass_terms(model.ce[region], p, aj, model.mass_scales,
                                           porosity=c["porosities"][region])
        elif kind == "charge_e":
            terms = electrolyte_charge_terms(model.ce[region], model.phie[region], p, aj,
                                             model.charge_scales, porosity=c["porosities"][region],
                                             temperature_K=c["temperature_K"])
        elif kind == "charge_s":
            terms = solid_charge_terms(model.phis[k], p, aj, model.charge_scales,
                                       conductivity_S_m=c["solid_conductivities_S_m"][k])
        else:
            raise ValueError(f"Unknown equation: {kind}")
        return terms["normalized_residual"]
    return residual


def physical_rms(function, bounds, settings, minimum_time, order, particle=False):
    """Normalized RMS under physical dx dt and optional spherical volume measure."""
    x, wx = quadrature(order, *bounds)
    time, wt = quadrature(order, minimum_time, settings["duration_s"])
    tau = time/settings["time_reference_s"]
    wx = wx*sum(settings["lengths_m"])
    if particle:
        rho, wr = quadrature(order)
        weights = np.einsum("r,x,t->rxt", 3*rho*rho*wr, wx, wt).ravel()
        points = grid(rho, x, tau)
    else:
        weights = np.outer(wx, wt).ravel()
        points = grid(x, tau)
    values = evaluate(function, points, derivatives=True)
    # Scale before squaring so finite large residuals cannot silently overflow.
    scale = float(np.abs(values).max())
    return 0. if scale == 0 else float(scale*np.sqrt(np.dot(weights, (values/scale)**2)/weights.sum()))


def audit_pdes(model, config):
    """Report per-equation RMS, quadrature sensitivity and sampled peak locations."""
    c, a = model.settings, config["audit"]
    limits = a["criteria"]
    time = times_for(config, c)
    tau = time/c["time_reference_s"]
    metrics, diagnostics = {}, {}
    orders = list(a["quadrature_orders"])+[a["quadrature_refinement_order"]]
    if len(set(orders)) != len(orders) or len(orders) < 2 or any(q < 2 for q in orders):
        raise ValueError("Require distinct quadrature orders >= 2")
    if not 0 < a["minimum_positive_time_s"] < c["duration_s"]:
        raise ValueError("Require a positive nonempty audit time interval")

    def record(name, value, criterion):
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite PDE metric: {name}")
        limit = limits[criterion]
        metrics[name] = {"value": float(value), "limit": limit,
                         "pass": bool(value <= limit), "criterion": criterion}

    for region, bounds in enumerate(model.bounds):
        kinds = ("salt", "charge_e") if region == 1 else ("salt", "charge_e", "charge_s", "particle")
        for kind in kinds:
            name = f"{kind}_{region}"
            print(f"Auditing local PDE {name}...", flush=True)
            function = equation(model, region, kind)
            x = np.linspace(*bounds, a["region_x_points"])
            particle = kind == "particle"
            points = grid(np.linspace(0, 1, a["particle_r_points"]), x, tau) if particle else grid(x, tau)
            values = evaluate(function, points, derivatives=True)
            peak = int(np.argmax(np.abs(values)))
            location = points[peak]
            record(name+"_sampled_max", abs(values[peak]), "per_equation_normalized_pde_sampled_max")
            previous, history = None, []
            for order in orders:
                rms = physical_rms(function, bounds, c, a["minimum_positive_time_s"], order, particle)
                history.append({"order": order, "rms": rms})
                if previous is not None:
                    gap = abs(rms-previous)/limits["per_equation_normalized_pde_rms"]
                    if gap <= limits["quadrature_metric_gap_over_metric_limit"]:
                        break
                previous = rms
            record(name+"_physical_rms", rms, "per_equation_normalized_pde_rms")
            record(name+"_quadrature_gap", gap, "quadrature_metric_gap_over_metric_limit")
            diagnostics[name] = {"sample_count": len(points), "peak_signed_residual": float(values[peak]),
                "peak_time_s": float(location[-1]*c["time_reference_s"]),
                "peak_x_m": float(location[-2]*sum(c["lengths_m"])), "quadrature_history": history}
            if particle:
                diagnostics[name]["peak_rho"] = float(location[0])
    return {"status": "PDE_DIAGNOSTIC_ONLY_NOT_ACCEPTANCE", "metrics": metrics,
        "diagnostics": diagnostics,
        "missing_gates": ["Combined field/balance audit and full-budget completion checks", "Cost/replay dry run"],
        "scope": "Ten local normalized equations; physical-measure RMS and sampled maxima including one-sided traces. Shared local operators, independent samples. No continuous-time bounds, training or acceptance.",
        "time_interval_s": [a["minimum_positive_time_s"], c["duration_s"]]}
