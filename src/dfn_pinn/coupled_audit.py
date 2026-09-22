"""Independent sampling and quadrature diagnostics for a frozen particle model."""

import math
import torch

from .coupled_training import reference
from .current_integral import CurrentIntegral
from .particle_balance import particle_mean_from_charge
from .particle_interface import particle_interface
from .projection import gauss_legendre
from .spherical_diffusion import diffusion_residual


def checked(value):
    if not torch.isfinite(value).all():
        raise FloatingPointError("Nonfinite audit value")
    return value.detach()


def grid(radius, times):
    r, t = torch.meshgrid(radius, times, indexing="ij")
    return torch.stack((r.flatten(), torch.zeros_like(r).flatten(), t.flatten()), dim=1)


def predict(model, points):
    with torch.no_grad():
        return torch.cat([checked(model(p)) for p in points.split(256)])


def pde_values(model, points):
    values = []
    for part in points.split(256):
        p = part.clone().requires_grad_()
        values.append(checked(diffusion_residual(model.concentration, p, model.current.scales)))
    return torch.cat(values)


def physical_pde_rms(model, order):
    end = model.config["duration_s"]/model.config["time_reference_s"]
    r, wr = gauss_legendre(order, 0., 1.)
    t, wt = gauss_legendre(order, 0., end)
    values = pde_values(model, grid(r, t)).reshape(order, order)
    return float(torch.sqrt((values.square()*(3*r*r*wr)[:, None]*(wt/end)[None, :]).sum()))


def boundary_values(model, times):
    c = model.config
    outputs = []
    for t in times.split(128):
        p = torch.stack((torch.zeros_like(t), t), dim=1).requires_grad_()
        interface = particle_interface(model.current, model.concentration, model.fields, p,
                                       c["electrode"], c["current_scale_A_m2"], mode=c["exchange_current_mode"])
        current = model.current.integral.current_model(p)
        flux = interface["flux_residual"]*c["current_scale_A_m2"]
        kinetics = interface["kinetics_residual"]*c["current_scale_A_m2"]
        outputs.append(checked(torch.cat((current, flux, kinetics, current+flux), dim=1)))
    return torch.cat(outputs)


def mean_values(model, times, order):
    r, w = gauss_legendre(order, 0., 1.)
    concentration = predict(model.concentration, grid(r, times)).reshape(order, len(times))
    return (concentration*(3*r*r*w)[:, None]).sum(0)


def target_values(model, times, order):
    p = torch.stack((torch.zeros_like(times), times), dim=1)
    integral = CurrentIntegral(model.current.integral.current_model, model.config["time_reference_s"], order)
    with torch.no_grad():
        charge = torch.cat([integral(part) for part in p.split(128)])
        return checked(particle_mean_from_charge(p.new_tensor(model.config["initial_mean"]),
                                                 charge, model.current.scales)).flatten()


def assess(metrics, limits, smoke):
    if any(not math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Nonfinite audit metrics")
    checks = {key: metrics[key] <= limit for key, limit in limits.items()}
    checks.update({"positive_current": metrics["minimum_current_A_m2"] > 0,
                   "positive_outward_flux": metrics["minimum_outward_flux_A_m2"] > 0,
                   "admissible_concentration": 0 < metrics["minimum_concentration"] <= metrics["maximum_concentration"] < 1,
                   "pde_quadrature_gap": metrics["pde_quadrature_gap"] <= 1e-4})
    # Never promote a smoke run, regardless of its individual diagnostics.
    status = "SMOKE_DIAGNOSTIC_ONLY" if smoke else "SAMPLED_TARGETS_PASS" if all(checks.values()) else "SAMPLED_TARGETS_FAIL"
    return {"status": status, "checks": checks, "all_sampled_targets_pass": all(checks.values())}


def audit(model, *, nr=201, nt=401, orders=(64, 128)):
    c = model.config
    end = c["duration_s"]/c["time_reference_s"]
    r = torch.linspace(0, 1, nr, dtype=torch.float64)
    t = torch.linspace(0, end, nt, dtype=torch.float64)
    points = grid(r, t)
    concentration = predict(model.concentration, points).reshape(nr, nt)
    error = (concentration-reference(points, c).reshape(nr, nt)).abs()
    peak = int(error.argmax())
    ri, ti = divmod(peak, nt)
    # Refine one coarse cell either side of the largest sampled error.
    fine_r = torch.linspace(float(r[max(0, ri-1)]), float(r[min(nr-1, ri+1)]), 41, dtype=torch.float64)
    fine_t = torch.linspace(float(t[max(0, ti-1)]), float(t[min(nt-1, ti+1)]), 81, dtype=torch.float64)
    local = grid(fine_r, fine_t)
    local_c = predict(model.concentration, local)
    local_error = (local_c-reference(local, c)).abs()
    boundary = boundary_values(model, t[1:])
    time_sets = [t[1:]]
    for values in ((boundary[:, 0]-c["reference_current_A_m2"]).abs(), boundary[:, 1].abs(), boundary[:, 2].abs()):
        idx = int(values.argmax())+1
        time_sets.append(torch.linspace(float(t[max(0, idx-1)]), float(t[min(nt-1, idx+1)]), 81, dtype=torch.float64))
    bt = torch.unique(torch.cat(time_sets))
    bt = bt[bt > 0]
    boundary = boundary_values(model, bt)
    means = [mean_values(model, t, order) for order in (128, 256)]
    targets = [target_values(model, t, order) for order in (64, 128)]
    pde_rms = [physical_pde_rms(model, order) for order in orders]
    metrics = {
        "max_concentration_error": max(float(error.max()), float(local_error.max())),
        "max_current_error_A_m2": float((boundary[:, 0]-c["reference_current_A_m2"]).abs().max()),
        "max_surface_flux_residual_A_m2": float(boundary[:, 1].abs().max()),
        "max_kinetics_residual_A_m2": float(boundary[:, 2].abs().max()),
        "max_inventory_error": float((means[-1]-targets[-1]).abs().max()),
        "max_initial_profile_error": float(error[:, 0].max()),
        "physical_pde_rms": pde_rms[-1],
        "pde_quadrature_gap": abs(pde_rms[0]-pde_rms[1]),
        "quadrature_mean_gap": max(float((means[0]-means[1]).abs().max()), float((targets[0]-targets[1]).abs().max())),
        "minimum_concentration": min(float(concentration.min()), float(local_c.min())),
        "maximum_concentration": max(float(concentration.max()), float(local_c.max())),
        "minimum_current_A_m2": float(boundary[:, 0].min()),
        "minimum_outward_flux_A_m2": float(boundary[:, 3].min()),
        "max_sampled_pde_residual": float(pde_values(model, points).abs().max()),
    }
    return metrics, {"nr": nr, "nt": nt, "pde_orders": list(orders),
                     "boundary_samples": len(bt), "radial_mean_orders": [128, 256],
                     "time_integral_orders": [64, 128], "local_concentration_grid": [41, 81],
                     "scope": "Sampled diagnostics, not continuous bounds; no training."}
