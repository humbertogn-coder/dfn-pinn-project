"""Reference-aware diagnostics; no reference data enters the training losses."""

import hashlib
import json
import math
import numpy as np
import torch

from .coupled_audit import (grid, predict, boundary_values, mean_values,
                            target_values, physical_pde_rms, pde_values)


def reference_identity(directory, config):
    report = json.loads((directory/"report.json").read_text())
    if report["status"] != "SAMPLED_REFERENCE_CHECKS_PASS":
        raise ValueError("Reference checks did not pass")
    mapping = {"radius_m": "radius_m", "D_m2_s": "diffusivity_m2_s",
               "cmax_mol_m3": "concentration_scale_mol_m3", "T_K": "temperature_K",
               "ce_mol_m3": "electrolyte_concentration_mol_m3", "initial_stoichiometry": "initial_mean",
               "end_time_s": "duration_s", "mode": "exchange_current_mode"}
    for key, config_key in mapping.items():
        if report["settings"][key] != config[config_key]:
            raise ValueError(f"Reference setting mismatch: {key}")
    if config["electrode"] != "n" or config["potential_amplitude_V"] != .005 or config["electrolyte_potential_V"] != 0:
        raise ValueError("Reference protocol mismatch")
    return {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ("report.json", "reference.npz")}


def assess(metrics, limits, smoke):
    if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite potential audit metrics")
    checks = {key: metrics[key] <= value for key, value in limits.items()
              if key != "sign_absolute_tolerance_A_m2"}
    tolerance = limits["sign_absolute_tolerance_A_m2"]
    checks.update(current_sign=metrics["minimum_current_A_m2"] >= -tolerance,
                  flux_sign=metrics["minimum_outward_flux_A_m2"] >= -tolerance,
                  admissibility=0 < metrics["minimum_concentration"] <= metrics["maximum_concentration"] < 1,
                  pde_quadrature_gap=metrics["pde_quadrature_gap"] <= 1e-4)
    return {"status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else "SAMPLED_TARGETS_PASS" if all(checks.values()) else "SAMPLED_TARGETS_FAIL",
            "checks": checks, "all_sampled_targets_pass": all(checks.values())}


def audit(model, directory):
    c = model.config
    with np.load(directory/"reference.npz", allow_pickle=False) as data:
        seconds = torch.tensor(data["times_s"], dtype=torch.float64)
        r = torch.tensor(data["r320_tight_nodes"], dtype=torch.float64)
        exact = torch.tensor(data["r320_tight_Stoichiometry"], dtype=torch.float64)
        surface = torch.tensor(data["r320_tight_Surface"], dtype=torch.float64)
        current = torch.tensor(data["r320_tight_Current"], dtype=torch.float64)
        potential = torch.tensor(data["r320_tight_Potential"], dtype=torch.float64)
    if seconds.ndim != 1 or seconds[0] != 0 or seconds[-1] != c["duration_s"] or not bool((seconds.diff() > 0).all()):
        raise ValueError("Invalid reference time axis")
    if exact.shape != (len(r), len(seconds)) or any(v.shape != seconds.shape for v in (surface, current, potential)):
        raise ValueError("Reference axes mismatch")
    if not all(torch.isfinite(v).all() for v in (seconds, r, exact, surface, current, potential)):
        raise ValueError("Nonfinite reference")
    times = seconds/c["time_reference_s"]
    xt = torch.stack((torch.zeros_like(times), times), dim=1)
    torch.testing.assert_close(model.fields(xt)["phi_s"].detach().flatten(), potential, rtol=0, atol=1e-12)
    predicted = predict(model.concentration, grid(r, times)).reshape_as(exact)
    predicted_surface = predict(model.concentration, grid(torch.ones(1, dtype=torch.float64), times)).flatten()
    predicted_current = predict(model.current.integral.current_model, xt).flatten()
    concentration_error = (predicted-exact).abs()
    surface_error = (predicted_surface-surface).abs()
    current_error = (predicted_current-current).abs()
    boundary_times = times[1:]
    boundary = boundary_values(model, boundary_times)
    additions = [boundary_times]
    for col in (1, 2):
        peak = int(boundary[:, col].abs().argmax())
        additions.append(torch.linspace(float(boundary_times[max(0, peak-1)]),
                                         float(boundary_times[min(len(boundary_times)-1, peak+1)]), 81, dtype=torch.float64))
    refined_times = torch.unique(torch.cat(additions))
    boundary = boundary_values(model, refined_times)
    means = [mean_values(model, times, order) for order in (128, 256)]
    targets = [target_values(model, times, order) for order in (64, 128)]
    rms = [physical_pde_rms(model, order) for order in (64, 128)]
    validation_r = torch.linspace(0, 1, c["validation_radial_points"], dtype=torch.float64)
    validation_t = torch.linspace(0, times[-1], c["validation_time_points"], dtype=torch.float64)
    validation_points = grid(validation_r, validation_t)
    bounds = predict(model.concentration, validation_points)
    initial = predict(model.concentration, grid(validation_r, times[:1]))
    significant = current.abs() > 1e-4
    metrics = {"max_concentration_error": float(concentration_error.max()),
               "max_surface_concentration_error": float(surface_error.max()),
               "max_current_error_A_m2": float(current_error.max()),
               "max_surface_flux_residual_A_m2": float(boundary[:, 1].abs().max()),
               "max_kinetics_residual_A_m2": float(boundary[:, 2].abs().max()),
               "max_inventory_error": float((means[-1]-targets[-1]).abs().max()),
               "max_initial_profile_error": float((initial-c["initial_mean"]).abs().max()),
               "physical_pde_rms": rms[-1], "pde_quadrature_gap": abs(rms[-1]-rms[0]),
               "quadrature_mean_gap": max(float((means[0]-means[1]).abs().max()), float((targets[0]-targets[1]).abs().max())),
               "minimum_current_A_m2": float(boundary[:, 0].min()),
               "minimum_outward_flux_A_m2": float(boundary[:, 3].min()),
               "maximum_current_A_m2": float(boundary[:, 0].max()),
               "maximum_outward_flux_A_m2": float(boundary[:, 3].max()),
               "minimum_concentration": min(float(bounds.min()), float(predicted.min()), float(predicted_surface.min())),
               "maximum_concentration": max(float(bounds.max()), float(predicted.max()), float(predicted_surface.max())),
               "max_sampled_pde_residual": float(pde_values(model, validation_points).abs().max()),
               "max_relative_current_error_above_1e_4": float((current_error[significant]/current[significant].abs()).max()) if significant.any() else 0.}
    ri, ti = divmod(int(concentration_error.argmax()), len(times))
    peaks = {"internal": {"rho": float(r[ri]), "time_s": float(seconds[ti])},
             "surface_time_s": float(seconds[int(surface_error.argmax())]),
             "current_time_s": float(seconds[int(current_error.argmax())]),
             "flux_time_s": float(refined_times[int(boundary[:, 1].abs().argmax())]*c["time_reference_s"]),
             "kinetics_time_s": float(refined_times[int(boundary[:, 2].abs().argmax())]*c["time_reference_s"])}
    return metrics, {"reference_radial_points": len(r), "reference_times": len(times),
                     "boundary_times": len(refined_times), "peaks": peaks,
                     "scope": "Native reference samples without interpolation. Residual-only local refinement; no continuous bounds.",
                     "pde_orders": [64, 128], "mean_orders": [128, 256], "time_integral_orders": [64, 128]}
