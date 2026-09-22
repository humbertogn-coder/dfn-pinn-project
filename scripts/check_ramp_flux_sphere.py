"""Independent series/FV benchmark for a sphere driven by a linear flux ramp."""

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybamm

from check_constant_flux_sphere import eigenvalues


END_TIME = .02
SLOPE = 1/END_TIME


def ramp_solution(radius, time, roots, slope=SLOPE):
    """Duhamel integral of the unit-step concentration response minus one."""
    r, t, a = (np.asarray(value, dtype=float) for value in (radius, time, roots))
    if (r.ndim != 1 or t.ndim != 1 or a.ndim != 1 or not r.size or not t.size or not a.size
            or not all(np.isfinite(v).all() for v in (r, t, a))
            or np.any((r < 0) | (r > 1)) or np.any(t < 0) or np.any(a <= 0)
            or not np.isfinite(slope)):
        raise ValueError("Require finite radial/time arrays in domain and positive roots")
    modes = np.sinc(r[:, None]*a[None, :]/np.pi)
    integrated_modes = -np.expm1(-a[:, None]**2*t[None, :])
    transient = (modes*(2/(a**4*np.cos(a)))) @ integrated_modes
    return 1+slope*(-1.5*t[None, :]**2+(.3-r[:, None]**2/2)*t[None, :]+transient)


def solve_ramp(points, times):
    model = pybamm.BaseModel()
    r = pybamm.SpatialVariable("r", domain=["sphere"], coord_sys="spherical polar")
    c = pybamm.Variable("Concentration", domain=["sphere"])
    model.rhs = {c: pybamm.div(pybamm.grad(c))}
    model.initial_conditions = {c: pybamm.Scalar(1)}
    model.boundary_conditions = {c: {"left": (pybamm.Scalar(0), "Neumann"),
                                      "right": (-SLOPE*pybamm.t, "Neumann")}}
    model.variables = {"Concentration": c, "Surface": pybamm.boundary_value(c, "right")}
    mesh = pybamm.Mesh({"sphere": {r: {"min": 0, "max": 1}}},
                      {"sphere": pybamm.Uniform1DSubMesh}, {r: points})
    pybamm.Discretisation(mesh, {"sphere": pybamm.FiniteVolume()}).process_model(model)
    solution = pybamm.IDAKLUSolver(rtol=1e-11, atol=1e-13).solve(model, times, t_interp=times)
    values = np.asarray(solution["Concentration"].entries)
    surface = np.asarray(solution["Surface"].entries).reshape(-1)
    if (values.shape != (points, len(times)) or surface.shape != times.shape
            or not np.isfinite(values).all() or not np.isfinite(surface).all()
            or not np.allclose(solution.t, times, atol=1e-12, rtol=0)):
        raise ValueError("Unexpected finite-volume solution")
    return mesh["sphere"].nodes, mesh["sphere"].edges, values, surface


def main():
    times = np.r_[0., np.geomspace(1e-5, END_TIME, 180)]
    roots = eigenvalues(4096)
    reference_surface = ramp_solution([1.], times, roots)[0]
    exact_mean = 1-1.5*SLOPE*times**2
    records, arrays = [], {}
    for size in (80, 160, 320):
        print(f"Running linear-ramp sphere: r{size}...", flush=True)
        nodes, edges, values, surface = solve_ramp(size, times)
        radius = np.r_[nodes, 1.]
        exact = ramp_solution(radius, times, roots)
        check_radius = np.r_[0., radius]
        truncation = float(np.max(np.abs(ramp_solution(check_radius, times, roots)
                                         -ramp_solution(check_radius, times, roots[:2048]))))
        if truncation > 1e-9:
            raise ValueError("Series truncation sensitivity exceeds 1e-9")
        mean = (edges[1:]**3-edges[:-1]**3) @ values
        record = {"radial_points": size, "series_2048_4096_max_difference": truncation,
                  "max_mean_balance_error": float(np.max(np.abs(mean-exact_mean)))}
        for field, error in (("particle", values-exact[:-1]), ("surface", surface-exact[-1])):
            record[field+"_initial_error"] = float(np.max(np.abs(error[..., 0])))
            record[field+"_positive_max_error"] = float(np.max(np.abs(error[..., 1:])))
            record[field+"_positive_rms_error"] = float(np.sqrt(np.mean(error[..., 1:]**2)))
        records.append(record)
        arrays.update({f"nodes_{size}": nodes, f"values_{size}": values,
                       f"surface_{size}": surface, f"reference_{size}": exact[:-1], f"mean_{size}": mean})
        print(f"r{size}: particle={record['particle_positive_max_error']:.6e}, "
              f"surface={record['surface_positive_max_error']:.6e}, "
              f"mass={record['max_mean_balance_error']:.6e}, series_gap={truncation:.3e}")
    output = Path(__file__).resolve().parents[1]/"results"/("ramp_flux_sphere_"+
             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    report = {"protocol": "q(t)=t/0.02 on [0,0.02]", "mean": "1 - 3*t^2/(2*0.02)",
              "initial_concentration": 1., "end_time": END_TIME, "slope": SLOPE,
              "pybamm_version": pybamm.__version__, "solver": "IDAKLUSolver", "rtol": 1e-11,
              "atol": 1e-13, "series_modes": 4096, "metrics": records,
              "scope": "Synthetic diffusion only; no PINN training or DFN coupling. Linear ramp, "
                       "not a constant-flux switch or arbitrary protocol. Native FV values compared "
                       "with series at native centers; includes representation error. Sampled maxima, "
                       "log-time unweighted RMS; truncation difference is not a rigorous bound."}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(output/"samples.npz", times=times, flux=SLOPE*times,
                        exact_mean=exact_mean, reference_surface=reference_surface, **arrays)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    axes[0].plot(times, SLOPE*times)
    axes[0].set(xlabel="Dimensionless time", ylabel="Outward flux", title="Linear ramp")
    axes[1].plot(times, reference_surface, label="Series", color="black")
    for size in (80, 160, 320):
        axes[1].plot(times, arrays[f"surface_{size}"], "--", label=f"r{size}")
        axes[2].loglog(times[1:], np.abs(arrays[f"surface_{size}"][1:]-reference_surface[1:]), label=f"r{size}")
    axes[1].set(xlabel="Dimensionless time", ylabel="Concentration", title="Surface response")
    axes[2].set(xlabel="Dimensionless time", ylabel="Absolute error", title="Surface error")
    axes[1].legend()
    axes[2].legend()
    fig.savefig(output/"ramp_flux_sphere.png", dpi=160)
    plt.close(fig)
    print(f"No training was run. Output directory: {output}")


if __name__ == "__main__":
    main()
