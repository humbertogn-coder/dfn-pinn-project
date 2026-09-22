"""Audit saved ramp extrema and inventory without training or changing weights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_ramp_flux_particle import build_ramp_model, target_mean, flux
from train_constant_flux_particle import predict, make_points, MIN_TIME, END_TIME
from check_ramp_flux_sphere import eigenvalues, ramp_solution
from dfn_pinn.projection import gauss_legendre


def peak_record(values, radii, times, absolute=False):
    values = np.asarray(values)
    if values.shape != (len(radii), len(times)) or not np.isfinite(values).all():
        raise ValueError("Invalid peak field")
    objective = np.abs(values) if absolute else values
    index = np.unravel_index(objective.argmax(), objective.shape)
    return {"value": float(objective[index]), "signed_value": float(values[index]),
            "rho": float(radii[index[0]]), "tau": float(times[index[1]])}


def global_grid():
    return np.linspace(0, 1, 201), np.geomspace(MIN_TIME, END_TIME, 401)


def flux_peak_record(residual, times):
    peak = peak_record(np.asarray(residual)[None, :], [1.], times, True)
    imposed = float(flux(peak["tau"]))
    peak.update(imposed_outward_flux=imposed,
                predicted_outward_flux=imposed-peak["signed_value"],
                error_over_instantaneous_flux=peak["value"]/imposed if imposed else None)
    return peak


def local_grid(peak):
    return (np.linspace(max(0., peak["rho"]-.025), min(1., peak["rho"]+.025), 81),
            np.geomspace(max(MIN_TIME, peak["tau"]/2), min(END_TIME, peak["tau"]*2), 161))


def field(model, r, t):
    rr, tt = np.meshgrid(r, t, indexing="ij")
    with torch.no_grad():
        result = predict(model, make_points(torch.tensor(rr.ravel()), torch.tensor(tt.ravel()))).numpy().reshape(rr.shape)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite model field")
    return result


def nested_records(prediction, reference, r, t):
    rows = []
    for stride in (4, 2, 1):
        p, e = prediction[::stride, ::stride], reference[::stride, ::stride]
        rows.append({"shape": list(p.shape),
                     "concentration_error": peak_record(p-e, r[::stride], t[::stride], True),
                     "overshoot_above_one": peak_record(p-1, r[::stride], t[::stride])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Completed full-budget ramp run")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [args.run] if args.run else sorted((root/"results").glob("ramp_flux_pinn_*"), reverse=True)
    selected = None
    for path in paths:
        if not (path/"report.json").is_file():
            continue
        report = json.loads((path/"report.json").read_text(encoding="utf-8"))
        if report["config"] == {"seed": 42, "adam_steps": 2000, "lbfgs_steps": 300}:
            selected = path
            break
    if selected is None:
        raise ValueError("No full-budget seed-42 ramp run found")
    expected = {"protocol": "linear_ramp", "slope": 50., "end_time": END_TIME,
                "minimum_positive_time": MIN_TIME, "projection_order": 64, "dtype": "float64",
                "torch_version": torch.__version__, "architecture": [3, 32, 32, 1], "sampling": "full_radius"}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported run configuration")
    hashes = {name: hashlib.sha256((selected/name).read_bytes()).hexdigest()
              for name in ("model.pt", "training_points.npz", "report.json", "evaluation.npz")}
    if hashes["model.pt"] != report["model_sha256"] or hashes["training_points.npz"] != report["training_points_sha256"]:
        raise ValueError("Saved artifact checksum mismatch")
    torch.set_num_threads(1)
    model = build_ramp_model()
    model.load_state_dict(torch.load(selected/"model.pt", map_location="cpu", weights_only=True), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with np.load(selected/"evaluation.npz", allow_pickle=False) as saved:
        replay = field(model, saved["rho"], saved["tau"])
        reload_gap = float(np.abs(replay-saved["prediction"]).max())
        if not np.allclose(replay, saved["prediction"], atol=1e-12, rtol=1e-12):
            raise ValueError("Saved predictions do not reproduce")
    print(f"Auditing saved ramp: {selected}", flush=True)
    roots = eigenvalues(4096)
    r, t = global_grid()
    prediction = field(model, r, t)
    reference = ramp_solution(r, t, roots)
    truncation = float(np.abs(reference-ramp_solution(r, t, roots[:2048])).max())
    if truncation > 1e-9:
        raise ValueError("Reference series check failed")
    global_records = nested_records(prediction, reference, r, t)
    local = {}
    arrays = {"rho": r, "tau": t, "prediction": prediction, "reference": reference}
    for name, peak in (("error", global_records[-1]["concentration_error"]),
                       ("overshoot", global_records[-1]["overshoot_above_one"])):
        lr, lt = local_grid(peak)
        lp, le = field(model, lr, lt), ramp_solution(lr, lt, roots)
        local_gap = float(np.abs(le-ramp_solution(lr, lt, roots[:2048])).max())
        if local_gap > 1e-9:
            raise ValueError("Local reference truncation check failed")
        local[name] = {"records": nested_records(lp, le, lr, lt), "series_gap": local_gap}
        arrays.update({name+"_rho": lr, name+"_tau": lt, name+"_prediction": lp, name+"_reference": le})
    st = np.unique(np.r_[np.geomspace(MIN_TIME, END_TIME, 2001), arrays["error_tau"]])
    surface = field(model, [1.], st)
    surface_error = peak_record(surface-ramp_solution([1.], st, roots), [1.], st, True)
    surface_flux = []
    for part in np.array_split(st, int(np.ceil(len(st)/64))):
        tau = torch.tensor(part)
        p = make_points(torch.ones_like(tau), tau)
        dr = torch.autograd.grad(model(p).sum(), p)[0][:, 0].detach().numpy()
        surface_flux.extend(dr+flux(part))
    flux_peak = flux_peak_record(surface_flux, st)
    means = {}
    mass_times = np.unique(np.r_[0., t, arrays["overshoot_tau"], arrays["error_tau"]])
    print("Checking independent mass quadratures at refined times...", flush=True)
    for order in (128, 256, 512):
        qr, w = gauss_legendre(order, 0., 1.)
        means[order] = (3*qr.numpy()**2*w.numpy()) @ field(model, qr.numpy(), mass_times)
    mass = {str(order): float(np.abs(values-target_mean(mass_times)).max()) for order, values in means.items()}
    payload = {"source": str(selected.resolve()), "source_hashes": hashes, "reload_max_difference": reload_gap,
               "global_nested": global_records, "local_refinement": local,
               "surface_error": surface_error, "surface_flux_residual": flux_peak,
               "mass_max_error_by_order": mass,
               "mass_256_512_max_difference": float(np.abs(means[256]-means[512]).max()),
               "global_series_2048_4096_gap": truncation,
               "minimum_sampled_concentration": float(min(prediction.min(), *(arrays[k].min() for k in ("error_prediction", "overshoot_prediction")))),
               "scope": "No training. Frozen seed-42 linear-ramp model. Nested sampled grids and local refinements, "
                        "not continuous-extremum bounds. Positive times [1e-5,0.02]; earlier positive times unresolved. "
                        "Overshoot above initial concentration one is retained, not clipped. "
                        "Single-seed ramp audit; no DFN, multi-seed ramp or general-protocol certification."}
    output = root/"results"/("ramp_pilot_audit_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    (output/"report.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(output/"samples.npz", **arrays, mass_times=mass_times,
                        mean_128=means[128], mean_256=means[256], mean_512=means[512],
                        surface_times=st, surface_flux_residual=surface_flux)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ax[0].semilogx(st, np.abs(surface[0]-ramp_solution([1.], st, roots)[0]))
    ax[0].set(xlabel="Dimensionless time", ylabel="Absolute concentration error", title="Refined surface error")
    color = ax[1].pcolormesh(arrays["overshoot_tau"], arrays["overshoot_rho"],
                            arrays["overshoot_prediction"]-1, shading="auto")
    ax[1].set(xlabel="Dimensionless time", ylabel="r/R", title="Concentration minus initial value")
    fig.colorbar(color, ax=ax[1])
    fig.savefig(output/"audit.png", dpi=160)
    plt.close(fig)
    for row in global_records:
        print(f"Global {row['shape']}: error={row['concentration_error']['value']:.6e}, "
              f"overshoot={row['overshoot_above_one']['value']:.6e}")
    print(f"Refined surface error: {surface_error}")
    print(f"Refined overshoot: {local['overshoot']['records'][-1]['overshoot_above_one']}")
    print(f"Refined flux residual: {flux_peak}")
    print(f"Independent mass errors: {mass}")
    print(f"No training was run. Output directory: {output}")


if __name__ == "__main__":
    main()
