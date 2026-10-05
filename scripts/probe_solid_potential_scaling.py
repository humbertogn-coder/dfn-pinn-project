"""Matched 20-step C0 amplitude probe, not a full-run checkpoint or acceptance audit."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from dfn_pinn.dfn_smoke import DFNSmoke, SETTINGS
from dfn_pinn.dfn_run_contract import adam_step, verify_physics
from dfn_pinn.solid_potential_scaling import ohmic_amplitude
from prepare_dfn_baseline import make_samples, sha256, verify_reference


def measure(model, samples):
    residuals, diagnostics = model.residuals(samples)
    result = {}
    for group, values in (("residuals", residuals), ("diagnostics", diagnostics)):
        result[group] = {}
        for name, value in values.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(name)
            result[group][name] = {
                "rms": float(value.detach().square().mean().sqrt()),
                "max_abs": float(value.detach().abs().max()),
            }
    result["total_mse"] = sum(v["rms"]**2 for v in result["residuals"].values())
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/dfn_baseline_v1.json"
    config = json.loads(config_path.read_text())
    reference = verify_reference(root, config)
    settings = dict(SETTINGS)
    for key in ("interior_points_per_region", "boundary_times", "inventory_quadrature"):
        settings[key] = config["training"][key]
    settings.update(adam_steps=20, seed=config["seed"], learning_rate=config["training"]["adam_learning_rate"])
    verify_physics(settings, reference["settings"])
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    initial = DFNSmoke(settings)
    models = {"original": copy.deepcopy(initial), "ohmic": copy.deepcopy(initial)}
    for k, field in enumerate(models["ohmic"].phis):
        field.amplitude = ohmic_amplitude(initial.charge_scales, settings["solid_conductivities_S_m"][k])
    for key, value in models["original"].state_dict().items():
        assert torch.equal(value, models["ohmic"].state_dict()[key]), key
    samples = {k: torch.from_numpy(v) for k, v in make_samples(config, settings).items()}
    fresh_config = dict(config, sampling_seed=20260928)
    fresh = {k: torch.from_numpy(v) for k, v in make_samples(fresh_config, settings).items()}
    output = root / "results" / ("solid_potential_probe_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    sources = [Path(__file__), root/"scripts/prepare_dfn_baseline.py", *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report = {"status": "RUNNING", "scope": "Paired C0 direct-kinetics amplitude probe; no physical acceptance",
              "settings": settings, "config_sha256": sha256(config_path),
              "reference_sha256": reference["reference_sha256"], "torch_version": str(torch.__version__),
              "sampling_seeds": [config["sampling_seed"], fresh_config["sampling_seed"]],
              "source_hashes": {str(p.relative_to(root)): sha256(p) for p in sources}, "variants": {}}

    def write_report():
        temp = output/"report.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(output/"report.json")

    torch.save({"initial_state": initial.state_dict(), "training_samples": samples,
                "fresh_samples": fresh, "settings": settings}, output/"paired_inputs.pt")
    report["inputs_sha256"] = sha256(output/"paired_inputs.pt")
    write_report()
    print(f"Bounded amplitude probe: 20 Adam steps per variant. Output: {output}", flush=True)
    start = time.perf_counter()
    try:
        for name, model in models.items():
            row = {"amplitudes": [f.amplitude for f in model.phis], "history": []}
            report["variants"][name] = row
            row["initial"] = {"training": measure(model, samples), "fresh": measure(model, fresh)}
            optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
            for step in range(20):
                if time.perf_counter()-start > 300:
                    raise TimeoutError("Five-minute probe cap reached at step boundary")
                row["history"].append(adam_step(model, optimizer, samples))
            row["final"] = {"training": measure(model, samples), "fresh": measure(model, fresh)}
            path = output/f"{name}.pt"
            torch.save({"schema": "solid_potential_probe_v1", "model": model.state_dict(),
                        "settings": settings, "amplitudes": row["amplitudes"]}, path)
            saved = torch.load(path, weights_only=True)
            replay = DFNSmoke(saved["settings"])
            for field, amplitude in zip(replay.phis, saved["amplitudes"]):
                field.amplitude = amplitude
            replay.load_state_dict(saved["model"], strict=True)
            assert measure(replay, samples) == row["final"]["training"]
            assert measure(replay, fresh) == row["final"]["fresh"]
            row.update(checkpoint_sha256=sha256(path), replay_verified=True)
            write_report()
            print(f'{name}: training MSE {row["initial"]["training"]["total_mse"]:.6g} -> {row["final"]["training"]["total_mse"]:.6g}', flush=True)
        report["status"] = "BOUNDED_PROBE_COMPLETE_NOT_VALIDATED"
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_s"] = time.perf_counter()-start
        write_report()
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
