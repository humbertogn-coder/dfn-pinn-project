"""Replay 20 steps and inspect one disposable Adam candidate, never resume a run."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from dfn_pinn.dfn_smoke import DFNSmoke
from dfn_pinn.dfn_run_contract import adam_step
from dfn_pinn.dfn_gradient_diagnosis import vector_gradient
from prepare_dfn_baseline import sha256
from probe_solid_potential_scaling import measure


def main():
    root = Path(__file__).resolve().parents[1]
    run = root/"results/solid_potential_probe_20260928T190218279084Z"
    source = json.loads((run/"report.json").read_text())
    if source["status"] != "BOUNDED_PROBE_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed probe required")
    if source["torch_version"] != str(torch.__version__):
        raise ValueError("Replay requires the original Torch version")
    hashes = {run/"report.json": sha256(run/"report.json"),
              run/"paired_inputs.pt": source["inputs_sha256"]}
    hashes.update({root/k: v for k, v in source["source_hashes"].items()})
    hashes.update({run/f"{n}.pt": source["variants"][n]["checkpoint_sha256"]
                   for n in ("original", "ohmic")})
    for p, digest in hashes.items():
        if sha256(p) != digest:
            raise ValueError(f"Input mismatch: {p}")
    bundle = torch.load(run/"paired_inputs.pt", weights_only=True)
    if bundle["settings"] != source["settings"]:
        raise ValueError("Settings mismatch")
    torch.set_num_threads(1)
    report = {"status": "ADAM_DIRECTION_DIAGNOSTIC_ONLY", "variants": {},
              "scope": "Exact 20-step replay; one disposable candidate plus line checks; no accepted step or physical audit.",
              "input_hashes": {str(p): h for p, h in hashes.items()}}
    start = time.perf_counter()
    for name in ("original", "ohmic"):
        saved = torch.load(run/f"{name}.pt", weights_only=True)
        if saved["settings"] != source["settings"] or saved["amplitudes"] != source["variants"][name]["amplitudes"]:
            raise ValueError("Checkpoint metadata mismatch")
        model = DFNSmoke(bundle["settings"])
        model.load_state_dict(bundle["initial_state"], strict=True)
        for field, amplitude in zip(model.phis, saved["amplitudes"]):
            field.amplitude = amplitude
        optimizer = torch.optim.Adam(model.parameters(), lr=bundle["settings"]["learning_rate"])
        for i in range(20):
            if time.perf_counter()-start > 300:
                raise TimeoutError("Replay cap exceeded")
            entry = adam_step(model, optimizer, bundle["training_samples"])
            if entry != source["variants"][name]["history"][i]:
                raise ValueError(f"History replay mismatch: {name}, step {i+1}")
        for key, value in model.state_dict().items():
            if not torch.equal(value, saved["model"][key]):
                raise ValueError(f"Checkpoint replay mismatch: {key}")
        frozen = copy.deepcopy(model.state_dict())
        parameters = list(model.parameters())
        candidate = copy.deepcopy(model)
        candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=bundle["settings"]["learning_rate"])
        candidate_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        adam_step(candidate, candidate_optimizer, bundle["training_samples"])
        delta = torch.cat([(q-p).detach().reshape(-1) for p, q in zip(parameters, candidate.parameters())])
        if not torch.isfinite(delta).all():
            raise ValueError("Nonfinite Adam direction")
        row = {"exact_history_and_checkpoint_replay": True,
               "adam_delta_norm": float(delta.norm()), "samples": {}}
        for label, key in (("training", "training_samples"), ("fresh", "fresh_samples")):
            samples = bundle[key]
            if measure(model, samples) != source["variants"][name]["final"][label]:
                raise ValueError("Metric replay mismatch")
            residuals, _ = model.residuals(samples)
            losses = {k: v.square().mean() for k, v in residuals.items()}
            terms = {}
            for term, loss in losses.items():
                gradient = vector_gradient(loss, parameters)
                terms[term] = {"initial_mse": float(loss.detach()),
                               "first_order_change": float(torch.dot(gradient, delta))}
            total_gradient = vector_gradient(sum(losses.values()), parameters)
            predicted_total = sum(v["first_order_change"] for v in terms.values())
            torch.testing.assert_close(delta.new_tensor(predicted_total), torch.dot(total_gradient, delta), rtol=1e-8, atol=1e-10)
            for fraction in (0.001, 0.1, 1.):
                line_model = copy.deepcopy(model)
                with torch.no_grad():
                    for p, base, end in zip(line_model.parameters(), parameters, candidate.parameters()):
                        p.copy_(base+fraction*(end-base))
                values, _ = line_model.residuals(samples)
                for term, value in values.items():
                    change = float(value.detach().square().mean())-terms[term]["initial_mse"]
                    terms[term][f"actual_change_fraction_{fraction}"] = change
            row["samples"][label] = {"terms": terms, "predicted_total_change": predicted_total,
                "actual_total_change": sum(v["actual_change_fraction_1.0"] for v in terms.values()),
                "small_step_derivative_max_abs_gap": max(abs(v["actual_change_fraction_0.001"]/0.001-v["first_order_change"]) for v in terms.values())}
        if not all(torch.equal(v, model.state_dict()[k]) for k, v in frozen.items()):
            raise ValueError("Frozen model mutated")
        report["variants"][name] = row
        print(f'{name}: exact replay; fresh predicted change={row["samples"]["fresh"]["predicted_total_change"]:.6g}, disposable actual={row["samples"]["fresh"]["actual_total_change"]:.6g}', flush=True)
    for p, digest in hashes.items():
        if sha256(p) != digest:
            raise ValueError("Input artifact changed")
    report.update(inputs_unchanged=True, wall_s=time.perf_counter()-start)
    report["diagnostic_source_sha256"] = sha256(Path(__file__))
    output = root/"results"/("probe_adam_direction_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"No updated checkpoint saved. Report: {output/'report.json'}", flush=True)


if __name__ == "__main__":
    main()
