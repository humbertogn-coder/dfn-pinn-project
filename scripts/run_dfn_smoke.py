"""Three-step DFN connectivity smoke; not an accuracy or convergence study."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import torch

from dfn_pinn.dfn_smoke import DFNSmoke, SETTINGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(SETTINGS["threads"])
    torch.manual_seed(SETTINGS["seed"])
    model = DFNSmoke()
    samples = model.sample()
    groups = {f"{name}_{i}": list(branch.parameters()) for name in ("ce", "phie", "phis", "cs", "reactions")
              for i, branch in enumerate(getattr(model, name))}
    start = time.perf_counter()
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
    print("DFN wiring smoke: three regions, two particles, 12 learned branches, float64 CPU.", flush=True)
    print("Synthetic initial concentrations; three optimizer steps only. No accuracy claim.", flush=True)
    for step in range(SETTINGS["adam_steps"]):
        optimizer.zero_grad(set_to_none=True)
        residuals, diagnostics = model.residuals(samples)
        if len(residuals) != 34 or len(diagnostics) != 4:
            raise AssertionError("Unexpected DFN residual inventory")
        losses = {key: value.square().mean() for key, value in residuals.items()}
        total = sum(losses.values())
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite assembled loss")
        total.backward()
        gradients = {}
        for name, parameters in groups.items():
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError(f"Missing/nonfinite gradients: {name}")
            norm = float(torch.sqrt(sum(p.grad.square().sum() for p in parameters)))
            if norm == 0:
                raise AssertionError(f"Disconnected branch: {name}")
            gradients[name] = norm
        history.append({"step": step+1, "losses": {k: float(v.detach()) for k, v in losses.items()},
                        "gradient_norms": gradients})
        optimizer.step()
        print(f"Adam {step+1}: total_loss={float(total.detach()):.6e}; gradients finite in {len(groups)}/12 branches", flush=True)
    residuals, diagnostics = model.residuals(samples)
    maxima = {k: float(v.detach().abs().max()) for k, v in residuals.items()}
    diagnostic_maxima = {k: float(v.detach().abs().max()) for k, v in diagnostics.items()}
    if not all(torch.isfinite(v).all() for v in (*residuals.values(), *diagnostics.values())):
        raise FloatingPointError("Nonfinite post-step diagnostics")
    with torch.no_grad():
        initial_errors = {}
        for i in range(3):
            p = samples[f"region_{i}"].clone()
            p[:, 1] = 0
            initial_errors[f"ce_{i}"] = float((model.ce[i](p)-1).abs().max())
        for k, i in enumerate((0, 2)):
            p = samples[f"particle_{i}"].clone()
            p[:, 2] = 0
            initial_errors[f"cs_{i}"] = float((model.cs[k](p)-SETTINGS["initial_stoichiometries"][k]).abs().max())
        if max(initial_errors.values()) > 1e-12:
            raise AssertionError("Hard initial condition failed")
        predictions = model.snapshot(samples)
    output = (args.output_root or root/"results")/("dfn_smoke_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"model": model.state_dict(), "settings": model.settings, "samples": samples,
                "predictions": predictions, "optimizer": optimizer.state_dict()}, output/"checkpoint.pt")
    saved = torch.load(output/"checkpoint.pt", weights_only=True)
    replay = DFNSmoke(saved["settings"])
    replay.load_state_dict(saved["model"])
    with torch.no_grad():
        for name, value in replay.snapshot(saved["samples"]).items():
            torch.testing.assert_close(value, saved["predictions"][name], rtol=0, atol=0)
    sources = [Path(__file__), *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report = {"status": "WIRING_SMOKE_COMPLETE_NOT_PHYSICAL_ACCEPTANCE", "settings": model.settings,
              "history": history, "residual_maxima": maxima, "diagnostic_maxima": diagnostic_maxima,
              "initial_errors": initial_errors, "replay_exact": True, "wall_s": time.perf_counter()-start,
              "torch_version": str(torch.__version__), "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("34 residual terms evaluated; initial conditions and checkpoint replay verified.", flush=True)
    print("Total-current diagnostics (not extra BC losses):", diagnostic_maxima, flush=True)
    print(f"WIRING ONLY; physical accuracy unassessed. Output directory: {output}", flush=True)


if __name__ == "__main__":
    main()
