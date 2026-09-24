"""Train the specified potential-driven particle; scientific audit is separate."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
from dfn_pinn.coupled_training import sample_points
from dfn_pinn.potential_particle import PotentialParticle
from dfn_pinn.potential_audit import reference_identity


def surface_points(config, smoke=False):
    points = sample_points(config, smoke)
    count = len(points["interior"])//4
    # Preserve all times and 3/4 of original radii; map 1/4 into the outer shell.
    points["interior"][-count:, 0] = .95+.05*points["interior"][-count:, 0]
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root/"configs/potential_particle_v1.json"
    c = json.loads(config_path.read_text())
    reference_hashes = reference_identity(root/c["reference_run"], c)
    torch.manual_seed(c["seed"])
    torch.set_num_threads(1)
    model = PotentialParticle(c)
    points = surface_points(c, args.smoke)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_root or root/"results")/f"potential_particle_surface_{'smoke' if args.smoke else 'full'}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    steps, iterations, evaluations = (3, 2, 4) if args.smoke else (c["adam_steps"], c["lbfgs_max_iter"], c["lbfgs_max_eval"])
    history = []
    start = time.perf_counter()
    def loss():
        terms = model.losses(points)
        total = sum(c["loss_weights"][key]*value for key, value in terms.items())
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite training loss")
        total.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Missing or nonfinite parameter gradient")
        history.append({"stage": stage, **{k: float(v.detach()) for k, v in terms.items()}})
        return total
    print(f"Potential-driven particle: {'SMOKE ONLY' if args.smoke else 'FULL BUDGET'}, float64 CPU", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=c["adam_learning_rate"])
    stage = "adam"
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = loss()
        optimizer.step()
        if args.smoke or (step+1) % 100 == 0:
            print(f"Adam {step+1}: loss={total.item():.6e}", flush=True)
    stage = "lbfgs"
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=iterations, max_eval=evaluations, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(set_to_none=True)
        return loss()
    optimizer.step(closure)
    final_losses = {k: float(v.detach()) for k, v in model.losses(points).items()}
    torch.save({"model": model.state_dict(), "config": c, "points": points,
                "optimizer": optimizer.state_dict()}, output/"checkpoint.pt")
    loaded = torch.load(output/"checkpoint.pt", weights_only=True)
    replay = PotentialParticle(loaded["config"])
    replay.load_state_dict(loaded["model"])
    with torch.no_grad():
        expected = model.concentration(points["interior"])
        actual = replay.concentration(points["interior"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(replay.current.integral.current_model(points["boundary"]),
                                   model.current.integral.current_model(points["boundary"]), rtol=0, atol=0)
    sources = [config_path, Path(__file__), *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report = {"status": "smoke_complete_not_scientific_acceptance" if args.smoke else "trained_pending_independent_audit",
              "sampling": {"kind": "outer_shell_mixture", "fraction": 0.25, "rho_min": 0.95},
              "reference_hashes": reference_hashes, "config": c, "actual_adam_steps": steps, "lbfgs_max_iter": iterations,
              "lbfgs_max_eval": evaluations, "threads": 1, "torch_version": str(torch.__version__),
              "wall_s": time.perf_counter()-start, "final_training_losses": final_losses,
              "checkpoint_replay_exact": True, "history": history,
              "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Checkpoint replay verified. Independent acceptance audit NOT run.", flush=True)
    print(f"Output directory: {output}", flush=True)


if __name__ == "__main__":
    main()
