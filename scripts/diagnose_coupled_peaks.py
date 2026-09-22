"""Locate current and kinetics error peaks on a frozen checkpoint; no training."""

import argparse
import hashlib
import json
from pathlib import Path
import torch

from dfn_pinn.coupled_training import CoupledParticle
from dfn_pinn.coupled_audit import boundary_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = json.loads((args.run/"report.json").read_text())
    for name, expected in report["source_hashes"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source mismatch: {name}")
    saved = torch.load(args.run/"checkpoint.pt", weights_only=True, map_location="cpu")
    if saved["config"] != report["config"]:
        raise ValueError("Configuration mismatch")
    torch.set_num_threads(1)
    model = CoupledParticle(saved["config"])
    model.load_state_dict(saved["model"])
    model.eval()
    c = saved["config"]
    end = c["duration_s"]
    for count in (1001, 2001):
        seconds = torch.unique(torch.cat((torch.linspace(0, end, count, dtype=torch.float64)[1:],
                                          torch.logspace(-8, -1, 281, dtype=torch.float64))))
        values = boundary_values(model, seconds/c["time_reference_s"])
        for name, error in (("current", values[:, 0]-c["reference_current_A_m2"]),
                            ("kinetics", values[:, 2])):
            index = int(error.abs().argmax())
            print(f"grid={count} {name}: signed_error={error[index]:.9e} A/m2, time={seconds[index]:.9e} s")
            print(f"  current={values[index, 0]:.9e}, BV_current={values[index, 0]-values[index, 2]:.9e}, outward_flux={values[index, 3]:.9e} A/m2")
        for a, b in ((0., .1), (.1, 1.), (1., end)):
            selected = (seconds > a) & (seconds <= b)
            print(f"  ({a:g},{b:g}]s: max_current_error={float((values[selected, 0]-c['reference_current_A_m2']).abs().max()):.9e}, max_BV_residual={float(values[selected, 2].abs().max()):.9e} A/m2")
    print("Sampled peaks only; times below 1e-8 s remain unchecked. No training was run.")


if __name__ == "__main__":
    main()
