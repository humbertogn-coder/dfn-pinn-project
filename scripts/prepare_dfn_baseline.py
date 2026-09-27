"""Freeze the first DFN experiment specification and samples; no training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from dfn_pinn.dfn_smoke import SETTINGS


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference(root, config):
    directory = root / config["reference"]
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report["settings"] != SETTINGS:
        raise ValueError("Assembly settings differ from the pinned physical reference")
    expected = config["reference_sha256"]
    if report["reference_sha256"] != expected or sha256(directory / "reference.h5") != expected:
        raise ValueError("Reference data hash mismatch")
    for name, digest in report["source_hashes"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Reference source mismatch: {name}")
    return report


def make_samples(config, settings):
    """Physical-time mixture converted once to the model's t/t_ref coordinates."""
    train = config["training"]
    n, nb = train["interior_points_per_region"], train["boundary_times"]
    start, end = train["minimum_positive_time_s"], settings["duration_s"]
    if any(not isinstance(v, int) or v < 2 or v % 2 for v in (n, nb)):
        raise ValueError("Point counts must be positive even integers >= 2")
    if not 0 < start < end:
        raise ValueError("Positive time window must lie within the experiment")
    generator = np.random.default_rng(config["sampling_seed"])
    def times(count):
        half = count // 2
        t = np.r_[generator.uniform(start, end, half),
                  np.exp(generator.uniform(np.log(start), np.log(end), half))]
        generator.shuffle(t)
        return t / settings["time_reference_s"]
    edges = np.r_[0., np.cumsum(settings["lengths_m"])] / sum(settings["lengths_m"])
    result = {}
    for index in range(3):
        x = generator.uniform(edges[index], edges[index+1], n)
        xt = np.column_stack((x, times(n)))
        result[f"region_{index}"] = xt
        if index != 1:
            result[f"particle_{index}"] = np.column_stack((generator.uniform(0., 1., n), xt))
    boundary = times(nb)
    boundary[0], boundary[1] = start / settings["time_reference_s"], end / settings["time_reference_s"]
    result["times"] = np.sort(boundary)[:, None]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dfn_baseline_v1.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / args.config
    config = json.loads(path.read_text(encoding="utf-8"))
    reference = verify_reference(root, config)
    samples = make_samples(config, SETTINGS)
    output = root / "results" / ("dfn_baseline_preparation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    np.savez(output / "samples.npz", **samples)
    with np.load(output / "samples.npz", allow_pickle=False) as saved:
        for name, value in samples.items():
            np.testing.assert_array_equal(saved[name], value)
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")
    manifest = {"status": "SPECIFICATION_ONLY_NOT_TRAINING_READY", "config_sha256": sha256(path),
                "samples_sha256": sha256(output / "samples.npz"), "physical_settings": SETTINGS,
                "reference_sha256": reference["reference_sha256"],
                "source_hashes": {str(p.relative_to(root)): sha256(p) for p in [Path(__file__), *sorted((root / "src/dfn_pinn").glob("*.py"))]},
                "sample_shapes": {k: list(v.shape) for k, v in samples.items()},
                "launch_blockers": ["Independent full-DFN auditor not implemented", "Training cost and replay dry run pending"],
                "scope": "Reference identity and sample preparation only. No training, new simulation or physical acceptance."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print("Reference identity and sample round trip verified.")
    for key, value in samples.items():
        print(f"{key}: {value.shape}")
    print("SPECIFICATION_ONLY_NOT_TRAINING_READY; no training or simulation.")
    print(f"Prepared experiment: {output}")


if __name__ == "__main__":
    main()
