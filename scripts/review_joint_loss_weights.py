"""Review recorded loss magnitudes without loading or updating a model."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "results/joint_kinetics_feasibility_20260929T200959116945Z"
    report_path = source / "report.json"
    original_hash = digest(report_path)
    report = json.loads(report_path.read_text())
    if report["status"] != "JOINT_PAIR_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed joint comparison required")
    for name, expected in report["source_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Changed historical source: {name}")
    rows = {}
    for arm, data in report["arms"].items():
        if len(data["history"]) != 200:
            raise ValueError("Expected 200 recorded Adam steps")
        final = data["history"][-1]
        losses = final["losses"]
        if len(losses) != 34 or any(not math.isfinite(v) or v < 0 for v in losses.values()):
            raise ValueError("Invalid loss inventory")
        total = sum(losses.values())
        if not math.isclose(total, final["total_loss"], rel_tol=1e-12):
            raise ValueError("Recorded total does not match component sum")
        groups = {
            "electrolyte_charge": sum(v for k, v in losses.items() if k.startswith("charge_e_")),
            "kinetics": sum(v for k, v in losses.items() if k.startswith("kinetics_")),
            "particle_inventory": sum(v for k, v in losses.items() if k.startswith("inventory_")),
        }
        groups["other"] = total - sum(groups.values())
        rows[arm] = {
            "state": "pre-update step 200 (after 199 accepted updates), training points",
            "total": total,
            "groups": {k: {"mse_sum": v, "percent_of_total": 100*v/total} for k, v in groups.items()},
            "terms": losses,
            "interpretation": "Loss magnitudes only, not gradient dominance or physical acceptance.",
        }
        print(arm)
        for key, value in rows[arm]["groups"].items():
            print(f"  {key}: MSE={value['mse_sum']:.6e}, share={value['percent_of_total']:.6g}%")
    if digest(report_path) != original_hash:
        raise ValueError("Input changed during review")
    out = root / "results" / ("joint_loss_review_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(exist_ok=False)
    result = {"status": "RECORDED_LOSS_REVIEW_ONLY", "source_report": str(report_path),
              "source_report_sha256": original_hash, "review_script_sha256": digest(Path(__file__)),
              "arms": rows, "training_run": False,
              "warning": "Direct and inverse kinetic objectives differ; totals cannot rank physical quality."}
    (out / "review.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"No training or model evaluation. Report: {out / 'review.json'}")


if __name__ == "__main__":
    main()
