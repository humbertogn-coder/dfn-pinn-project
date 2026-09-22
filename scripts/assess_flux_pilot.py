"""Apply explicitly post-hoc working targets to a saved paired particle study."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


SEEDS = (0, 1, 2, 3, 42)
SAMPLERS = ("legacy", "full_radius")
CRITERIA = {
    "initial_max_error": 1e-12,
    "positive_time_max_error": 1e-3,
    "max_surface_flux_residual": 1e-2,
    "max_volume_mean_balance_error": 0.001 * (3 * 0.02),
}


def evaluate(study):
    rows = study["runs"]
    indexed = {}
    for row in rows:
        identity = (row["seed"], row["sampling"])
        if identity in indexed or identity[0] not in SEEDS or identity[1] not in SAMPLERS:
            raise ValueError("Duplicate or unexpected run identity")
        indexed[identity] = row
    results = {}
    for sampling in SAMPLERS:
        checks = []
        for seed in SEEDS:
            row = indexed.get((seed, sampling))
            if row is None or row.get("status") != "completed":
                checks.append({"seed": seed, "status": "INCOMPLETE", "checks": {}})
                continue
            metrics = {}
            for key, limit in CRITERIA.items():
                value = row.get(key)
                valid = (isinstance(value, (float, int)) and not isinstance(value, bool)
                         and math.isfinite(value) and value >= 0)
                metrics[key] = {"value": value if valid else None, "limit": limit,
                                "ratio": value/limit if valid else None,
                                "status": "INVALID" if not valid else "PASS" if value <= limit else "FAIL"}
            statuses = [check["status"] for check in metrics.values()]
            status = "INVALID" if "INVALID" in statuses else "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
            checks.append({"seed": seed, "status": status, "checks": metrics})
        complete = all(check["status"] not in ("INVALID", "INCOMPLETE") for check in checks)
        aggregates = {}
        for key, limit in CRITERIA.items():
            values = [check["checks"][key]["value"] for check in checks
                      if key in check["checks"] and check["checks"][key]["value"] is not None]
            aggregates[key] = {"available": len(values), "passed": sum(v <= limit for v in values),
                               "required": len(SEEDS), "worst_available": max(values) if values else None,
                               "limit": limit}
        results[sampling] = {"status": "INCOMPLETE" if not complete else
                             "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
                             "seeds": checks, "metrics": aggregates}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, help="Study directory; defaults to latest, including incomplete studies")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidates = sorted((root/"results").glob("flux_particle_seeds_*/summary.json"))
    if args.study is None and not candidates:
        parser.error("No paired study found")
    source = args.study/"summary.json" if args.study else candidates[-1]
    study = json.loads(source.read_text(encoding="utf-8"))
    results = evaluate(study)
    output = root/"results"/("flux_pilot_assessment_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    payload = {"criteria_version": "exploratory_v1", "criteria": CRITERIA,
               "source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "results": results,
               "scope": "Post-hoc proposed engineering targets, not application-derived DFN tolerances. "
                        "All five seeds required. Saved summary metrics only; no checkpoint revalidation or training. "
                        "Concentration scale=1, imposed flux=1, extracted volume-mean concentration=3*0.02=0.06. "
                        "PDE RMS is diagnostic, not gated. Sampled evidence only; no continuous-time certification. "
                        "New independent seeds are required after freezing a method and acceptance protocol."}
    (output/"report.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Study: {source}")
    for sampling, result in results.items():
        print(f"{sampling}: {result['status']} (exploratory targets)")
        for key, metric in result["metrics"].items():
            print(f"  {key}: {metric['passed']}/5 pass, worst={metric['worst_available']}, limit={metric['limit']}")
    print("No training was run. A target failure is a scientific finding, not a script error.")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
