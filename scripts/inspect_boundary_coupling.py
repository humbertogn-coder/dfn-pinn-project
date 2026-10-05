"""Read-only SI current/source/particle-flux profiles for the frozen boundary probe."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dfn_pinn.charge_conservation import solid_charge_terms, electrolyte_charge_terms
from dfn_pinn.constitutive import FARADAY_CONSTANT, ocp, exchange_current
from dfn_pinn.kinetics import direct_bv
from dfn_pinn.spherical_diffusion import _gradient
from prepare_dfn_baseline import sha256
from probe_boundary_200 import restore_probe
from probe_potential_200 import assert_equal_tree
from probe_solid_potential_scaling import measure


def main():
    root = Path(__file__).resolve().parents[1]
    run = root/"results/boundary_200_20260928T201841393914Z"
    prior = json.loads((run/"report.json").read_text())
    if prior["status"] != "BOUNDARY_200_COMPLETE_NOT_VALIDATED":
        raise ValueError("Expected completed boundary probe")
    checkpoint = prior["milestones"]["200"]
    hashes = {run/"report.json": sha256(run/"report.json"),
              run/checkpoint["checkpoint"]: checkpoint["sha256"],
              run/"paired_inputs.pt": prior["paired_inputs_sha256"]}
    hashes.update({root/k: v for k, v in prior["source_hashes"].items()})
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError(f"Frozen input mismatch: {path}")
    torch.set_num_threads(1)
    saved = torch.load(run/checkpoint["checkpoint"], weights_only=True)
    model = restore_probe(saved)
    if saved["step"] != 200 or saved["settings"] != prior["settings"]:
        raise ValueError("Checkpoint identity mismatch")
    bundle = torch.load(run/"paired_inputs.pt", weights_only=True)
    for label, key in (("training", "training_samples"), ("fresh", "fresh_samples")):
        assert_equal_tree(measure(model, bundle[key]), checkpoint["metrics"][label])
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    c, scales = model.settings, model.charge_scales
    times = [1e-6, 1e-4, .001, .01, .1, .5, 1.]
    rows, summaries = [], []
    for time_s in times:
        for region, (left, right) in enumerate(model.bounds):
            x = torch.linspace(left, right, 81, dtype=torch.float64)
            p = torch.stack((x, torch.full_like(x, time_s/c["time_reference_s"])), dim=1).requires_grad_()
            k = 0 if region == 0 else 1
            j = model.reactions[k].integral.current_model(p) if region != 1 else p[:, :1]*0
            aj = model.active_area[k]*j if region != 1 else j
            electrolyte = electrolyte_charge_terms(model.ce[region], model.phie[region], p, aj, scales,
                                                    porosity=c["porosities"][region])
            fields = {"i_e_A_m2_geom": electrolyte["current_A_m2"],
                      "source_A_m3": aj, "electrolyte_charge_residual_A_m3": electrolyte["balance_A_m3"]}
            if region != 1:
                solid = solid_charge_terms(model.phis[k], p, aj, scales,
                                          conductivity_S_m=c["solid_conductivities_S_m"][k])
                surface = torch.cat((torch.ones_like(p[:, :1]), p), dim=1)
                theta = model.cs[k](surface)
                particle_scale = model.reactions[k].scales
                flux = -particle_scale.diffusivity_m2_s*particle_scale.concentration_mol_m3/particle_scale.radius_m*_gradient(theta, surface)[:, :1]
                particle_current = FARADAY_CONSTANT*flux
                normalized = model.reactions[k].flux_residual(model.cs[k], surface, model.jref[k])
                torch.testing.assert_close((particle_current-j)/model.jref[k], normalized, rtol=1e-10, atol=1e-12)
                temp = torch.full_like(j, c["temperature_K"])
                electrode = ("n", "p")[k]
                eta = scales.potential_V*(model.phis[k](p)-model.phie[region](p))-ocp(theta, electrode)
                j0 = exchange_current(scales.concentration_mol_m3*model.ce[region](p),
                                      theta*c["cmax_mol_m3"][k], temp, electrode, mode=c["kinetics_mode"])
                bv = direct_bv(eta, j0, temp)
                fields.update(i_s_A_m2_geom=solid["current_A_m2"],
                    solid_charge_residual_A_m3=solid["balance_A_m3"],
                    j_learned_A_m2_active=j, j_particle_A_m2_active=particle_current,
                    j_bv_A_m2_active=bv, outward_flux_mol_m2_s=flux,
                    particle_flux_mismatch_A_m2_active=particle_current-j)
            else:
                fields["i_s_A_m2_geom"] = p[:, :1]*0
            fields["i_total_A_m2_geom"] = fields["i_s_A_m2_geom"]+fields["i_e_A_m2_geom"]
            arrays = {key: value.detach().numpy().ravel() for key, value in fields.items()}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Nonfinite profile")
            summaries.append({"time_s": time_s, "region": region,
                "ranges": {key: [float(a.min()), float(a.max())] for key, a in arrays.items()},
                "max_total_current_error_A_m2_geom": float(np.max(np.abs(arrays["i_total_A_m2_geom"]-scales.current_A_m2)))})
            for i, coordinate in enumerate(x.tolist()):
                rows.append({"time_s": time_s, "region": region, "x_m": coordinate*scales.length_m,
                             **{key: float(a[i]) for key, a in arrays.items()}})
    for key, value in frozen.items():
        assert_equal_tree(value, model.state_dict()[key])
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("Frozen input changed")
    output = root/"results"/("boundary_coupling_profiles_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output/"profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for region in range(3):
        selected = [r for r in rows if r["time_s"] == 1. and r["region"] == region]
        x = [r["x_m"]*1e6 for r in selected]
        for key, label, color in (("i_s_A_m2_geom", "Solid", "#2563eb"),
                                  ("i_e_A_m2_geom", "Electrolyte", "#db2777"),
                                  ("i_total_A_m2_geom", "Total", "#15803d")):
            axes[0].plot(x, [r[key] for r in selected], color=color, label=label if region == 0 else None)
        if region != 1:
            ax = axes[1 if region == 0 else 2]
            for key, label in (("j_learned_A_m2_active", "Learned j"),
                               ("j_particle_A_m2_active", "F times particle flux"),
                               ("j_bv_A_m2_active", "Butler-Volmer")):
                ax.plot(x, [r[key] for r in selected], label=label)
    axes[0].axhline(scales.current_A_m2, color="black", linestyle="--", label="Applied")
    for ax, title in zip(axes, ("Through-cell current", "Negative particle interface", "Positive particle interface")):
        ax.set_title(title+" at 1 s")
        ax.set_xlabel("x [micrometers]")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Current density [A/m2 geometric area]")
    for ax in axes[1:]:
        ax.set_ylabel("Current density [A/m2 active area]")
    fig.savefig(output/"profiles_at_1s.png", dpi=150)
    plt.close(fig)
    sources = [Path(__file__), root/"scripts/probe_boundary_200.py", root/"scripts/probe_potential_200.py",
               root/"scripts/probe_solid_potential_scaling.py", root/"scripts/prepare_dfn_baseline.py",
               *sorted((root/"src/dfn_pinn").glob("*.py"))]
    report = {"status": "FROZEN_PROFILES_ONLY_NOT_VALIDATED", "input_run": str(run),
        "times_s": times, "points_per_region": 81, "applied_current_density_A_m2_geom": scales.current_A_m2,
        "summaries": summaries, "artifacts_unchanged": True, "metric_replay_verified": True,
        "flux_unit_conversion_verified": True, "input_hashes": {str(p): v for p, v in hashes.items()},
        "source_hashes": {str(p.relative_to(root)): sha256(p) for p in sources},
        "scope": "One-sided endpoint traces retained separately per region. No interpolation across interfaces, training, quadrature certification or reference comparison."}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Frozen profiles verified. No training. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
