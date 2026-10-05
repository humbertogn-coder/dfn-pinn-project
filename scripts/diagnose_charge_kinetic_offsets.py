"""Frozen potential level/slope diagnosis; disposable shifts, no optimizer."""

import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from dfn_pinn.inventory_weighting import restore_weighted
from dfn_pinn.charge_conservation import electrolyte_charge_terms, solid_charge_terms
from dfn_pinn.constitutive import ocp, exchange_current, electrolyte_conductivity
from dfn_pinn.kinetics import inverse_bv, direct_bv
from dfn_pinn.spherical_diffusion import _gradient
from prepare_dfn_baseline import sha256
from probe_potential_200 import assert_equal_tree
from train_joint_kinetics_feasibility import common_measure


class Shift(nn.Module):
    def __init__(self, field, offset):
        super().__init__()
        self.field, self.offset = field, offset

    def forward(self, points):
        return self.field(points) + self.offset


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "results/inventory_weight_pair_20260929T222006194779Z"
    report = json.loads((source / "report.json").read_text())
    if report["status"] != "PAIR_COMPLETE_NOT_VALIDATED":
        raise ValueError("Completed fixed-weight experiment required")
    milestone = report["arms"]["control"]["milestones"]["200"]
    hashes = {source / "report.json": sha256(source / "report.json"),
              source / milestone["checkpoint"]: milestone["sha256"],
              source / "samples.pt": report["samples_sha256"]}
    hashes.update({root / k: v for k, v in report["source_hashes"].items()})
    for p, h in hashes.items():
        if sha256(p) != h:
            raise ValueError(f"Changed frozen source: {p}")
    torch.set_num_threads(1)
    model = restore_weighted(torch.load(source / milestone["checkpoint"], weights_only=True))
    frozen = copy.deepcopy(model.state_dict())
    samples = torch.load(source / "samples.pt", weights_only=True)
    before = {label: common_measure(model, data) for label, data in samples.items()}
    assert_equal_tree(before, milestone["common_metrics"])
    c, s = model.settings, model.charge_scales
    rows, gaps = [], {}
    for time_s in (1e-6, 1e-4, .001, .01, .1, .5, 1.):
        for i, (left, right) in enumerate(model.bounds):
            x = torch.linspace(left, right, 81, dtype=torch.float64)
            p = torch.stack((x, torch.full_like(x, time_s/c["time_reference_s"])), dim=1).requires_grad_()
            k = 0 if i == 0 else 1
            j = model.reactions[k].integral.current_model(p) if i != 1 else p[:, :1]*0
            aj = model.active_area[k]*j if i != 1 else j
            electrolyte = electrolyte_charge_terms(model.ce[i], model.phie[i], p, aj, s, porosity=c["porosities"][i])
            solid_current = solid_charge_terms(model.phis[k], p, aj, s,
                conductivity_S_m=c["solid_conductivities_S_m"][k])["current_A_m2"] if i != 1 else j
            ce = model.ce[i](p)
            conductivity = c["porosities"][i]**1.5 * electrolyte_conductivity(s.concentration_mol_m3*ce)
            actual_slope = s.potential_V/s.length_m * _gradient(model.phie[i](p), p)[:, :1]
            # Required slope for total-current closure with frozen solid current/ce.
            required_slope = actual_slope + (electrolyte["current_A_m2"] - (s.current_A_m2-solid_current))/conductivity
            values = {"phi_e_V": s.potential_V*model.phie[i](p), "i_e_A_m2": electrolyte["current_A_m2"],
                      "i_e_required_for_total_current_A_m2": s.current_A_m2-solid_current,
                      "actual_phi_e_slope_V_m": actual_slope,
                      "required_phi_e_slope_V_m": required_slope}
            if i != 1:
                theta = model.cs[k](torch.cat((torch.ones_like(p[:, :1]), p), dim=1))
                temp = torch.full_like(j, c["temperature_K"])
                electrode = ("n", "p")[k]
                j0 = exchange_current(s.concentration_mol_m3*ce, theta*c["cmax_mol_m3"][k],
                                      temp, electrode, mode=c["kinetics_mode"])
                eta = s.potential_V*(model.phis[k](p)-model.phie[i](p))-ocp(theta, electrode)
                required_eta = inverse_bv(j, j0, temp)
                torch.testing.assert_close(direct_bv(required_eta, j0, temp), j, rtol=1e-12, atol=1e-12)
                values.update(eta_V=eta, eta_required_V=required_eta, eta_gap_V=eta-required_eta,
                              j_A_m2=j, j_bv_A_m2=direct_bv(eta, j0, temp))
                if time_s == 1.:
                    gaps[k] = float((eta-required_eta).detach().mean())
            arrays = {key: value.detach().flatten().tolist() for key, value in values.items()}
            if any(not torch.isfinite(value).all() for value in values.values()):
                raise ValueError("Nonfinite diagnostic")
            for n, coordinate in enumerate(x.tolist()):
                rows.append({"time_s": time_s, "region": i, "x_m": coordinate*s.length_m,
                             **{key: value[n] for key, value in arrays.items()}})
    # Match mean eta gaps at 1 s, without changing any spatial derivative.
    delta_e = gaps[0]
    delta_s_p = delta_e-gaps[1]
    shifted = copy.deepcopy(model)
    shifted.phie = nn.ModuleList([Shift(f, delta_e/s.potential_V) for f in shifted.phie])
    shifted.phis[1] = Shift(shifted.phis[1], delta_s_p/s.potential_V)
    after = {label: common_measure(shifted, data) for label, data in samples.items()}
    max_nonkinetic_gap = 0.
    for label, data in samples.items():
        rb, db = model.direct_residuals(data)
        ra, da = shifted.direct_residuals(data)
        for key in rb:
            if key.startswith("kinetics_"):
                continue
            torch.testing.assert_close(ra[key], rb[key], rtol=1e-10, atol=1e-10)
            max_nonkinetic_gap = max(max_nonkinetic_gap, float((ra[key]-rb[key]).detach().abs().max()))
        for key in db:
            torch.testing.assert_close(da[key], db[key], rtol=1e-10, atol=1e-10)
    assert_equal_tree(model.state_dict(), frozen)
    for p, h in hashes.items():
        if sha256(p) != h:
            raise ValueError("Input changed during diagnosis")
    out = root / "results" / ("charge_kinetic_offsets_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(exist_ok=False)
    with (out / "profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for i in range(3):
        selected = [r for r in rows if r["time_s"] == 1 and r["region"] == i]
        keys = [k for k in selected[0] if k not in ("time_s", "region", "x_m")]
        summary.append({"region": i, "ranges_at_1s": {k: [min(r[k] for r in selected), max(r[k] for r in selected)] for k in keys}})
    result = {"status": "FROZEN_DIAGNOSTIC_ONLY_NOT_VALIDATED", "source": str(source),
              "arm": "control", "input_hashes": {str(p): h for p, h in hashes.items()},
              "script_sha256": sha256(Path(__file__)), "before": before, "disposable_shift": after,
              "delta_e_V": delta_e, "delta_s_p_V": delta_s_p, "summaries": summary,
              "max_nonkinetic_residual_change": max_nonkinetic_gap,
              "source_replay_verified": True, "source_unchanged": True, "bv_roundtrip_verified": True,
              "scope": "Offsets chosen from mean eta gaps on 81 uniform x points at 1 s. No optimization or accepted checkpoint. Required slopes impose total-current closure only, not local charge with the frozen j. No quadrature or full validation."}
    (out / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Disposable shifts: electrolyte={delta_e:.6g} V; positive solid={delta_s_p:.6g} V")
    for label, metrics in (("before", before), ("shifted", after)):
        m = metrics["weight_check"]["residuals"]
        print(f'{label}: kinetics RMS n={m["kinetics_0"]["rms"]:.6g}, p={m["kinetics_2"]["rms"]:.6g}')
    print(f"Nonkinetic invariance verified; max change={max_nonkinetic_gap:.3e}")
    print(f"No training. Report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
