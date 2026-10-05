"""Focused objective, gradient and checkpoint identity checks."""

import copy
import pytest
import torch
from dfn_pinn.dfn_smoke import SETTINGS
from dfn_pinn.dfn_run_contract import adam_step
from dfn_pinn.joint_kinetics import JointKinetics, SCHEMA as JOINT
from dfn_pinn.inventory_weighting import weighted_step, restore_weighted, SCHEMA


def test_unit_weight_exact_step():
    torch.set_num_threads(1)
    a = JointKinetics(SETTINGS, "joint_direct")
    b = copy.deepcopy(a)
    samples = a.sample()
    expected = adam_step(a, torch.optim.Adam(a.parameters(), lr=.001), samples)
    actual = weighted_step(b, torch.optim.Adam(b.parameters(), lr=.001), samples, 1)
    raw = actual.pop("raw_losses")
    assert actual == expected
    assert raw == expected["losses"]
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_weighted_gradient_matches_explicit_objective():
    a = JointKinetics(SETTINGS, "joint_direct")
    b = copy.deepcopy(a)
    samples = a.sample()
    r, _ = a.residuals(samples)
    sum(v.square().mean() * (1e8 if k.startswith("inventory_") else 1)
        for k, v in r.items()).backward()
    row = weighted_step(b, torch.optim.Adam(b.parameters(), lr=.001), samples, 1e8)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-11, atol=1e-10)
    for k, raw in row["raw_losses"].items():
        assert row["losses"][k] == pytest.approx(raw * (1e8 if k.startswith("inventory_") else 1))


def test_weight_metadata_and_round_trip():
    a = JointKinetics(SETTINGS, "joint_direct")
    saved = {"schema": SCHEMA, "arm": "weighted", "inventory_weight": 1e8,
             "joint": {"schema": JOINT, "settings": a.settings, "mode": a.mode,
                       "metadata": a.metadata(), "model": a.state_dict()}}
    b = restore_weighted(saved)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    saved["inventory_weight"] = 1
    with pytest.raises(ValueError):
        restore_weighted(saved)
