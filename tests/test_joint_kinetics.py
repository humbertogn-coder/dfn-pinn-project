"""Focused checks of the one changed residual family and joint gradients."""

import copy
import pytest
import torch
from dfn_pinn.dfn_smoke import SETTINGS
from dfn_pinn.joint_kinetics import JointKinetics, SCHEMA, restore_joint


def test_only_kinetics_changes_and_all_branches_train():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    direct = JointKinetics(SETTINGS, "joint_direct")
    inverse = JointKinetics(SETTINGS, "joint_inverse")
    inverse.load_state_dict(direct.state_dict(), strict=True)
    samples = direct.sample()
    rd, _ = direct.residuals(samples)
    ri, _ = inverse.residuals(samples)
    assert len(rd) == len(ri) == 34
    for name in rd:
        if not name.startswith("kinetics_"):
            torch.testing.assert_close(rd[name], ri[name], rtol=0, atol=0)
    common, _ = inverse.direct_residuals(samples)
    for name in rd:
        torch.testing.assert_close(rd[name], common[name], rtol=0, atol=0)
    for model, residuals in ((direct, rd), (inverse, ri)):
        sum(v.square().mean() for v in residuals.values()).backward()
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        for group in (model.ce, model.phie, model.phis, model.cs, model.reactions):
            assert all(sum(float(p.grad.square().sum()) for p in branch.parameters()) > 0 for branch in group)


@pytest.mark.parametrize("mutation", [None, "mode", "metadata"])
def test_strict_joint_restore(mutation):
    model = JointKinetics(SETTINGS, "joint_inverse")
    saved = {"schema": SCHEMA, "settings": model.settings, "mode": model.mode,
             "metadata": copy.deepcopy(model.metadata()), "model": model.state_dict()}
    if mutation == "mode":
        saved["mode"] = "joint_direct"
    elif mutation == "metadata":
        saved["metadata"]["trainable"] = "cs"
    if mutation:
        with pytest.raises(ValueError):
            restore_joint(saved)
    else:
        restored = restore_joint(saved)
        assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
