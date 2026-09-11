from dfn_pinn import project_summary


def test_project_summary_contains_first_goal() -> None:
    summary = project_summary()
    assert summary["name"] == "dfn-pinn"
    assert "PyBaMM" in summary["first_goal"]
