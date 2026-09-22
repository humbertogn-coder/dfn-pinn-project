import importlib.util
from pathlib import Path
import sys

import numpy as np

scripts = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("surface_peak", scripts/"check_flux_surface_peak.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_nested_grid_domain():
    r, t = audit.refined_grid()
    assert r[0] == .9 and r[-1] == 1 and t[0] == 1e-5 and t[-1] == 1e-3
    assert np.all(np.diff(r)>0) and np.all(np.diff(t)>0)
    assert len(r[::4]) == len(t[::4]) == 81


def test_peak_preserves_sign_and_component_axes():
    dt = np.array([[1., -7.], [2., 3.]])
    diffusion = np.ones((2, 2))
    peak = audit.peak_record(dt-diffusion, dt, diffusion, [.9, 1.], [1e-5, 1e-3])
    assert peak == {"max_abs": 8., "rho": .9, "tau": 1e-3,
                    "signed_residual": -8., "time_derivative": -7., "diffusion_term": 1.}
