import importlib.util
from pathlib import Path
import sys

import numpy as np

scripts = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("flux_diagnosis", scripts/"diagnose_flux_residual.py")
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


def test_physical_measure_and_constant_rms():
    r, wr, t, wt = diagnosis.physical_quadrature(8, 4)
    np.testing.assert_allclose(wt.sum(), diagnosis.END_TIME-diagnosis.MIN_TIME, rtol=1e-14)
    np.testing.assert_allclose((3*r*r*wr).sum(), 1., rtol=1e-14)
    assert abs(diagnosis.volume_time_rms(np.full((len(r), len(t)), 2.), r, wr, wt)-2) < 1e-14
    # Integral of rho^2 under normalized spherical volume is 3/5.
    actual = diagnosis.volume_time_rms(np.broadcast_to(r[:, None], (len(r), len(t))), r, wr, wt)
    np.testing.assert_allclose(actual, np.sqrt(3/5), rtol=1e-14)


def test_batched_residual_terms():
    model = lambda p: p[:, :1]**2+6*p[:, 2:3]
    r = np.array([0., .2, .9, 1.])
    t = np.full(4, .01)
    dt, diffusion, residual = diagnosis.residual_components(model, r, t, batch_size=2)
    np.testing.assert_allclose(dt, 6)
    np.testing.assert_allclose(diffusion, 6)
    np.testing.assert_allclose(residual, 0, atol=1e-14)
