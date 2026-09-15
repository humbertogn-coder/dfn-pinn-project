import numpy as np
import pytest

from dfn_pinn.reference import ReferenceBundle, normalize


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        normalize([1], scale)


def test_fixed_scale_roundtrip():
    original = np.array([0.0, 1000.0, 2000.0])
    np.testing.assert_array_equal(normalize(original, 1000) * 1000, original)


def test_segments_have_unique_endpoints():
    bundle = ReferenceBundle.__new__(ReferenceBundle)
    bundle.times = {"fine_startup": np.array([0, .1, .2]),
                    "startup": np.array([0, .1, .2, 5, 10]),
                    "full": np.array([0, 10, 20, 30])}
    selected = np.concatenate([bundle.times[s][i] for s, i in bundle.selected_indices()])
    np.testing.assert_array_equal(selected, [0, .1, .2, 5, 10, 20, 30])


def test_missing_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReferenceBundle(tmp_path)
