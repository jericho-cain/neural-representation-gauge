import numpy as np

from experiments.utils import mean_jaccard_curve


def test_jaccard_curve_identity_is_one() -> None:
    rng = np.random.default_rng(0)
    h = rng.normal(size=(120, 16))
    curve = mean_jaccard_curve(h, h, [1, 5, 10])

    assert curve[1] == 1.0
    assert curve[5] == 1.0
    assert curve[10] == 1.0


def test_jaccard_curve_changes_under_anisotropic_scaling() -> None:
    rng = np.random.default_rng(1)
    h = rng.normal(size=(150, 20))
    scales = np.logspace(-0.8, 0.8, 20)
    h_g = h * scales[None, :]

    curve = mean_jaccard_curve(h, h_g, [10])
    assert curve[10] < 0.99
