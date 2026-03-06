import numpy as np

from experiments.utils import whiten


def test_whitening_maps_covariance_to_identity() -> None:
    rng = np.random.default_rng(123)
    h = rng.normal(size=(500, 16))
    scales = np.linspace(0.4, 3.2, 16)
    h = h * scales[None, :]

    _, _, cov_white = whiten(h)
    eye = np.eye(cov_white.shape[0])

    assert np.allclose(cov_white, eye, atol=1e-4)
