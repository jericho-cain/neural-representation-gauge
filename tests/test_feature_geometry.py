import numpy as np

from experiments.feature_geometry import principal_alignment_cosines, topk_pca_basis
from experiments.utils import make_orthogonal_gauge


def test_pca_subspace_alignment_under_orthogonal_gauge() -> None:
    rng = np.random.default_rng(7)
    h = rng.normal(size=(600, 24))

    d = make_orthogonal_gauge(24, seed=11).detach().cpu().numpy()
    h_g = h @ d.T

    k = 8
    basis_before, var_before = topk_pca_basis(h, k)
    basis_after, var_after = topk_pca_basis(h_g, k)

    expected_basis_after = d @ basis_before
    cosines = principal_alignment_cosines(expected_basis_after, basis_after)

    assert np.min(cosines) > 0.999
    assert np.max(np.abs(var_before - var_after)) < 1e-7
