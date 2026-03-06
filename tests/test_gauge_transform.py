import torch

from experiments.gauge_transform import GaugedMLP
from experiments.utils import MLP, make_invertible_gauge


def test_gauged_model_logits_match_original() -> None:
    torch.manual_seed(0)
    model = MLP(d_in=8, d_hidden=12, d_out=4).eval()
    x = torch.randn(32, 8)

    d = make_invertible_gauge(12, seed=3)
    gauged = GaugedMLP(model, d).eval()

    y = model(x)
    y_g = gauged(x)

    assert torch.allclose(y, y_g, atol=5e-5)


def test_invertible_gauge_is_full_rank() -> None:
    d = make_invertible_gauge(20, seed=5)
    assert torch.linalg.matrix_rank(d).item() == 20
