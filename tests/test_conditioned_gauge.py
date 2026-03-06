import torch

from experiments.utils import make_conditioned_gauge


def test_conditioned_gauge_has_increasing_condition_number() -> None:
    d2 = make_conditioned_gauge(32, kappa=2.0, seed=1)
    d20 = make_conditioned_gauge(32, kappa=20.0, seed=1)

    c2 = torch.linalg.cond(d2).item()
    c20 = torch.linalg.cond(d20).item()

    assert c2 > 1.2
    assert c20 > c2
