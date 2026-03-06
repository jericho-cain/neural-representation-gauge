from pathlib import Path

import numpy as np
import torch

from experiments.plots import save_cosine_distortion_figure
from experiments.utils import (
    accuracy,
    make_invertible_gauge,
    pairwise_cosine,
    representations,
    train_or_load_model,
)


class GaugedMLP(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, d_matrix: torch.Tensor) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(base_model.fc1.in_features, base_model.fc1.out_features)
        self.fc2 = torch.nn.Linear(base_model.fc2.in_features, base_model.fc2.out_features)
        self.relu = torch.nn.ReLU()
        self.register_buffer("d_matrix", d_matrix)

        self.fc1.load_state_dict(base_model.fc1.state_dict())

        w = base_model.fc2.weight.detach().clone()
        b = base_model.fc2.bias.detach().clone()
        d_inv = torch.linalg.inv(d_matrix)
        w_prime = w @ d_inv

        with torch.no_grad():
            self.fc2.weight.copy_(w_prime)
            self.fc2.bias.copy_(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        h_g = h @ self.d_matrix.T
        return self.fc2(h_g)


def main() -> None:
    root = Path(__file__).resolve().parent
    ckpt = root / "artifacts" / "mlp_digits.pt"
    fig_out = root / "figures" / "fig_cosine_distortion.png"

    model, data = train_or_load_model(checkpoint_path=ckpt)

    acc = accuracy(model, data.x_test, data.y_test)

    h = representations(model, data.x_test)
    d = make_invertible_gauge(h.shape[1])

    gauged_model = GaugedMLP(model, d).eval()

    y_original = model(data.x_test)
    y_transformed = gauged_model(data.x_test)

    max_abs_diff = (y_original - y_transformed).abs().max().item()
    pred_equal = (y_original.argmax(dim=1) == y_transformed.argmax(dim=1)).float().mean().item()

    h_transformed = h @ d.T

    # Pairwise cosine on a subset for speed and visual clarity.
    n = min(350, h.shape[0])
    cos_before = pairwise_cosine(h[:n])
    cos_after = pairwise_cosine(h_transformed[:n])

    save_cosine_distortion_figure(cos_before, cos_after, fig_out)

    iu = np.triu_indices_from(cos_before, k=1)
    cosine_shift = np.mean(np.abs(cos_before[iu] - cos_after[iu]))

    print(f"Test accuracy (original): {acc:.4f}")
    print(f"Max |logits_original - logits_transformed|: {max_abs_diff:.6e}")
    print(f"Prediction agreement: {pred_equal:.4f}")
    print(f"Mean |cos_before - cos_after| (upper triangle): {cosine_shift:.4f}")
    print(f"Saved figure: {fig_out}")


if __name__ == "__main__":
    main()
