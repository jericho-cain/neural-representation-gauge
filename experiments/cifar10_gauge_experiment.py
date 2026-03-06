import json
from pathlib import Path

import numpy as np
import torch

from experiments.plots import save_cosine_distortion_figure, save_neighbor_overlap_curve
from experiments.utils import (
    collect_hidden_logits_labels,
    make_invertible_gauge,
    mean_jaccard_curve,
    train_or_load_cifar10_model,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    ckpt = root / "artifacts" / "cnn_cifar10.pt"
    data_root = root / "data"
    fig_cos = root / "figures" / "fig_cosine_distortion_cifar10.png"
    fig_nn = root / "figures" / "fig_nn_instability_cifar10.png"
    metrics_out = root / "artifacts" / "metrics_cifar10_nn.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, loaders = train_or_load_cifar10_model(
        checkpoint_path=ckpt,
        data_root=data_root,
        epochs=5,
        batch_size=128,
        lr=1e-3,
        device=device,
    )

    h, logits, labels = collect_hidden_logits_labels(model, loaders.test_loader, device=device)
    h = h.float()

    d = make_invertible_gauge(h.shape[1], seed=123).to(h.dtype)
    h_g = h @ d.T

    w = model.classifier.weight.detach().cpu()
    b = model.classifier.bias.detach().cpu()
    d_inv = torch.linalg.inv(d)
    w_prime = w @ d_inv

    logits_original = logits.detach().cpu()
    logits_transformed = h_g @ w_prime.T + b

    max_abs_diff = (logits_original - logits_transformed).abs().max().item()
    pred_equal = (
        logits_original.argmax(dim=1) == logits_transformed.argmax(dim=1)
    ).float().mean().item()

    test_acc = (logits_original.argmax(dim=1) == labels).float().mean().item()

    h_np = h.detach().cpu().numpy()
    h_g_np = h_g.detach().cpu().numpy()

    # Use a fixed subset for pairwise cosine diagnostics.
    n = min(800, h_np.shape[0])
    h_np = h_np[:n]
    h_g_np = h_g_np[:n]

    def _pairwise_cos(x: np.ndarray) -> np.ndarray:
        x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
        return x_norm @ x_norm.T

    cos_before = _pairwise_cos(h_np)
    cos_after = _pairwise_cos(h_g_np)
    save_cosine_distortion_figure(cos_before, cos_after, fig_cos)

    iu = np.triu_indices_from(cos_before, k=1)
    mean_abs_delta_cos = float(np.mean(np.abs(cos_before[iu] - cos_after[iu])))

    ks = [1, 5, 10, 20, 50]
    jaccard = mean_jaccard_curve(h_np, h_g_np, ks)
    save_neighbor_overlap_curve(jaccard, fig_nn, title="CIFAR-10: NN Instability Under Gauge Transform")

    metrics = {
        "setting": "cifar10_smallcnn",
        "num_points": n,
        "test_accuracy": test_acc,
        "max_abs_logit_diff": max_abs_diff,
        "prediction_agreement": pred_equal,
        "mean_abs_delta_cos": mean_abs_delta_cos,
        "mean_jaccard_at_k": {str(k): float(v) for k, v in jaccard.items()},
        "figures": [str(fig_cos), str(fig_nn)],
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"CIFAR10 test accuracy: {test_acc:.4f}")
    print(f"Max |logits_original - logits_transformed|: {max_abs_diff:.6e}")
    print(f"Prediction agreement: {pred_equal:.4f}")
    print(f"Mean |cos_before - cos_after|: {mean_abs_delta_cos:.4f}")
    print(f"Mean Jaccard@10: {jaccard[10]:.4f}")
    print(f"Saved figure: {fig_cos}")
    print(f"Saved figure: {fig_nn}")
    print(f"Saved metrics: {metrics_out}")


if __name__ == "__main__":
    main()
