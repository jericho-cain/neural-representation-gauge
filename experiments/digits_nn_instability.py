import json
from pathlib import Path

import numpy as np

from experiments.gauge_transform import GaugedMLP
from experiments.plots import save_neighbor_overlap_curve
from experiments.utils import (
    make_invertible_gauge,
    mean_jaccard_curve,
    representations,
    train_or_load_model,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    ckpt = root / "artifacts" / "mlp_digits.pt"
    fig_out = root / "figures" / "fig_nn_instability_digits.png"
    metrics_out = root / "artifacts" / "metrics_digits_nn.json"

    model, data = train_or_load_model(checkpoint_path=ckpt)
    h = representations(model, data.x_test)

    d = make_invertible_gauge(h.shape[1], seed=42)
    gauged_model = GaugedMLP(model, d).eval()

    y_original = model(data.x_test)
    y_transformed = gauged_model(data.x_test)
    max_abs_diff = (y_original - y_transformed).abs().max().item()
    pred_equal = (y_original.argmax(dim=1) == y_transformed.argmax(dim=1)).float().mean().item()

    h_np = h.detach().cpu().numpy()
    h_g_np = (h @ d.T).detach().cpu().numpy()

    # Keep subset fixed so results are fast and stable.
    n = min(350, h_np.shape[0])
    h_np = h_np[:n]
    h_g_np = h_g_np[:n]

    def _pairwise_cos(x: np.ndarray) -> np.ndarray:
        x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
        return x_norm @ x_norm.T

    cos_before = _pairwise_cos(h_np)
    cos_after = _pairwise_cos(h_g_np)
    iu = np.triu_indices_from(cos_before, k=1)
    mean_abs_delta_cos = float(np.mean(np.abs(cos_before[iu] - cos_after[iu])))

    ks = [1, 5, 10, 20, 50]
    jaccard = mean_jaccard_curve(h_np, h_g_np, ks)

    save_neighbor_overlap_curve(jaccard, fig_out, title="Digits: NN Instability Under Gauge Transform")

    metrics = {
        "setting": "digits_mlp",
        "num_points": n,
        "max_abs_logit_diff": max_abs_diff,
        "prediction_agreement": pred_equal,
        "mean_abs_delta_cos": mean_abs_delta_cos,
        "mean_jaccard_at_k": {str(k): float(v) for k, v in jaccard.items()},
        "figure": str(fig_out),
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"Max |logits_original - logits_transformed|: {max_abs_diff:.6e}")
    print(f"Prediction agreement: {pred_equal:.4f}")
    print(f"Mean |cos_before - cos_after|: {mean_abs_delta_cos:.4f}")
    print(f"Mean Jaccard@10: {jaccard[10]:.4f}")
    print(f"Saved figure: {fig_out}")
    print(f"Saved metrics: {metrics_out}")


if __name__ == "__main__":
    main()
