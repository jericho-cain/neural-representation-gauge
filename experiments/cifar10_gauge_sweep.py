import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.utils import (
    collect_hidden_logits_labels,
    make_conditioned_gauge,
    mean_jaccard_curve,
    train_or_load_cifar10_model,
)


def pairwise_cosine(x: np.ndarray) -> np.ndarray:
    x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return x_norm @ x_norm.T


def top1_flip_rate(h_before: np.ndarray, h_after: np.ndarray) -> float:
    sim_a = pairwise_cosine(h_before)
    sim_b = pairwise_cosine(h_after)
    np.fill_diagonal(sim_a, -np.inf)
    np.fill_diagonal(sim_b, -np.inf)
    n1 = np.argmax(sim_a, axis=1)
    n2 = np.argmax(sim_b, axis=1)
    return float(np.mean(n1 != n2))


def save_sweep_figure(
    kappas: list[float],
    deltas: list[float],
    j10: list[float],
    top1: list[float],
    agreement_note: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    axes[0].plot(kappas, deltas, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_title("Cosine Distortion vs Gauge Strength")
    axes[0].set_xlabel("condition number kappa")
    axes[0].set_ylabel("mean |Δcos|")

    axes[1].plot(kappas, j10, marker="o", color="#d62728")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("NN Stability vs Gauge Strength")
    axes[1].set_xlabel("condition number kappa")
    axes[1].set_ylabel("mean Jaccard@10")

    axes[2].plot(kappas, top1, marker="o", color="#2ca02c")
    axes[2].set_xscale("log")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("Top-1 Neighbor Flip Rate")
    axes[2].set_xlabel("condition number kappa")
    axes[2].set_ylabel("flip rate")

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.suptitle("Gauge transforms distort cosine geometry while preserving model function", y=0.995, fontsize=13)
    fig.text(
        0.5,
        0.905,
        agreement_note,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    ckpt = root / "artifacts" / "cnn_cifar10.pt"
    data_root = root / "data"

    fig_out = root / "figures" / "fig_cifar10_kappa_sweep.png"
    metrics_out = root / "artifacts" / "metrics_cifar10_kappa_sweep.json"

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

    test_acc = (logits.argmax(dim=1) == labels).float().mean().item()

    # Fixed subset for efficient pairwise comparisons.
    n = min(800, h.shape[0])
    h_base = h[:n]
    h_base_np = h_base.detach().cpu().numpy()

    kappas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    rows = []
    deltas = []
    j10s = []
    top1s = []

    w = model.classifier.weight.detach().cpu()
    b = model.classifier.bias.detach().cpu()
    logits_base = logits[:n].detach().cpu()

    for i, kappa in enumerate(kappas):
        d = make_conditioned_gauge(h_base.shape[1], kappa=kappa, seed=123 + i).to(h.dtype)
        d_inv = torch.linalg.inv(d)

        h_g = h_base @ d.T
        w_prime = w @ d_inv
        logits_g = h_g @ w_prime.T + b

        max_abs_diff = (logits_base - logits_g).abs().max().item()
        pred_equal = (
            logits_base.argmax(dim=1) == logits_g.argmax(dim=1)
        ).float().mean().item()

        h_g_np = h_g.detach().cpu().numpy()
        cos_before = pairwise_cosine(h_base_np)
        cos_after = pairwise_cosine(h_g_np)
        iu = np.triu_indices_from(cos_before, k=1)
        mean_abs_delta_cos = float(np.mean(np.abs(cos_before[iu] - cos_after[iu])))

        jacc = mean_jaccard_curve(h_base_np, h_g_np, [10])[10]
        flip = top1_flip_rate(h_base_np, h_g_np)

        deltas.append(mean_abs_delta_cos)
        j10s.append(jacc)
        top1s.append(flip)

        rows.append(
            {
                "kappa": kappa,
                "max_abs_logit_diff": max_abs_diff,
                "prediction_agreement": pred_equal,
                "mean_abs_delta_cos": mean_abs_delta_cos,
                "mean_jaccard_at_10": jacc,
                "top1_flip_rate": flip,
            }
        )

    min_agree = min(r["prediction_agreement"] for r in rows)
    max_agree = max(r["prediction_agreement"] for r in rows)
    if np.isclose(min_agree, max_agree):
        note = f"Prediction agreement = {min_agree:.1f} for all $\\kappa$"
    else:
        note = f"Prediction agreement range: [{min_agree:.3f}, {max_agree:.3f}]"

    save_sweep_figure(kappas, deltas, j10s, top1s, note, fig_out)

    payload = {
        "setting": "cifar10_smallcnn_kappa_sweep",
        "num_points": n,
        "test_accuracy": test_acc,
        "rows": rows,
        "figure": str(fig_out),
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(payload, indent=2))

    print(f"CIFAR10 test accuracy (reference): {test_acc:.4f}")
    print("kappa sweep results:")
    for r in rows:
        print(
            "  kappa={kappa:>4.1f} | mean|Δcos|={d:.4f} | Jaccard@10={j:.4f} | top1-flip={f:.4f} | max|Δlogits|={m:.2e}".format(
                kappa=r["kappa"],
                d=r["mean_abs_delta_cos"],
                j=r["mean_jaccard_at_10"],
                f=r["top1_flip_rate"],
                m=r["max_abs_logit_diff"],
            )
        )
    print(f"Saved figure: {fig_out}")
    print(f"Saved metrics: {metrics_out}")


if __name__ == "__main__":
    main()
