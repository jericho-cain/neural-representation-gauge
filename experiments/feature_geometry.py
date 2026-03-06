from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.utils import make_orthogonal_gauge, representations, train_or_load_model


def topk_pca_basis(h: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    h_centered = h - h.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(h_centered, full_matrices=False)
    basis = vt[:k].T
    var_ratio = (s[:k] ** 2) / np.sum(s**2)
    return basis, var_ratio


def principal_alignment_cosines(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    _, singular_vals, _ = np.linalg.svd(basis_a.T @ basis_b, full_matrices=False)
    return np.clip(singular_vals, 0.0, 1.0)


def save_feature_geometry_figure(
    cosines: np.ndarray,
    var_before: np.ndarray,
    var_after: np.ndarray,
    out_path: Path,
) -> None:
    idx = np.arange(1, len(cosines) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(idx, cosines, color="#1f77b4", alpha=0.9)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Principal-Angle Cosines")
    axes[0].set_xlabel("component index")
    axes[0].set_ylabel("cos(theta_i)")

    axes[1].plot(idx, var_before, marker="o", label="original")
    axes[1].plot(idx, var_after, marker="o", label="gauged")
    axes[1].set_title("Top-k Explained Variance")
    axes[1].set_xlabel("component index")
    axes[1].set_ylabel("variance ratio")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    ckpt = root / "artifacts" / "mlp_digits.pt"
    out_fig = root / "figures" / "fig_feature_geometry_stability.png"

    model, data = train_or_load_model(checkpoint_path=ckpt)
    h = representations(model, data.x_test).detach().cpu().numpy()

    d = make_orthogonal_gauge(h.shape[1], seed=7).detach().cpu().numpy()
    h_g = h @ d.T

    k = 10
    basis_before, var_before = topk_pca_basis(h, k)
    basis_after, var_after = topk_pca_basis(h_g, k)

    expected_basis_after = d @ basis_before
    cosines = principal_alignment_cosines(expected_basis_after, basis_after)

    save_feature_geometry_figure(cosines, var_before, var_after, out_fig)

    print(f"Min principal-angle cosine: {cosines.min():.6f}")
    print(f"Mean principal-angle cosine: {cosines.mean():.6f}")
    print(f"Max variance-ratio difference: {np.max(np.abs(var_before - var_after)):.6e}")
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    main()
