from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def save_cosine_distortion_figure(cos_before: np.ndarray, cos_after: np.ndarray, out_path: Path) -> None:
    iu = np.triu_indices_from(cos_before, k=1)
    a = cos_before[iu]
    b = cos_after[iu]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(a, bins=40, alpha=0.85, color="#1f77b4")
    axes[0].set_title("Cosine Before Gauge")
    axes[0].set_xlabel("cos(h_i, h_j)")
    axes[0].set_ylabel("count")

    axes[1].hist(b, bins=40, alpha=0.85, color="#ff7f0e")
    axes[1].set_title("Cosine After Gauge")
    axes[1].set_xlabel("cos(Dh_i, Dh_j)")

    axes[2].scatter(a, b, s=5, alpha=0.35, color="#2ca02c")
    axes[2].plot([-1, 1], [-1, 1], "k--", linewidth=1)
    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(-1, 1)
    axes[2].set_title("Cosine Distortion")
    axes[2].set_xlabel("before")
    axes[2].set_ylabel("after")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_covariance_spectrum(cov_before: np.ndarray, cov_after: np.ndarray, out_path: Path) -> None:
    eig_before = np.linalg.eigvalsh(cov_before)
    eig_after = np.linalg.eigvalsh(cov_after)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.sort(eig_before)[::-1], marker="o", linewidth=1.2, label="before whitening")
    ax.plot(np.sort(eig_after)[::-1], marker="o", linewidth=1.2, label="after whitening")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
    ax.set_title("Covariance Eigenvalue Spectrum")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_whitening_pca(h_before: np.ndarray, h_after: np.ndarray, out_path: Path) -> None:
    pca_b = PCA(n_components=2)
    pca_a = PCA(n_components=2)
    z_b = pca_b.fit_transform(h_before)
    z_a = pca_a.fit_transform(h_after)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(z_b[:, 0], z_b[:, 1], s=7, alpha=0.35, color="#1f77b4")
    axes[0].set_title("PCA Before Whitening")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(z_a[:, 0], z_a[:, 1], s=7, alpha=0.35, color="#ff7f0e")
    axes[1].set_title("PCA After Whitening")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_neighbor_overlap_curve(jaccard_by_k: dict[int, float], out_path: Path, title: str) -> None:
    ks = sorted(jaccard_by_k.keys())
    vals = [jaccard_by_k[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, vals, marker="o", linewidth=1.5, color="#d62728")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.set_xlabel("k (top-k neighbors)")
    ax.set_ylabel("mean Jaccard overlap")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
