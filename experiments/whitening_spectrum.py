import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.utils import representations, train_or_load_model


def covariance_from_columns(h_col_major: np.ndarray) -> np.ndarray:
    # h_col_major has shape (d, n), matching Sigma = (1/n) H H^T.
    n = h_col_major.shape[1]
    return (h_col_major @ h_col_major.T) / n


def whitening_from_covariance(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    evals, evecs = np.linalg.eigh(cov)
    inv_sqrt = 1.0 / np.sqrt(np.clip(evals, eps, None))
    return evecs @ np.diag(inv_sqrt) @ evecs.T


def main() -> None:
    root = Path(__file__).resolve().parent
    ckpt = root / "artifacts" / "mlp_digits.pt"
    fig_out = root / "figures" / "fig_whitening_spectrum.png"
    metrics_out = root / "artifacts" / "metrics_whitening.json"

    model, data = train_or_load_model(checkpoint_path=ckpt)

    # representations(...) returns shape (n, d). Convert to (d, n) for this experiment.
    h = representations(model, data.x_test).detach().cpu().numpy().T

    sigma = covariance_from_columns(h)
    eigvals_before = np.linalg.eigvalsh(sigma)

    d_white = whitening_from_covariance(sigma)
    h_white = d_white @ h
    sigma_white = covariance_from_columns(h_white)
    eigvals_after = np.linalg.eigvalsh(sigma_white)

    eigvals_before_sorted = np.sort(eigvals_before)[::-1]
    eigvals_after_sorted = np.sort(eigvals_after)[::-1]

    plt.figure(figsize=(7, 4.5))
    plt.plot(eigvals_before_sorted, label="before whitening")
    plt.plot(eigvals_after_sorted, label="after whitening")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, color="k", alpha=0.8, label="identity (\u03bb=1)")
    plt.yscale("log")
    plt.ylim(1e-3, float(np.max(eigvals_before)) * 1.1)
    plt.xlabel("component index")
    plt.ylabel("covariance eigenvalue")
    plt.legend()
    plt.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_out, dpi=180)
    plt.close()

    metrics = {
        "min_eig_before": float(np.min(eigvals_before)),
        "max_eig_before": float(np.max(eigvals_before)),
        "mean_abs_eig_after_minus_1": float(np.mean(np.abs(eigvals_after - 1.0))),
        "figure": str(fig_out),
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"min_eig_before: {metrics['min_eig_before']:.6f}")
    print(f"max_eig_before: {metrics['max_eig_before']:.6f}")
    print(f"mean_abs(|eig_after - 1|): {metrics['mean_abs_eig_after_minus_1']:.6e}")
    print(f"Saved figure: {fig_out}")
    print(f"Saved metrics: {metrics_out}")


if __name__ == "__main__":
    main()
