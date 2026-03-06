from pathlib import Path

import numpy as np

from experiments.plots import save_covariance_spectrum, save_whitening_pca
from experiments.utils import representations, train_or_load_model, whiten


def main() -> None:
    root = Path(__file__).resolve().parent
    ckpt = root / "artifacts" / "mlp_digits.pt"
    fig_spectrum = root / "figures" / "fig_covariance_spectrum.png"
    fig_pca = root / "figures" / "fig_whitening_effect.png"

    model, data = train_or_load_model(checkpoint_path=ckpt)

    h = representations(model, data.x_test).detach().cpu().numpy()
    h_white, cov_before, cov_after = whiten(h)

    eig_before = np.linalg.eigvalsh(cov_before)
    eig_after = np.linalg.eigvalsh(cov_after)

    save_covariance_spectrum(cov_before, cov_after, fig_spectrum)
    save_whitening_pca(h, h_white, fig_pca)

    print(f"Eigenvalue range before whitening: [{eig_before.min():.4f}, {eig_before.max():.4f}]")
    print(f"Eigenvalue range after whitening:  [{eig_after.min():.4f}, {eig_after.max():.4f}]")
    print(f"Mean |eig_after - 1|: {np.mean(np.abs(eig_after - 1.0)):.6f}")
    print(f"Saved figure: {fig_spectrum}")
    print(f"Saved figure: {fig_pca}")


if __name__ == "__main__":
    main()
