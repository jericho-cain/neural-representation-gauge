# Experiments

This directory contains reproducible experiments for:

1. Gauge invariance of model function under invertible representation transforms.
2. Cosine similarity distortion under gauge transforms.
3. Nearest-neighbor instability under gauge transforms.
4. Whitening as a canonical gauge fixing.
5. PCA feature-geometry stability under orthogonal gauge transforms.
6. A stronger setting on CIFAR-10 with a small CNN.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
. .venv/bin/activate
python -m experiments.gauge_transform
python -m experiments.digits_nn_instability
python -m experiments.whitening
python -m experiments.whitening_spectrum
python -m experiments.feature_geometry
python -m experiments.cifar10_gauge_experiment
python -m experiments.cifar10_gauge_sweep
python -m experiments.summarize_metrics
```

## Outputs

Generated figures:

- `experiments/figures/fig_cosine_distortion.png`
- `experiments/figures/fig_nn_instability_digits.png`
- `experiments/figures/fig_covariance_spectrum.png`
- `experiments/figures/fig_whitening_spectrum.png`
- `experiments/figures/fig_whitening_effect.png`
- `experiments/figures/fig_feature_geometry_stability.png`
- `experiments/figures/fig_cosine_distortion_cifar10.png`
- `experiments/figures/fig_nn_instability_cifar10.png`
- `experiments/figures/fig_cifar10_kappa_sweep.png`

Model checkpoint:

- `experiments/artifacts/mlp_digits.pt`
- `experiments/artifacts/cnn_cifar10.pt`
- `experiments/artifacts/metrics_digits_nn.json`
- `experiments/artifacts/metrics_cifar10_nn.json`
- `experiments/artifacts/metrics_cifar10_kappa_sweep.json`
- `experiments/artifacts/metrics_whitening.json`
- `experiments/artifacts/metrics_table.md`

## Tests

```bash
. .venv/bin/activate
pytest -q
```
