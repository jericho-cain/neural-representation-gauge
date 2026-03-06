# Experiments

This directory contains all runnable experiment entrypoints used in the paper.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Entrypoints

### Digits / pedagogical setting

```bash
python -m experiments.gauge_transform
python -m experiments.digits_nn_instability
python -m experiments.whitening
python -m experiments.whitening_spectrum
python -m experiments.feature_geometry
```

### CIFAR-10 / stronger setting

```bash
python -m experiments.cifar10_gauge_experiment
python -m experiments.cifar10_gauge_sweep
```

### Metrics summary

```bash
python -m experiments.summarize_metrics
```

## Main outputs

### Figures (`experiments/figures/`)

- `fig_cosine_distortion.png`
- `fig_nn_instability_digits.png`
- `fig_whitening_spectrum.png`
- `fig_whitening_effect.png`
- `fig_feature_geometry_stability.png`
- `fig_cosine_distortion_cifar10.png`
- `fig_nn_instability_cifar10.png`
- `fig_cifar10_kappa_sweep.png`

### Artifacts (`experiments/artifacts/`)

- `mlp_digits.pt`
- `cnn_cifar10.pt` (local large file; gitignored)
- `metrics_digits_nn.json`
- `metrics_whitening.json`
- `metrics_cifar10_nn.json`
- `metrics_cifar10_kappa_sweep.json`
- `metrics_table.md`

## Tests

From repo root:

```bash
pytest -q
```

## Notes

- `experiments/data/` is used for local CIFAR-10 download cache and is gitignored.
- Scripts are designed to be re-runnable; existing checkpoints are reused if present.
