# Neural Representation Gauge Experiments

This repository contains the empirical experiment suite supporting the manuscript in `docs/main.tex`.

## Scope

The code demonstrates three core claims:

1. Model function is invariant under invertible gauge transforms of hidden representation space when downstream weights are adjusted by the inverse transform.
2. Cosine similarity geometry is not gauge invariant.
3. Whitening acts as a canonical gauge fixing that removes covariance anisotropy.

An additional experiment probes feature-geometry stability under orthogonal gauge transforms.
The expanded suite now includes nearest-neighbor instability in both a pedagogical setting (Digits + MLP) and a stronger setting (CIFAR-10 + small CNN).

## Environment

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run Experiments

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

## Run Tests

```bash
. .venv/bin/activate
pytest -q
```

## Current Findings (Baseline Run)

These values come from the current run in this repo and are expected to vary slightly across environments/seeds.

### Experiment 1: Gauge Transform

- Test accuracy (original MLP): `0.9711`
- Max logit difference after gauge insertion: `1.478195e-05`
- Prediction agreement: `1.0000`
- Mean absolute pairwise cosine shift: `0.1328`

Interpretation: function is preserved, cosine geometry changes.

### Experiment 2: Whitening

- Covariance eigenvalue range before whitening: `[0.0150, 36.5792]`
- Covariance eigenvalue range after whitening: `[1.0000, 1.0000]`
- Mean `|eig_after - 1|`: `0.000006`

Interpretation: whitening removes second-order anisotropy.

### Experiment 3: Feature Geometry Stability

- Min principal-angle cosine between expected and observed transformed PCA subspaces: `1.000000`
- Mean principal-angle cosine: `1.000000`
- Max explained-variance-ratio difference: `2.980232e-08`

Interpretation: under an orthogonal gauge transform, principal subspace structure is preserved up to basis rotation.

### Nearest-Neighbor Instability (Digits MLP)

Run `python -m experiments.digits_nn_instability` to produce:

- `experiments/figures/fig_nn_instability_digits.png`
- `experiments/artifacts/metrics_digits_nn.json`

This reports mean Jaccard overlap vs `k` between top-`k` cosine neighbors before/after gauge transform.

### CIFAR-10 Corroboration (Small CNN)

Run `python -m experiments.cifar10_gauge_experiment` to produce:

- `experiments/figures/fig_cosine_distortion_cifar10.png`
- `experiments/figures/fig_nn_instability_cifar10.png`
- `experiments/artifacts/metrics_cifar10_nn.json`

This replicates the same invariance/distortion/neighbor-instability story on a modern-ish vision benchmark.

### CIFAR-10 Dramatic Panel: Gauge-Strength Sweep

Run `python -m experiments.cifar10_gauge_sweep` to produce:

- `experiments/figures/fig_cifar10_kappa_sweep.png`
- `experiments/artifacts/metrics_cifar10_kappa_sweep.json`

This sweeps condition number `kappa` (1, 2, 5, 10, 20, 50) and reports:

- `mean |Δcos|` (increasing with `kappa`)
- `Jaccard@10` (decreasing with `kappa`)
- top-1 neighbor flip rate (increasing with `kappa`)

while prediction agreement remains 1.0 after gauge compensation.

## Artifacts

- `experiments/figures/fig_cosine_distortion.png`
- `experiments/figures/fig_covariance_spectrum.png`
- `experiments/figures/fig_whitening_spectrum.png`
- `experiments/figures/fig_whitening_effect.png`
- `experiments/figures/fig_feature_geometry_stability.png`
- `experiments/artifacts/mlp_digits.pt`
- `experiments/artifacts/metrics_table.md`

## Notes

- This README is a findings-first baseline and will later be refactored into strict reproducibility instructions.
- The manuscript lives in `docs/main.tex`; no LaTeX build steps are required in this repository workflow.
