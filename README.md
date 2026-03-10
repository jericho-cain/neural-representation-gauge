# Neural Representation Gauge Experiments

[![arXiv](https://img.shields.io/badge/arXiv-2603.06774-b31b1b.svg)](https://arxiv.org/abs/2603.06774)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18932182.svg)](https://doi.org/10.5281/zenodo.18932182)

This repository contains the experiment suite for the paper draft in `docs/main.tex`.
The goal is to reproduce empirical evidence for three claims:

1. Model function is invariant under invertible gauge transforms when downstream weights are compensated by the inverse transform.
2. Cosine geometry is not gauge invariant.
3. Whitening acts as a canonical gauge fixing that removes covariance anisotropy.

A stronger corroboration is included on CIFAR-10 with a small CNN, including a controlled gauge-strength (`kappa`) sweep.

## Reproducibility Status

- Deterministic seeds are set in code where relevant (`seed=42` defaults).
- All required scripts are runnable via `python -m ...`.
- Tests are included under `tests/`.
- Large local data/model files are intentionally gitignored (`experiments/data/`, `experiments/artifacts/cnn_cifar10.pt`).

## Environment Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Citation

If you use this repository or paper in academic work, cite:

```bibtex
@article{cain2026gauge,
  title={Gauge Freedom in Neural Representation Spaces},
  author={Cain, Jericho},
  journal={arXiv preprint arXiv:2603.06774},
  year={2026},
  url={https://arxiv.org/abs/2603.06774}
}
```

## Replication Paths

### Path A: Fast Digits-only replication (sanity + core theory support)

```bash
. .venv/bin/activate
python -m experiments.gauge_transform
python -m experiments.digits_nn_instability
python -m experiments.whitening_spectrum
python -m experiments.feature_geometry
```

Expected key outputs:

- `experiments/figures/fig_cosine_distortion.png`
- `experiments/figures/fig_nn_instability_digits.png`
- `experiments/figures/fig_whitening_spectrum.png`
- `experiments/figures/fig_feature_geometry_stability.png`
- `experiments/artifacts/metrics_digits_nn.json`
- `experiments/artifacts/metrics_whitening.json`

### Path B: Full paper replication (Digits + CIFAR-10)

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

Expected additional outputs:

- `experiments/figures/fig_cosine_distortion_cifar10.png`
- `experiments/figures/fig_nn_instability_cifar10.png`
- `experiments/figures/fig_cifar10_kappa_sweep.png`
- `experiments/artifacts/metrics_cifar10_nn.json`
- `experiments/artifacts/metrics_cifar10_kappa_sweep.json`
- `experiments/artifacts/metrics_table.md`

## Validation

Run tests:

```bash
. .venv/bin/activate
pytest -q
```

Expected: all tests pass.

## Reproducibility Acceptance Checks

Use these checks to confirm results are consistent with paper claims (small numeric drift is acceptable across environments):

1. Gauge invariance checks:
   - `prediction_agreement` should be `1.0` (or effectively 1.0 within floating-point tolerance).
   - `max_abs_logit_diff` should remain small (`~1e-5` to `1e-4`).

2. Cosine non-invariance:
   - `mean_abs_delta_cos` should be non-zero in both Digits and CIFAR scripts.

3. Neighbor instability:
   - `mean_jaccard_at_k["10"]` should be meaningfully below `1.0` when `kappa > 1`.

4. Whitening effect:
   - `mean_abs_eig_after_minus_1` in `metrics_whitening.json` should be near zero.

5. Kappa sweep behavior:
   - In `metrics_cifar10_kappa_sweep.json`, increasing `kappa` should generally increase distortion metrics and reduce stability metrics.

## Artifact Inventory

Figures are written to `experiments/figures/`.
Metrics and checkpoints are written to `experiments/artifacts/`.
