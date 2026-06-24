# Quantum Neural Network Benchmark for Wind Power Prediction

A reproducible benchmarking pipeline comparing a Variational Quantum Regressor (VQR)
against standard classical machine-learning models for short-term wind power prediction.

This repository contains the code accompanying the paper. It provides everything needed
to regenerate the data splits, train every model across multiple seeds and training-set
sizes, and aggregate the results into the tables and statistics reported in the paper.

## Overview

The pipeline trains a quantum neural network and five classical baselines on a common set
of train/test splits, then aggregates metrics across random seeds to enable statistical
comparison.

**Models**

| Model | Description |
| ----- | ----------- |
| QNN   | Variational Quantum Regressor (PennyLane, `lightning.qubit`) |
| ANN   | `MLPRegressor`, `hidden_layer_sizes=(3,)` — 19 parameters, comparable to QNN's 16 |
| SVR   | Support Vector Regressor |
| RF    | Random Forest Regressor |
| DTR   | Decision Tree Regressor |
| XGBoost | Gradient-boosted trees |

**Experimental design**

- **Seeds:** 10 random seeds (`42`–`51`) for mean ± std reporting and Wilcoxon signed-rank tests.
- **Training sizes:** 800, 1600, 2400, 3200 samples.
- **Test set:** 893 held-out samples.
- **QNN architecture:** `ZFeatureMap` (reps=1) + `RealAmplitudes` (reps=3, circular
  entanglement), 4 qubits, 16 trainable parameters, optimized with L-BFGS-B (maxiter=50).
- **Outputs per experiment:** metrics (JSON), predicted-vs-actual values (CSV), and training
  loss curves (CSV) for QNN and ANN.

All configuration lives in `config.py` as a single source of truth.

## Installation

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

The dataset is included at `data/total_dataset.xlsx`.

## Usage

### Generate data splits

```bash
python -m src.generate_splits --seed 42   # single seed
python -m src.generate_splits --all        # all seeds
```

This creates `data/seed_<N>/` containing `test_set.csv` and `train_{800,1600,2400,3200}.csv`.

### Train a classical model

```bash
python -m src.run_classical --model rf --train_size 800 --seed 42
```

Results are written to `results/<model>/seed_<N>/` as `exp_<size>_metrics.json` and
`exp_<size>_predictions.csv`. ANN additionally writes `exp_<size>_convergence.csv`.

### Train the QNN

```bash
# single experiment (CPU-only quantum simulation — runs for hours):
python -m src.run_qnn --train_size 800 --seed 42

# all four training sizes for one seed:
python -m src.run_qnn --seed 42
```

### Aggregate results

```bash
python -m src.aggregate_results
```

Outputs in `aggregated/run_<RUN_ID>/`:

- `summary_table.csv` — mean ± std across seeds
- `latex_table.tex` — formatted table for the paper
- `statistical_tests.txt` — Wilcoxon signed-rank test results (QNN vs. classical)

## Noise Robustness Evaluation

A companion pipeline in [`qnn_wind_benchmark_noise_eval/`](qnn_wind_benchmark_noise_eval/)
evaluates the trained QNN under a depolarizing-noise sweep (six error rates) to
assess hardware robustness, complementing the multi-seed algorithmic robustness
measured here. Its committed `results/` and `aggregated/` outputs back the noise
figures and tables in the paper. See its
[README](qnn_wind_benchmark_noise_eval/README.md) for details.

## HPC Deployment (SLURM)

The full benchmark is designed to run on an HPC cluster (developed for NERSC Perlmutter).
Pre-generated SLURM scripts live in `slurm/jobs/`.

```bash
# On the cluster login node:
bash run_on_perlmutter.sh
```

The launcher submits a dependency-chained set of jobs:

```text
splits → classical experiments + QNN array tasks → aggregation
```

Set the SLURM account and container image for your allocation in `run_on_perlmutter.sh`
and the scripts under `slurm/jobs/` before submitting.

| Job         | Resources    | What it runs                       |
| ----------- | ------------ | ---------------------------------- |
| Splits      | 1 CPU, 30m   | Data preprocessing                 |
| Classical   | 32 CPU, 12h  | 5 models × 4 sizes × N seeds       |
| QNN (array) | 16 CPU, 12h  | Parallel array tasks, 1 experiment each |
| Aggregation | 1 CPU, 30m   | Summary tables + statistics        |

Monitor and retrieve:

```bash
squeue -u $USER
tail -f slurm/logs/*.out
scp -r USER@cluster:~/qnn_wind_benchmark/aggregated/ ./aggregated/
```

## Repository Layout

```
.
├── config.py                # All configuration (single source of truth)
├── requirements.txt
├── run_on_perlmutter.sh     # SLURM pipeline launcher
├── data/
│   ├── total_dataset.xlsx   # Dataset
│   └── seed_*/              # Generated splits (gitignored)
├── src/
│   ├── utils.py             # Shared utilities
│   ├── generate_splits.py   # Data split generator
│   ├── run_classical.py     # Classical model runner
│   ├── run_qnn.py           # QNN runner
│   └── aggregate_results.py # Multi-seed aggregation + statistics
├── slurm/
│   └── jobs/                # Pre-generated SLURM scripts
├── results/                 # Experiment outputs (gitignored)
└── aggregated/              # Aggregated results (gitignored)
```

## Output Formats

**Metrics** (`exp_<size>_metrics.json`)

```json
{
  "model": "rf", "train_size": 800, "seed": 42,
  "train_r2": 0.98, "test_r2": 0.93,
  "test_rmse": 188.0, "test_mae": 65.0,
  "cv_mean_r2": 0.92, "cv_std_r2": 0.01,
  "best_params": {}, "wall_time_seconds": 45.3
}
```

**Predictions** (`exp_<size>_predictions.csv`): `Actual,Predicted`

**Convergence** (`exp_<size>_convergence.csv`, ANN/QNN): `epoch,loss`

## Notes

- Input data splits use a semicolon (`;`) delimiter; output files use commas.
- QNN experiments run on CPU (statevector simulation) and take several hours each.

## Citation

If you use this code, please cite the accompanying paper. (BibTeX to be added upon publication.)

## License

See [LICENSE](LICENSE).
