# QNN Noise Robustness Evaluation

Hardware-robustness companion to the main benchmark. Where the main pipeline
measures *algorithmic* robustness (multi-seed variance), this pipeline measures
*hardware* robustness by evaluating the trained QNN under a depolarizing-noise
sweep.

## Method

The QNN is retrained noise-free using the **same seeds and data splits as the
main benchmark**, producing identical weights. The frozen weights are then
evaluated on the held-out test set under a single-qubit `DepolarizingChannel`
applied after every gate, at six error rates:

```
p ∈ {0.000, 0.001, 0.005, 0.010, 0.020, 0.050}
```

- `p = 0.000` uses exact statevector simulation (`lightning.qubit`) and matches
  the main benchmark's test metrics — a built-in consistency check.
- `p > 0` uses the mixed-state simulator (`default.mixed`).

All configuration is in `config.py`.

## Usage

```bash
# Single experiment:
python -m src.run_noise_eval --train_size 800 --seed 42

# All train sizes for one seed (used by the SLURM array):
python -m src.run_noise_eval --seed 42

# Aggregate across seeds:
python -m src.aggregate_noise
```

Aggregation produces, under `aggregated/run_<RUN_ID>/`:

- `noise_summary.csv` — mean ± std of test metrics per (noise level, train size)
- `noise_confidence_intervals.csv` — 95% CIs (t-based and normal approximation)
- `noise_statistical_tests.txt` — Wilcoxon signed-rank tests, noisy (p>0) vs. p=0
- `noise_latex_table.tex` — formatted table for the paper

## Dataset

The dataset lives once at the repository root: `../data/total_dataset.xlsx`.
Copy it to `data/total_dataset.xlsx` here before running:

```bash
cp ../data/total_dataset.xlsx data/total_dataset.xlsx
python -m src.generate_splits --all
```

## HPC

`run_noise_pipeline.sh` submits the dependency-chained SLURM jobs in
`slurm/jobs/` (`splits → noise_eval array → aggregate`). Set the SLURM account
and container image in those scripts for your allocation before submitting.

## Tracked results

The `results/` and `aggregated/` directories are committed here (unlike the main
pipeline, where they are regenerable HPC outputs) because they back the noise
figures and tables in the paper.
