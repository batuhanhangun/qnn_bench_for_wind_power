#!/bin/bash
# ============================================================
# QNN NOISE ROBUSTNESS PIPELINE — PERLMUTTER
# ============================================================
# Hardware robustness under depolarizing noise.
#
# Submits 3 SLURM jobs:
#   1. splits     → generate data splits (same seeds as the main benchmark)
#   2. noise_eval → 10 array tasks (one per seed, all 4 train sizes,
#                   6 noise levels each)
#   3. aggregate  → noise_summary.csv, CIs, LaTeX table, Wilcoxon tests
#
# Usage:
#     cd $SCRATCH/qnn_wind_benchmark_noise_eval
#     bash run_noise_pipeline.sh
#
# Prerequisites:
#     - data/total_dataset.xlsx must be present
#     - Container image localhost/quantum_ml:v4 must be available
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

export QNN_RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXPORT="--export=ALL,QNN_RUN_ID=$QNN_RUN_ID"

mkdir -p slurm/logs

LOG="slurm/logs/pipeline_${QNN_RUN_ID}.log"

echo "=============================================="  | tee "$LOG"
echo " QNN Noise Robustness Evaluation — Perlmutter" | tee -a "$LOG"
echo " Run ID:  $QNN_RUN_ID"                          | tee -a "$LOG"
echo " Results: results/run_${QNN_RUN_ID}/"           | tee -a "$LOG"
echo " Noise levels: 0.0, 0.001, 0.005, 0.01, 0.02, 0.05" | tee -a "$LOG"
echo "=============================================="  | tee -a "$LOG"

# Validate dataset
[ -f "data/total_dataset.xlsx" ] || {
    echo "ERROR: data/total_dataset.xlsx not found!" | tee -a "$LOG"
    exit 1
}

# Submit pipeline
SPLITS=$(sbatch --parsable $EXPORT slurm/jobs/01_splits.sh)
echo "[1/3] Splits:     $SPLITS"   | tee -a "$LOG"

NOISE=$(sbatch --parsable $EXPORT --dependency=afterok:$SPLITS slurm/jobs/02_noise_eval.sh)
echo "[2/3] Noise eval: $NOISE"    | tee -a "$LOG"

AGG=$(sbatch --parsable $EXPORT --dependency=afterok:$NOISE slurm/jobs/03_aggregate.sh)
echo "[3/3] Aggregate:  $AGG"      | tee -a "$LOG"

echo ""                            | tee -a "$LOG"
echo "Done. Monitor: squeue -u \$USER"  | tee -a "$LOG"
echo "Cancel:  scancel -u \$USER"       | tee -a "$LOG"
echo "Log:     $LOG"                    | tee -a "$LOG"
