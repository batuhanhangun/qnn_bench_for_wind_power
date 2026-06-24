#!/bin/bash
# ============================================================
# QNN WIND BENCHMARK — PERLMUTTER PIPELINE
# ============================================================
# Usage:
#     cd $SCRATCH/qnn_wind_benchmark
#     bash run_on_perlmutter.sh
#
# Submits 4 SLURM jobs:
#   1. splits       → generate data splits
#   2. classical    → all 100 classical experiments
#   3. qnn          → 40 QNN experiments (10 seeds × 4 sizes, 40 parallel array tasks)
#   4. aggregate    → summary tables + stats
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

export QNN_RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXPORT="--export=ALL,QNN_RUN_ID=$QNN_RUN_ID"

mkdir -p slurm/logs

LOG="slurm/logs/pipeline_${QNN_RUN_ID}.log"

echo "=============================================="  | tee "$LOG"
echo " QNN Wind Benchmark (PennyLane) — Perlmutter"    | tee -a "$LOG"
echo " Run ID:  $QNN_RUN_ID"                           | tee -a "$LOG"
echo " Results: results/run_${QNN_RUN_ID}/"            | tee -a "$LOG"
echo "=============================================="  | tee -a "$LOG"

# Validate
[ -f "data/total_dataset.xlsx" ] || { echo "ERROR: data/total_dataset.xlsx not found!" | tee -a "$LOG"; exit 1; }

# Submit pipeline
SPLITS=$(sbatch --parsable $EXPORT slurm/jobs/01_splits.sh)
echo "[1/4] Splits:      $SPLITS"                      | tee -a "$LOG"

CLASSICAL=$(sbatch --parsable $EXPORT --dependency=afterok:$SPLITS slurm/jobs/02_classical.sh)
echo "[2/4] Classical:   $CLASSICAL"                   | tee -a "$LOG"

QNN=$(sbatch --parsable $EXPORT --dependency=afterok:$SPLITS slurm/jobs/03_qnn.sh)
echo "[3/4] QNN:         $QNN"                         | tee -a "$LOG"

AGG=$(sbatch --parsable $EXPORT --dependency=afterok:${CLASSICAL}:${QNN} slurm/jobs/04_aggregate.sh)
echo "[4/4] Aggregate:   $AGG"                         | tee -a "$LOG"

echo ""                                                | tee -a "$LOG"
echo "Done. Monitor: squeue -u \$USER"                 | tee -a "$LOG"
echo "Cancel:  scancel -u \$USER"                      | tee -a "$LOG"
echo "Log:     $LOG"                                   | tee -a "$LOG"
