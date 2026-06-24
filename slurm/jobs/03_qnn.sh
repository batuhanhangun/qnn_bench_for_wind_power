#!/bin/bash
#SBATCH --job-name=qnn_pl
#SBATCH --output=slurm/logs/qnn_%A_%a.out
#SBATCH --error=slurm/logs/qnn_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --qos=regular
#SBATCH --array=0-39

WORK_DIR="$SCRATCH/qnn_wind_benchmark"
cd $WORK_DIR

# 40 tasks: 10 seeds x 4 train sizes (must match config.py)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
SIZES=(800 1600 2400 3200)

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
SIZE_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
SEED=${SEEDS[$SEED_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

echo "=== QNN (PennyLane) Experiment — seed ${SEED}, size ${SIZE} ==="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Run ID: $QNN_RUN_ID"

podman-hpc run --rm \
    -v $WORK_DIR:/workspace:rw \
    -e QNN_RUN_ID=${QNN_RUN_ID} \
    -w /workspace \
    localhost/quantum_ml:v4 \
    python -m src.run_qnn --train_size ${SIZE} --seed ${SEED}

echo "Finished: $(date)"
