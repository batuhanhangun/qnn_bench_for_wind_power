#!/bin/bash
#SBATCH --job-name=qnn_noise
#SBATCH --output=slurm/logs/noise_%A_%a.out
#SBATCH --error=slurm/logs/noise_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --qos=regular
#SBATCH --array=0-9

# Array over seeds only (each task handles all 4 train sizes for one seed).
# 10 tasks total: one per seed.
# Each task: trains QNN noise-free (deterministic) + evaluates at 6 noise levels
# for all 4 train sizes.

WORK_DIR="$SCRATCH/qnn_wind_benchmark_noise_eval"
cd $WORK_DIR

SEEDS=(42 43 44 45 46 47 48 49 50 51)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== QNN Noise Evaluation — seed ${SEED} (all train sizes) ==="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Run ID: $QNN_RUN_ID"
echo "Noise levels: 0.0, 0.001, 0.005, 0.01, 0.02, 0.05"

podman-hpc run --rm \
    -v $WORK_DIR:/workspace:rw \
    -e QNN_RUN_ID=${QNN_RUN_ID} \
    -w /workspace \
    localhost/quantum_ml:v4 \
    python -m src.run_noise_eval --seed ${SEED}

echo "Finished: $(date)"
