#!/bin/bash
#SBATCH --job-name=noise_agg
#SBATCH --output=slurm/logs/aggregate_%j.out
#SBATCH --error=slurm/logs/aggregate_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --qos=regular

WORK_DIR="$SCRATCH/qnn_wind_benchmark_noise_eval"
cd $WORK_DIR

echo "=== Noise Results Aggregation ==="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $QNN_RUN_ID"

podman-hpc run --rm \
    -v $WORK_DIR:/workspace:rw \
    -e QNN_RUN_ID=${QNN_RUN_ID} \
    -w /workspace \
    localhost/quantum_ml:v4 \
    python -m src.aggregate_noise

echo "Finished: $(date)"
