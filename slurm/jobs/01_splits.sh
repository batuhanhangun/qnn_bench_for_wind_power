#!/bin/bash
#SBATCH --job-name=splits
#SBATCH --output=slurm/logs/splits_%j.out
#SBATCH --error=slurm/logs/splits_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --qos=regular

WORK_DIR="$SCRATCH/qnn_wind_benchmark"
cd $WORK_DIR

echo "=== Data Split Generation ==="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"

podman-hpc run --rm \
    -v $WORK_DIR:/workspace:rw \
    -w /workspace \
    localhost/quantum_ml:v4 \
    python -m src.generate_splits --all

echo "Finished: $(date)"
