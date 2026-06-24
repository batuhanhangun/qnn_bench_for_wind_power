#!/bin/bash
#SBATCH --job-name=classical
#SBATCH --output=slurm/logs/classical_%j.out
#SBATCH --error=slurm/logs/classical_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=cpu
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --qos=regular

WORK_DIR="$SCRATCH/qnn_wind_benchmark"
cd $WORK_DIR

echo "=== All Classical Experiments (200 total: 5 models x 10 seeds x 4 sizes) ==="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $QNN_RUN_ID"

podman-hpc run --rm \
    -v $WORK_DIR:/workspace:rw \
    -e QNN_RUN_ID=${QNN_RUN_ID} \
    -w /workspace \
    localhost/quantum_ml:v4 \
    python -m src.run_classical --all

echo "Finished: $(date)"
