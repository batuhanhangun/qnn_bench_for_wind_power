"""Central configuration for the QNN noise robustness evaluation pipeline.

This pipeline distinguishes algorithmic robustness (multi-seed variance, handled
by the main benchmark) from hardware robustness (this project: a depolarizing
noise sweep on trained QNN weights).

The QNN is retrained deterministically using the same seeds and data splits as
the main benchmark, producing identical weights. Evaluation then sweeps over six
depolarizing error rates. The p=0.0 result serves as a built-in consistency
check against the noise-free benchmark.
"""

import os

# === Run identification ===
RUN_ID = os.environ.get("QNN_RUN_ID", "")

# === Seeds (must match the main benchmark exactly for consistency) ===
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# === Dataset ===
DATASET_PATH = "data/total_dataset.xlsx"
TEST_SIZE = 893
TRAIN_SIZES = [800, 1600, 2400, 3200]
TARGET_COLUMN = "Power"
CSV_DELIMITER = ";"

# === Noise levels for depolarizing sweep ===
# p=0.000 : noise-free baseline (should match the main benchmark results)
# p=0.001 : best current superconducting hardware (~0.1 % gate error)
# p=0.005 : near-term realistic
# p=0.010 : moderate noise
# p=0.020 : noisy regime
# p=0.050 : high noise / stress test
NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

# === QNN configuration (locked — must match the main benchmark exactly) ===
QNN_CONFIG = {
    "framework": "pennylane",
    "feature_map": "ZFeatureMap",
    "feature_map_reps": 1,
    "ansatz": "RealAmplitudes",
    "ansatz_reps": 3,
    "entanglement": "circular",
    "optimizer": "L_BFGS_B",
    "maxiter": 50,
    "num_qubits": 4,
    "num_parameters": 16,
    "device_train": "lightning.qubit",   # noise-free training
    "device_eval": "default.mixed",      # noisy evaluation
    "diff_method": "adjoint",
}

# === Cross-validation (for reference, not used in noise eval) ===
CV_FOLDS = 5

# === Paths ===
if not RUN_ID:
    import warnings
    warnings.warn(
        "QNN_RUN_ID environment variable not set. "
        "On Perlmutter, run via: bash run_noise_pipeline.sh"
    )
    RUN_ID = "unset"

RESULTS_DIR = os.path.join("results", f"run_{RUN_ID}")
AGGREGATED_DIR = os.path.join("aggregated", f"run_{RUN_ID}")
