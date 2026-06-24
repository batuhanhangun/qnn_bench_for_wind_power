"""Central configuration for the QNN wind power benchmarking pipeline (PennyLane version).

All constants and hyperparameters are defined here. Other modules import from this file
to ensure a single source of truth.
"""

import os

# === Run identification (set by SLURM scripts for timestamped output) ===
# QNN_RUN_ID is exported by run_on_perlmutter.sh → results go to results/run_<RUN_ID>/
RUN_ID = os.environ.get("QNN_RUN_ID", "")

# === Seeds for statistical significance ===
# 10 seeds for mean +/- std reporting and Wilcoxon signed-rank tests
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# === Dataset ===
DATASET_PATH = "data/total_dataset.xlsx"
TEST_SIZE = 893
TRAIN_SIZES = [800, 1600, 2400, 3200]
TARGET_COLUMN = "Power"
CSV_DELIMITER = ";"  # Original data uses semicolon

# === QNN Configuration (L-BFGS-B, reps=3) ===
# PennyLane implementation, gate-for-gate equivalent of the Qiskit VQR
QNN_CONFIG = {
    "framework": "pennylane",
    "feature_map": "ZFeatureMap",       # H + RZ(2x) per qubit
    "feature_map_reps": 1,
    "ansatz": "RealAmplitudes",         # RY + CNOT circular ring
    "ansatz_reps": 3,
    "entanglement": "circular",
    "optimizer": "L_BFGS_B",
    "maxiter": 50,
    "num_qubits": 4,
    "num_parameters": 16,  # 4 * (3+1) = 16
    "device": "lightning.qubit",
    "diff_method": "adjoint",
}

# === ANN Configuration (19 params — comparable to QNN's 16) ===
ANN_CONFIG = {
    "hidden_layer_sizes": (3,),  # 4*3+3 + 3*1+1 = 19 parameters
    "max_iter": 1000,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "param_grid": {
        "model__activation": ["relu", "tanh"],
        "model__alpha": [0.0001, 0.001, 0.01],
        "model__learning_rate_init": [0.001, 0.01],
        "model__solver": ["adam"],
    },
}

# === Classical Model Hyperparameter Grids ===
SVR_PARAM_GRID = {
    "model__kernel": ["rbf", "linear"],
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", "auto"],
}

RF_PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}

XGBOOST_PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
}

DTR_PARAM_GRID = {
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}

# === Cross-Validation ===
CV_FOLDS = 5

# === Paths ===
# RUN_ID is set by SLURM scripts for timestamped output directories
if not RUN_ID:
    import warnings
    warnings.warn(
        "QNN_RUN_ID environment variable not set. "
        "On Perlmutter, run via: bash run_on_perlmutter.sh"
    )
    RUN_ID = "unset"

RESULTS_DIR = os.path.join("results", f"run_{RUN_ID}")
AGGREGATED_DIR = os.path.join("aggregated", f"run_{RUN_ID}")

# === Model Registry ===
CLASSICAL_MODELS = ["ann", "svr", "rf", "dtr", "xgboost"]
ALL_MODELS = ["qnn"] + CLASSICAL_MODELS

