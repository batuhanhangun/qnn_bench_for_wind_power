"""Shared utilities for the QNN noise robustness evaluation pipeline."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_data(seed: int, train_size: int):
    """Load training and test data for a specific seed and train size.

    Args:
        seed: Random seed used for data split
        train_size: Number of training samples

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as pandas objects
    """
    data_dir = Path(f"data/seed_{seed}")

    train_path = data_dir / f"train_{train_size}.csv"
    test_path = data_dir / "test_set.csv"

    if not train_path.exists() or not test_path.exists():
        missing = train_path if not train_path.exists() else test_path
        raise FileNotFoundError(
            f"Data split not found: {missing}\n"
            f"Run: python -m src.generate_splits --all"
        )

    train_df = pd.read_csv(train_path, delimiter=config.CSV_DELIMITER)
    test_df = pd.read_csv(test_path, delimiter=config.CSV_DELIMITER)

    X_train = train_df.drop(config.TARGET_COLUMN, axis=1)
    y_train = train_df[config.TARGET_COLUMN]
    X_test = test_df.drop(config.TARGET_COLUMN, axis=1)
    y_test = test_df[config.TARGET_COLUMN]

    return X_train, y_train, X_test, y_test


def compute_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics.

    Returns:
        Dictionary with keys: r2, mse, rmse, mae
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def save_metrics(metrics, path: str):
    """Save metrics (dict or list of dicts) to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def print_flush(msg: str):
    """Print message and flush stdout for HPC log buffering."""
    print(msg)
    sys.stdout.flush()
