"""Unified QNN runner using PennyLane with fixed configuration.

Gate-for-gate equivalent of the Qiskit VQR version:
  - Feature map: ZFeatureMap(4, reps=1) -> H + RZ(2x)
  - Ansatz: RealAmplitudes(4, reps=3, circular) -> RY + CNOT ring
  - Measurement: Z x Z x Z x Z expectation value
  - Optimizer: L-BFGS-B (scipy) with maxiter=50
  - 16 trainable parameters

Usage:
    python -m src.run_qnn --train_size 800 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import (
    compute_metrics,
    load_data,
    print_flush,
    save_convergence,
    save_metrics,
    save_predictions,
)

# --- Quantum circuit ---

NUM_QUBITS = config.QNN_CONFIG["num_qubits"]
ANSATZ_REPS = config.QNN_CONFIG["ansatz_reps"]
WEIGHT_SHAPE = (ANSATZ_REPS + 1, NUM_QUBITS)  # (4, 4) = 16 params

dev = qml.device(config.QNN_CONFIG["device"], wires=NUM_QUBITS)


@qml.qnode(dev, interface="autograd", diff_method=config.QNN_CONFIG["diff_method"])
def circuit(weights, x):
    """Quantum circuit matching Qiskit's ZFeatureMap + RealAmplitudes + Z^4.

    Args:
        weights: Trainable parameters, shape (ansatz_reps+1, num_qubits)
        x: Input features, length num_qubits

    Returns:
        Expectation value of Z x Z x Z x Z (scalar in [-1, 1])
    """
    # === Feature Map: ZFeatureMap(feature_dimension=4, reps=1) ===
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(2.0 * x[i], wires=i)

    # === Ansatz: RealAmplitudes(num_qubits=4, reps=3, entanglement='circular') ===
    # Initial RY layer (no entanglement before first layer)
    for i in range(NUM_QUBITS):
        qml.RY(weights[0, i], wires=i)

    # reps repetitions of [CNOT circular ring + RY layer]
    for rep in range(1, ANSATZ_REPS + 1):
        # Circular CNOT entanglement: (0,1), (1,2), (2,3), (3,0)
        for i in range(NUM_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
        # RY rotation layer
        for i in range(NUM_QUBITS):
            qml.RY(weights[rep, i], wires=i)

    # === Measurement: Z^{x4} ===
    return qml.expval(
        qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
    )


def batch_predict(weights_flat, X):
    """Run circuit on a batch of inputs.

    Args:
        weights_flat: Flattened parameter array, length num_parameters
        X: Input array, shape (n_samples, num_qubits)

    Returns:
        Predictions array, shape (n_samples,)
    """
    weights_2d = weights_flat.reshape(WEIGHT_SHAPE)
    return pnp.array([circuit(weights_2d, x) for x in X])


def cost_fn(weights_flat, X, y):
    """MSE cost function for training.

    Args:
        weights_flat: Flattened parameter array, length num_parameters
        X: Scaled training features
        y: Scaled training targets

    Returns:
        Mean squared error (scalar)
    """
    preds = batch_predict(weights_flat, X)
    return pnp.mean((preds - y) ** 2)


# --- Training utilities ---

def make_callback(loss_history, X_train, y_train):
    """Create a scipy callback that records loss at each L-BFGS-B iteration.

    Args:
        loss_history: List to append loss values to
        X_train: Scaled training features (for re-evaluating cost)
        y_train: Scaled training targets

    Returns:
        Callback function compatible with scipy.optimize.minimize
    """
    def callback(xk):
        loss_val = float(cost_fn(
            pnp.array(xk, requires_grad=False), X_train, y_train
        ))
        loss_history.append({
            "iteration": len(loss_history) + 1,
            "loss": loss_val,
        })

    return callback


def train_model(X_train_scaled, y_train_scaled, maxiter, seed,
                loss_history=None):
    """Train the quantum model using scipy L-BFGS-B.

    Args:
        X_train_scaled: Scaled training features
        y_train_scaled: Scaled training targets
        maxiter: Maximum optimizer iterations
        seed: Random seed for weight initialization
        loss_history: Optional list for convergence tracking

    Returns:
        Optimal weights as numpy array, shape WEIGHT_SHAPE
    """
    # Deterministic weight initialization
    rng = np.random.RandomState(seed)
    init_weights = rng.uniform(-np.pi, np.pi, size=NUM_QUBITS * (ANSATZ_REPS + 1))

    # Wrap data as non-differentiable PennyLane arrays
    X_pnp = pnp.array(X_train_scaled, requires_grad=False)
    y_pnp = pnp.array(y_train_scaled, requires_grad=False)

    # Gradient function
    grad_fn = qml.grad(cost_fn, argnum=0)

    # Build callback
    cb = None
    if loss_history is not None:
        cb = make_callback(loss_history, X_pnp, y_pnp)

    # Optimize with scipy L-BFGS-B
    result = minimize(
        fun=lambda w: float(cost_fn(
            pnp.array(w, requires_grad=True), X_pnp, y_pnp
        )),
        x0=init_weights,
        method="L-BFGS-B",
        jac=lambda w: np.array(grad_fn(
            pnp.array(w, requires_grad=True), X_pnp, y_pnp
        )),
        options={"maxiter": maxiter},
        callback=cb,
    )

    return result.x.reshape(WEIGHT_SHAPE)


def predict_original_scale(weights, X_scaled, scaler_y):
    """Predict on scaled features and inverse-transform to original scale.

    Args:
        weights: Trained weight array, shape WEIGHT_SHAPE
        X_scaled: Scaled input features
        scaler_y: Fitted MinMaxScaler for targets

    Returns:
        Predictions on original scale, clipped to >= 0
    """
    y_pred_scaled = np.array([
        float(circuit(weights, x))
        for x in X_scaled
    ])
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()
    y_pred = np.clip(y_pred, 0, None)  # Wind power cannot be negative
    return y_pred


# --- Experiment runners ---

def scale_data(X_train, y_train, X_val, y_val):
    """Scale features and targets using MinMaxScaler.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets

    Returns:
        Tuple of (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                  scaler_X, scaler_y)
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(
        np.array(y_train).reshape(-1, 1)
    ).ravel()

    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(
        np.array(y_val).reshape(-1, 1)
    ).ravel()

    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            scaler_X, scaler_y)


def run_cross_validation(X_train, y_train, seed):
    """Run cross-validation and return aggregated metrics.

    Args:
        X_train: Training features DataFrame
        y_train: Training targets Series
        seed: Random seed for CV split

    Returns:
        Dictionary with CV metrics (mean and std for r2, rmse, mae)
    """
    kfold = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)

    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        print_flush(f"  CV Fold {fold_idx}/{config.CV_FOLDS}...")

        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        X_tr_sc, y_tr_sc, X_val_sc, _, scaler_X, scaler_y = scale_data(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold
        )

        fold_start = time.time()
        weights = train_model(X_tr_sc, y_tr_sc,
                              maxiter=config.QNN_CONFIG["maxiter"],
                              seed=seed + fold_idx)
        fold_time = time.time() - fold_start

        y_val_pred = predict_original_scale(weights, X_val_sc, scaler_y)
        metrics = compute_metrics(y_val_fold, y_val_pred)

        r2_scores.append(metrics["r2"])
        rmse_scores.append(metrics["rmse"])
        mae_scores.append(metrics["mae"])

        print_flush(f"    R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f}, "
                    f"Time={fold_time:.1f}s")

    return {
        "cv_mean_r2": float(np.mean(r2_scores)),
        "cv_std_r2": float(np.std(r2_scores)),
        "cv_mean_rmse": float(np.mean(rmse_scores)),
        "cv_std_rmse": float(np.std(rmse_scores)),
        "cv_mean_mae": float(np.mean(mae_scores)),
        "cv_std_mae": float(np.std(mae_scores)),
    }


def run_experiment(train_size, seed):
    """Run a single QNN experiment.

    Args:
        train_size: Number of training samples
        seed: Random seed
    """
    print_flush(f"\n{'='*60}")
    print_flush(f"Running QNN experiment (PennyLane)")
    print_flush(f"  Train size: {train_size}")
    print_flush(f"  Seed: {seed}")
    print_flush(f"  Config: L_BFGS_B, reps={config.QNN_CONFIG['ansatz_reps']}, "
                f"maxiter={config.QNN_CONFIG['maxiter']}")
    print_flush(f"{'='*60}")

    start_time = time.time()

    # Load data
    X_train, y_train, X_test, y_test = load_data(seed, train_size)
    print_flush(f"Loaded data: {len(X_train)} train, {len(X_test)} test samples")

    # === Phase 1: Cross-validation ===
    print_flush(f"\nPhase 1: {config.CV_FOLDS}-fold Cross-Validation...")
    cv_metrics = run_cross_validation(X_train, y_train, seed)
    print_flush(f"CV Results: R²={cv_metrics['cv_mean_r2']:.4f} "
                f"+/- {cv_metrics['cv_std_r2']:.4f}")

    # === Phase 2: Final training with convergence tracking ===
    print_flush("\nPhase 2: Final training on full dataset...")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(
        np.array(y_train).reshape(-1, 1)
    ).ravel()
    X_test_scaled = scaler_X.transform(X_test)

    # Train with convergence tracking
    loss_history = []
    train_start = time.time()
    optimal_weights = train_model(
        X_train_scaled, y_train_scaled,
        maxiter=config.QNN_CONFIG["maxiter"],
        seed=seed,
        loss_history=loss_history,
    )
    train_time = time.time() - train_start
    print_flush(f"Training completed in {train_time:.1f}s")

    # Evaluate on training set
    y_train_pred = predict_original_scale(optimal_weights, X_train_scaled, scaler_y)
    train_metrics = compute_metrics(y_train, y_train_pred)
    print_flush(f"Train R²: {train_metrics['r2']:.4f}")

    # Evaluate on test set
    y_test_pred = predict_original_scale(optimal_weights, X_test_scaled, scaler_y)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print_flush(f"Test R²: {test_metrics['r2']:.4f}")

    wall_time = time.time() - start_time
    print_flush(f"Total wall time: {wall_time:.1f} seconds")

    # Prepare output directory
    output_dir = Path(config.RESULTS_DIR) / "qnn" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_dict = {
        "model": "qnn",
        "train_size": train_size,
        "seed": seed,
        "train_r2": train_metrics["r2"],
        "train_mse": train_metrics["mse"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        **cv_metrics,
        "config": config.QNN_CONFIG,
        "wall_time_seconds": wall_time,
        "training_time_seconds": train_time,
    }

    metrics_path = output_dir / f"exp_{train_size}_metrics.json"
    save_metrics(metrics_dict, str(metrics_path))
    print_flush(f"Saved metrics: {metrics_path}")

    # Save predictions on ORIGINAL scale
    predictions_path = output_dir / f"exp_{train_size}_predictions.csv"
    save_predictions(y_test, y_test_pred, str(predictions_path))
    print_flush(f"Saved predictions: {predictions_path}")

    # Save convergence data
    if loss_history:
        convergence_path = output_dir / f"exp_{train_size}_convergence.csv"
        save_convergence(loss_history, str(convergence_path))
        print_flush(f"Saved convergence: {convergence_path} "
                    f"({len(loss_history)} points)")
    else:
        print_flush("WARNING: No convergence data captured from optimizer")

    print_flush(f"\n{'='*60}")
    print_flush(f"Experiment completed successfully!")
    print_flush(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run QNN experiment with PennyLane (fixed configuration)"
    )
    parser.add_argument(
        "--train_size", type=int, choices=config.TRAIN_SIZES,
        help=f"Training set size: {config.TRAIN_SIZES}"
    )
    parser.add_argument(
        "--seed", type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run ALL sizes x seeds"
    )

    args = parser.parse_args()

    if args.all:
        total = len(config.TRAIN_SIZES) * len(config.SEEDS)
        done = 0
        failed = 0
        for seed in config.SEEDS:
            for size in config.TRAIN_SIZES:
                done += 1
                print(f"\n>>> [{done}/{total}] QNN size={size} seed={seed}",
                      flush=True)
                try:
                    run_experiment(size, seed)
                except Exception as e:
                    failed += 1
                    print(f"FAILED: {e}", flush=True)
        print(f"\nAll QNN done: {done - failed}/{total} succeeded, "
              f"{failed} failed", flush=True)
    elif args.seed is not None and args.train_size is None:
        # --seed-only mode: run all train sizes for one seed (SLURM array)
        total = len(config.TRAIN_SIZES)
        done = 0
        failed = 0
        for size in config.TRAIN_SIZES:
            done += 1
            print(f"\n>>> [{done}/{total}] QNN size={size} seed={args.seed}",
                  flush=True)
            try:
                run_experiment(size, args.seed)
            except Exception as e:
                failed += 1
                print(f"FAILED: {e}", flush=True)
        print(f"\nSeed {args.seed} done: {done - failed}/{total} succeeded, "
              f"{failed} failed", flush=True)
    else:
        if not all([args.train_size, args.seed is not None]):
            parser.error(
                "Provide --train_size --seed  OR  --seed (all sizes)  OR  --all"
            )
        run_experiment(args.train_size, args.seed)


if __name__ == "__main__":
    main()
