"""QNN hardware robustness evaluation under depolarizing noise.

Strategy
--------
1. Train the QNN noise-free using lightning.qubit + L-BFGS-B (identical to
   the main benchmark, same seeds → same weights).
2. Save the optimal weights to disk for reference.
3. Evaluate the frozen weights on the test set using default.mixed with
   gate-level DepolarizingChannel at six error rates p in config.NOISE_LEVELS.
4. p=0.0 uses lightning.qubit (exact statevector baseline) and should match
   the main benchmark's test metrics — serves as a built-in consistency check.

Noise model
-----------
Single-qubit depolarizing channel after every gate:
  - After each H and RZ gate  (feature map)
  - After each RY gate        (ansatz)
  - After BOTH qubits of each CNOT (control and target independently)

Usage:
    python -m src.run_noise_eval --train_size 800 --seed 42
    python -m src.run_noise_eval --seed 42          # all train sizes
    python -m src.run_noise_eval --all              # all seeds x all sizes
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import compute_metrics, load_data, print_flush, save_metrics

# --- Circuit constants (must match the main benchmark exactly) ---

NUM_QUBITS = config.QNN_CONFIG["num_qubits"]
ANSATZ_REPS = config.QNN_CONFIG["ansatz_reps"]
WEIGHT_SHAPE = (ANSATZ_REPS + 1, NUM_QUBITS)  # (4, 4) = 16 params

# --- Noise-free training circuit (lightning.qubit, adjoint diff) ---

_dev_train = qml.device(config.QNN_CONFIG["device_train"], wires=NUM_QUBITS)


@qml.qnode(_dev_train, interface="autograd",
           diff_method=config.QNN_CONFIG["diff_method"])
def _circuit_train(weights, x):
    """Noise-free circuit used for training (identical to the main benchmark)."""
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(2.0 * x[i], wires=i)
    for i in range(NUM_QUBITS):
        qml.RY(weights[0, i], wires=i)
    for rep in range(1, ANSATZ_REPS + 1):
        for i in range(NUM_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
        for i in range(NUM_QUBITS):
            qml.RY(weights[rep, i], wires=i)
    return qml.expval(
        qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
    )


def _cost_fn(weights_flat, X, y):
    weights_2d = weights_flat.reshape(WEIGHT_SHAPE)
    preds = pnp.array([_circuit_train(weights_2d, x) for x in X])
    return pnp.mean((preds - y) ** 2)


# --- Training ---

def train_qnn(X_train_scaled, y_train_scaled, seed: int) -> np.ndarray:
    """Train the QNN noise-free, identical to the main benchmark.

    Args:
        X_train_scaled: Scaled training features
        y_train_scaled: Scaled training targets
        seed: Random seed for deterministic weight initialisation

    Returns:
        Optimal weights, shape WEIGHT_SHAPE
    """
    rng = np.random.RandomState(seed)
    init_weights = rng.uniform(-np.pi, np.pi,
                               size=NUM_QUBITS * (ANSATZ_REPS + 1))

    X_pnp = pnp.array(X_train_scaled, requires_grad=False)
    y_pnp = pnp.array(y_train_scaled, requires_grad=False)

    grad_fn = qml.grad(_cost_fn, argnum=0)

    result = minimize(
        fun=lambda w: float(_cost_fn(
            pnp.array(w, requires_grad=True), X_pnp, y_pnp
        )),
        x0=init_weights,
        method="L-BFGS-B",
        jac=lambda w: np.array(grad_fn(
            pnp.array(w, requires_grad=True), X_pnp, y_pnp
        )),
        options={"maxiter": config.QNN_CONFIG["maxiter"]},
    )

    return result.x.reshape(WEIGHT_SHAPE)


# --- Noisy evaluation circuits ---

def make_noisy_circuit(p: float):
    """Return a QNode for the given depolarizing error rate.

    p=0.0  → lightning.qubit statevector (exact noise-free baseline).
    p>0.0  → default.mixed density matrix with DepolarizingChannel(p)
              inserted after every gate (feature map + ansatz).

    Args:
        p: Per-gate depolarizing error probability in [0, 1]

    Returns:
        Callable QNode: circuit(weights_2d, x) → scalar expectation value
    """
    if p == 0.0:
        dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

        @qml.qnode(dev, interface="numpy")
        def _noisefree(weights, x):
            for i in range(NUM_QUBITS):
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * x[i], wires=i)
            for i in range(NUM_QUBITS):
                qml.RY(weights[0, i], wires=i)
            for rep in range(1, ANSATZ_REPS + 1):
                for i in range(NUM_QUBITS):
                    qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
                for i in range(NUM_QUBITS):
                    qml.RY(weights[rep, i], wires=i)
            return qml.expval(
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
            )

        return _noisefree

    else:
        dev = qml.device("default.mixed", wires=NUM_QUBITS)

        @qml.qnode(dev, interface="numpy")
        def _noisy(weights, x):
            # Feature map: H + depol, RZ + depol
            for i in range(NUM_QUBITS):
                qml.Hadamard(wires=i)
                qml.DepolarizingChannel(p, wires=i)
                qml.RZ(2.0 * x[i], wires=i)
                qml.DepolarizingChannel(p, wires=i)

            # Initial RY layer + depol
            for i in range(NUM_QUBITS):
                qml.RY(weights[0, i], wires=i)
                qml.DepolarizingChannel(p, wires=i)

            # reps: CNOT ring (depol on both qubits) + RY layer + depol
            for rep in range(1, ANSATZ_REPS + 1):
                for i in range(NUM_QUBITS):
                    qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
                    qml.DepolarizingChannel(p, wires=i)
                    qml.DepolarizingChannel(p, wires=(i + 1) % NUM_QUBITS)
                for i in range(NUM_QUBITS):
                    qml.RY(weights[rep, i], wires=i)
                    qml.DepolarizingChannel(p, wires=i)

            return qml.expval(
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
            )

        return _noisy


def evaluate_under_noise(weights_2d, X_test_scaled, scaler_y,
                         y_test) -> list:
    """Evaluate frozen weights at every noise level in config.NOISE_LEVELS.

    Args:
        weights_2d: Trained weights, shape WEIGHT_SHAPE
        X_test_scaled: Scaled test features
        scaler_y: Fitted MinMaxScaler for targets
        y_test: Ground-truth test targets (original scale)

    Returns:
        List of dicts, one per noise level, each containing noise_p + metrics
    """
    results = []

    for p in config.NOISE_LEVELS:
        label = "noise-free" if p == 0.0 else f"p={p}"
        t0 = time.time()
        try:
            circuit_fn = make_noisy_circuit(p)

            y_pred_scaled = np.array([
                float(circuit_fn(weights_2d, x)) for x in X_test_scaled
            ])

            # Guard against NaN/Inf from density matrix simulator
            n_bad = int(np.sum(~np.isfinite(y_pred_scaled)))
            if n_bad > 0:
                print_flush(
                    f"    [{label}]  WARNING: {n_bad}/{len(y_pred_scaled)} "
                    f"non-finite predictions — replacing with 0"
                )
                y_pred_scaled = np.where(
                    np.isfinite(y_pred_scaled), y_pred_scaled, 0.0
                )

            y_pred = scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            y_pred = np.clip(y_pred, 0, None)

            metrics = compute_metrics(y_test, y_pred)
            elapsed = time.time() - t0

            print_flush(
                f"    [{label}]  R²={metrics['r2']:.4f}  "
                f"RMSE={metrics['rmse']:.2f}  t={elapsed:.1f}s"
            )

            results.append({
                "noise_p": p,
                "test_r2": metrics["r2"],
                "test_mse": metrics["mse"],
                "test_rmse": metrics["rmse"],
                "test_mae": metrics["mae"],
            })

        except Exception as e:
            elapsed = time.time() - t0
            print_flush(
                f"    [{label}]  FAILED after {elapsed:.1f}s: {e}"
            )
            # Record failure so the JSON reflects every noise level attempted
            results.append({
                "noise_p": p,
                "test_r2": None,
                "test_mse": None,
                "test_rmse": None,
                "test_mae": None,
                "error": str(e),
            })

    return results


# --- Experiment runner ---

def run_experiment(train_size: int, seed: int):
    """Run noise evaluation for one (train_size, seed) pair.

    Steps:
      1. Load data and scale.
      2. Train QNN noise-free (deterministic → same weights as the main benchmark).
      3. Save weights.
      4. Evaluate at all noise levels.
      5. Save results.

    Args:
        train_size: Number of training samples
        seed: Random seed
    """
    print_flush(f"\n{'='*60}")
    print_flush(f"Noise eval: train_size={train_size}  seed={seed}")
    print_flush(f"Noise levels: {config.NOISE_LEVELS}")
    print_flush(f"{'='*60}")

    start = time.time()

    # Load data
    X_train, y_train, X_test, y_test = load_data(seed, train_size)
    print_flush(f"Data: {len(X_train)} train, {len(X_test)} test")

    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(
        np.array(y_train).reshape(-1, 1)
    ).ravel()
    X_test_scaled = scaler_X.transform(X_test)

    # Train noise-free
    output_dir = Path(config.RESULTS_DIR) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / f"exp_{train_size}_weights.npy"

    if weights_path.exists():
        try:
            weights_2d = np.load(str(weights_path)).reshape(WEIGHT_SHAPE)
            print_flush(f"Loaded saved weights: {weights_path}")
        except Exception as e:
            print_flush(f"WARNING: Could not load weights ({e}). Retraining...")
            weights_2d = None
    else:
        weights_2d = None

    if weights_2d is None:
        print_flush("Training noise-free QNN (deterministic)...")
        t_train = time.time()
        weights_2d = train_qnn(X_train_scaled, y_train_scaled, seed)
        print_flush(f"Training done in {time.time() - t_train:.1f}s")
        np.save(str(weights_path), weights_2d)
        print_flush(f"Saved weights: {weights_path}")

    # Evaluate under all noise levels
    print_flush("\nEvaluating under noise:")
    noise_results = evaluate_under_noise(
        weights_2d, X_test_scaled, scaler_y, y_test
    )

    if not noise_results:
        print_flush("ERROR: No noise-level results produced. Experiment failed.")
        return

    # Attach metadata
    for entry in noise_results:
        entry.update({
            "model": "qnn",
            "train_size": train_size,
            "seed": seed,
        })

    n_failed = sum(1 for e in noise_results if e.get("error"))
    if n_failed:
        print_flush(
            f"WARNING: {n_failed}/{len(noise_results)} noise levels failed. "
            f"Results saved with None values for failed levels."
        )

    # Save
    metrics_path = output_dir / f"exp_{train_size}_noise_metrics.json"
    save_metrics(noise_results, str(metrics_path))
    print_flush(f"Saved: {metrics_path}")
    print_flush(f"Total time: {time.time() - start:.1f}s")


# --- Entry point ---

def main():
    parser = argparse.ArgumentParser(
        description="QNN noise robustness evaluation"
    )
    parser.add_argument("--train_size", type=int, choices=config.TRAIN_SIZES)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--all", action="store_true",
        help="Run all seeds × all train sizes"
    )
    args = parser.parse_args()

    if args.all:
        total = len(config.SEEDS) * len(config.TRAIN_SIZES)
        done = failed = 0
        for seed in config.SEEDS:
            for size in config.TRAIN_SIZES:
                done += 1
                print_flush(
                    f"\n>>> [{done}/{total}] noise eval  size={size}  seed={seed}"
                )
                try:
                    run_experiment(size, seed)
                except Exception as e:
                    failed += 1
                    print_flush(f"FAILED: {e}")
        print_flush(
            f"\nAll done: {done - failed}/{total} succeeded, {failed} failed"
        )

    elif args.seed is not None and args.train_size is None:
        # --seed only: run all train sizes for one seed (SLURM array mode)
        total = len(config.TRAIN_SIZES)
        done = failed = 0
        for size in config.TRAIN_SIZES:
            done += 1
            print_flush(
                f"\n>>> [{done}/{total}] noise eval  size={size}  seed={args.seed}"
            )
            try:
                run_experiment(size, args.seed)
            except Exception as e:
                failed += 1
                print_flush(f"FAILED: {e}")
        print_flush(
            f"\nSeed {args.seed} done: {done - failed}/{total} succeeded"
        )

    else:
        if args.train_size is None or args.seed is None:
            parser.error(
                "Provide --train_size --seed  OR  --seed (all sizes)  OR  --all"
            )
        try:
            run_experiment(args.train_size, args.seed)
        except Exception as e:
            print_flush(f"FAILED: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
