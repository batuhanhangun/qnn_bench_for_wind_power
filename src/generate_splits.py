"""Generate seed-dependent train/test splits from the full dataset.

This script creates deterministic data splits for each random seed,
enabling reproducible experiments across multiple runs.

Usage:
    python -m src.generate_splits --seed 42
    python -m src.generate_splits --all
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_dataset(path: str):
    """Load dataset, handling semicolon-separated data within Excel cells.

    Some Excel files have semicolon-separated data stored as text in cells.
    This function detects and handles that case.

    Args:
        path: Path to the dataset file

    Returns:
        DataFrame with properly parsed columns
    """
    from io import StringIO

    df = pd.read_excel(path)

    # Check if data is semicolon-separated within cells
    if len(df.columns) == 1 and ";" in df.columns[0]:
        col_name = df.columns[0]
        # Reconstruct as CSV text and re-parse
        text = col_name + "\n" + df[col_name].str.cat(sep="\n")
        df = pd.read_csv(StringIO(text), sep=";")

    return df


def generate_splits_for_seed(seed: int, verbose: bool = True):
    """Generate train/test splits for a specific seed.

    Args:
        seed: Random seed for shuffling
        verbose: Whether to print progress messages
    """
    if verbose:
        print(f"Generating splits for seed {seed}...")

    # Load full dataset
    df = load_dataset(config.DATASET_PATH)

    if verbose:
        print(f"  Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")

    # Shuffle with seed
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Create output directory
    output_dir = Path(f"data/seed_{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # First TEST_SIZE rows become test set
    test_df = df_shuffled.iloc[:config.TEST_SIZE]
    test_path = output_dir / "test_set.csv"
    test_df.to_csv(test_path, index=False, sep=config.CSV_DELIMITER)
    if verbose:
        print(f"  Saved test set: {test_path} ({len(test_df)} rows)")

    # Remaining rows are available for training
    remaining_df = df_shuffled.iloc[config.TEST_SIZE:]

    # Create training sets of different sizes
    for train_size in config.TRAIN_SIZES:
        if train_size > len(remaining_df):
            print(f"  WARNING: Requested train_size {train_size} exceeds available data ({len(remaining_df)} rows)")
            train_df = remaining_df.copy()
        else:
            train_df = remaining_df.iloc[:train_size]

        train_path = output_dir / f"train_{train_size}.csv"
        train_df.to_csv(train_path, index=False, sep=config.CSV_DELIMITER)
        if verbose:
            print(f"  Saved training set: {train_path} ({len(train_df)} rows)")

    if verbose:
        print(f"  Done generating splits for seed {seed}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate seed-dependent train/test splits"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Single seed to generate splits for"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate splits for all seeds in config.SEEDS"
    )

    args = parser.parse_args()

    if args.all:
        print(f"Generating splits for all {len(config.SEEDS)} seeds: {config.SEEDS}")
        for seed in config.SEEDS:
            generate_splits_for_seed(seed)
        print("\nAll splits generated successfully!")
    elif args.seed is not None:
        generate_splits_for_seed(args.seed)
    else:
        parser.error("Must specify either --seed <value> or --all")


if __name__ == "__main__":
    main()
