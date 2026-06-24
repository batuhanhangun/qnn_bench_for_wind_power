"""Generate seed-dependent train/test splits from the full dataset.

Uses the same seeds and logic as the main benchmark, ensuring identical data
splits and therefore identical trained weights.

Usage:
    python -m src.generate_splits --seed 42
    python -m src.generate_splits --all
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_dataset(path: str):
    """Load dataset, handling semicolon-separated data within Excel cells."""
    from io import StringIO

    df = pd.read_excel(path)

    if len(df.columns) == 1 and ";" in df.columns[0]:
        col_name = df.columns[0]
        text = col_name + "\n" + df[col_name].str.cat(sep="\n")
        df = pd.read_csv(StringIO(text), sep=";")

    return df


def generate_splits_for_seed(seed: int, verbose: bool = True):
    """Generate train/test splits for a specific seed."""
    if verbose:
        print(f"Generating splits for seed {seed}...")

    df = load_dataset(config.DATASET_PATH)

    if verbose:
        print(f"  Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    output_dir = Path(f"data/seed_{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = df_shuffled.iloc[:config.TEST_SIZE]
    test_df.to_csv(output_dir / "test_set.csv", index=False, sep=config.CSV_DELIMITER)
    if verbose:
        print(f"  Saved test set: {len(test_df)} rows")

    remaining_df = df_shuffled.iloc[config.TEST_SIZE:]

    for train_size in config.TRAIN_SIZES:
        if train_size > len(remaining_df):
            print(f"  WARNING: train_size {train_size} exceeds available rows")
            train_df = remaining_df.copy()
        else:
            train_df = remaining_df.iloc[:train_size]
        train_df.to_csv(
            output_dir / f"train_{train_size}.csv",
            index=False, sep=config.CSV_DELIMITER
        )
        if verbose:
            print(f"  Saved train_{train_size}.csv: {len(train_df)} rows")

    if verbose:
        print(f"  Done for seed {seed}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate seed-dependent train/test splits"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all", action="store_true",
                        help="Generate splits for all seeds in config.SEEDS")
    args = parser.parse_args()

    if args.all:
        print(f"Generating splits for {len(config.SEEDS)} seeds: {config.SEEDS}")
        for seed in config.SEEDS:
            generate_splits_for_seed(seed)
        print("All splits generated.")
    elif args.seed is not None:
        generate_splits_for_seed(args.seed)
    else:
        parser.error("Provide --seed <value> or --all")


if __name__ == "__main__":
    main()
