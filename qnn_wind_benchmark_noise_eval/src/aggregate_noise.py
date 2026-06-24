"""Aggregation and statistical analysis of noise robustness results.

Reporting:
- Mean ± std of test metrics across seeds at each noise level and train size
- 95 % confidence intervals (t-based and normal approximation)
- Wilcoxon signed-rank tests: noisy (p>0) vs noise-free (p=0) across seeds
- LaTeX table: noise levels as rows, train sizes as columns

Usage:
    python -m src.aggregate_noise
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# --- Data collection ---

def collect_noise_results() -> list:
    """Collect all noise evaluation results from the results directory.

    Returns:
        List of dicts, one per (seed, train_size, noise_p) combination
    """
    results = []
    results_dir = Path(config.RESULTS_DIR)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue

        seed = int(seed_dir.name.split("_")[1])

        for metrics_file in sorted(seed_dir.glob("exp_*_noise_metrics.json")):
            try:
                train_size = int(metrics_file.stem.split("_")[1])
                with open(metrics_file) as f:
                    entries = json.load(f)   # list of dicts

                for entry in entries:
                    if entry.get("error"):
                        print(
                            f"  Skipping failed entry: "
                            f"seed={seed}, size={train_size}, "
                            f"p={entry['noise_p']} — {entry['error']}"
                        )
                        continue
                    results.append({
                        "seed": seed,
                        "train_size": train_size,
                        "noise_p": entry["noise_p"],
                        "test_r2": entry["test_r2"],
                        "test_mse": entry["test_mse"],
                        "test_rmse": entry["test_rmse"],
                        "test_mae": entry["test_mae"],
                    })

            except Exception as e:
                print(f"Warning: Could not parse {metrics_file}: {e}")

    return results


def check_coverage(results: list):
    """Print a coverage report showing which combinations are complete."""
    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    all_seeds = set(config.SEEDS)

    print("\nData Coverage:")
    print("-" * 50)
    complete = True
    for size in config.TRAIN_SIZES:
        for p in config.NOISE_LEVELS:
            subset = df[(df["train_size"] == size) & (df["noise_p"] == p)]
            present = set(subset["seed"].tolist())
            missing = all_seeds - present
            if missing:
                complete = False
                print(f"  size={size}, p={p}: missing seeds {sorted(missing)}")

    if complete:
        print("  All experiments complete!")


# --- Aggregation helpers ---

def _ci_bounds(values, alpha: float = 0.05):
    """Return (t-based half-width, normal half-width) for a 1-D array.

    Args:
        values: Per-seed metric values
        alpha: Significance level (0.05 → 95 % CI)

    Returns:
        Tuple (t_hw, norm_hw)
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n)
    t_hw = float(stats.t.ppf(1 - alpha / 2, df=n - 1) * se)
    norm_hw = float(stats.norm.ppf(1 - alpha / 2) * se)
    return t_hw, norm_hw


def aggregate_noise_results(results: list) -> pd.DataFrame:
    """Aggregate noise results across seeds.

    Groups by (train_size, noise_p) and computes mean, std, and both
    95 % CI variants for every test metric.

    Args:
        results: Flat list of per-(seed, train_size, noise_p) dicts

    Returns:
        DataFrame with one row per (train_size, noise_p)
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    metrics = ["r2", "mse", "rmse", "mae"]
    aggregated = []

    for (train_size, noise_p), group in df.groupby(["train_size", "noise_p"]):
        row = {
            "train_size": train_size,
            "noise_p": noise_p,
            "n_seeds": len(group),
        }
        for m in metrics:
            col = f"test_{m}"
            vals = group[col].dropna()
            mean_ = float(vals.mean())
            std_ = float(vals.std(ddof=1))
            t_hw, norm_hw = _ci_bounds(vals)
            row[f"mean_{col}"] = mean_
            row[f"std_{col}"] = std_
            row[f"ci95t_lo_{col}"] = mean_ - t_hw
            row[f"ci95t_hi_{col}"] = mean_ + t_hw
            row[f"ci95n_lo_{col}"] = mean_ - norm_hw
            row[f"ci95n_hi_{col}"] = mean_ + norm_hw
        aggregated.append(row)

    return pd.DataFrame(aggregated).sort_values(
        ["train_size", "noise_p"]
    ).reset_index(drop=True)


# --- Statistical tests ---

def run_noise_wilcoxon_tests(results: list) -> list:
    """Wilcoxon signed-rank tests: each noise level vs noise-free (p=0).

    For every (train_size, p>0) pair, tests whether test R² differs
    significantly from the p=0 baseline across seeds.

    Args:
        results: Flat list of per-(seed, train_size, noise_p) dicts

    Returns:
        List of test result dicts
    """
    df = pd.DataFrame(results)
    test_results = []

    for train_size in sorted(df["train_size"].unique()):
        df_size = df[df["train_size"] == train_size]

        baseline = (
            df_size[df_size["noise_p"] == 0.0]
            .set_index("seed")["test_r2"]
        )

        for p in sorted(df_size["noise_p"].unique()):
            if p == 0.0:
                continue

            noisy = (
                df_size[df_size["noise_p"] == p]
                .set_index("seed")["test_r2"]
            )
            common = baseline.index.intersection(noisy.index)

            entry = {
                "train_size": train_size,
                "noise_p": p,
                "n_pairs": len(common),
                "baseline_mean_r2": float(baseline.loc[common].mean()),
                "noisy_mean_r2": float(noisy.loc[common].mean()),
            }

            if len(common) < 5:
                entry["statistic"] = None
                entry["p_value"] = None
                entry["note"] = f"Insufficient pairs ({len(common)} < 5)"
            else:
                try:
                    stat, pval = stats.wilcoxon(
                        baseline.loc[common].values,
                        noisy.loc[common].values,
                        alternative="two-sided",
                    )
                    entry["statistic"] = float(stat)
                    entry["p_value"] = float(pval)
                    entry["significant_0.05"] = pval < 0.05
                    entry["significant_0.01"] = pval < 0.01
                except Exception as e:
                    entry["statistic"] = None
                    entry["p_value"] = None
                    entry["note"] = f"Test failed: {e}"

            test_results.append(entry)

    return test_results


# --- LaTeX output ---

def generate_noise_latex_table(agg_df: pd.DataFrame) -> str:
    """Generate LaTeX table for noise robustness results.

    Layout: noise level (p) as rows, training size as columns.
    Cells show mean test R² ± std across seeds.
    Bold the p=0 row as the noise-free baseline.

    Args:
        agg_df: Aggregated DataFrame from aggregate_noise_results

    Returns:
        LaTeX table string (single table environment)
    """
    if agg_df.empty:
        return "% No noise evaluation data available"

    train_sizes = sorted(agg_df["train_size"].unique())
    noise_levels = sorted(agg_df["noise_p"].unique())
    n_seeds = int(agg_df["n_seeds"].max())

    col_spec = "c" + "c" * len(train_sizes)
    size_header = " & ".join(f"$N={s}$" for s in train_sizes)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{QNN test $R^2$ (mean $\\pm$ std, $n="
        + str(n_seeds)
        + "$ seeds) under gate-level depolarizing noise. "
        "Bold row is the noise-free baseline.}",
        "\\label{tab:noise_robustness}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Noise $p$ & {size_header} \\\\",
        "\\midrule",
    ]

    for p in noise_levels:
        label = "\\textbf{0 (ideal)}" if p == 0.0 else f"{p}"
        cells = [label]

        for s in train_sizes:
            row = agg_df[
                (agg_df["train_size"] == s) & (agg_df["noise_p"] == p)
            ]
            if len(row) > 0:
                mean_ = row["mean_test_r2"].values[0]
                std_ = row["std_test_r2"].values[0]
                cell = f"{mean_:.4f} $\\pm$ {std_:.4f}"
                if p == 0.0:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            else:
                cells.append("--")

        line = " & ".join(cells) + " \\\\"
        lines.append(line)

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    return "\n".join(lines)


# --- Saving ---

def save_noise_results(agg_df: pd.DataFrame, test_results: list,
                       latex: str):
    """Save all aggregated noise results to the aggregated directory.

    Outputs:
        noise_summary.csv              - wide-format aggregated table
        noise_confidence_intervals.csv - long-format CI table
        noise_latex_table.tex          - ready-to-paste LaTeX
        noise_statistical_tests.txt    - human-readable Wilcoxon results
        noise_statistical_tests.json   - machine-readable Wilcoxon results
    """
    output_dir = Path(config.AGGREGATED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    csv_path = output_dir / "noise_summary.csv"
    agg_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Long-format CI table
    ci_rows = []
    for _, row in agg_df.iterrows():
        for m in ["r2", "mse", "rmse", "mae"]:
            col = f"test_{m}"
            ci_rows.append({
                "train_size": row["train_size"],
                "noise_p": row["noise_p"],
                "metric": m,
                "n_seeds": row["n_seeds"],
                "mean": row.get(f"mean_{col}"),
                "std": row.get(f"std_{col}"),
                "ci95_t_lo": row.get(f"ci95t_lo_{col}"),
                "ci95_t_hi": row.get(f"ci95t_hi_{col}"),
                "ci95_norm_lo": row.get(f"ci95n_lo_{col}"),
                "ci95_norm_hi": row.get(f"ci95n_hi_{col}"),
            })
    ci_path = output_dir / "noise_confidence_intervals.csv"
    pd.DataFrame(ci_rows).to_csv(ci_path, index=False)
    print(f"Saved: {ci_path}")

    # LaTeX
    tex_path = output_dir / "noise_latex_table.tex"
    tex_path.write_text(latex)
    print(f"Saved: {tex_path}")

    # Wilcoxon tests — TXT
    txt_path = output_dir / "noise_statistical_tests.txt"
    n_seeds = len(config.SEEDS)
    with open(txt_path, "w") as f:
        f.write("Wilcoxon Signed-Rank Tests: Noisy vs Noise-Free (p=0)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"NOTE: n={n_seeds} paired samples. Min two-sided p-value = "
                f"{2 / (2**n_seeds):.4f}\n\n")

        current_size = None
        for t in test_results:
            if t["train_size"] != current_size:
                current_size = t["train_size"]
                f.write("=" * 60 + "\n")
                f.write(f"Train size = {current_size}\n")
                f.write("=" * 60 + "\n\n")

            f.write(f"Noise p = {t['noise_p']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Pairs: {t['n_pairs']}\n")
            f.write(f"  Baseline (p=0) mean R²: {t['baseline_mean_r2']:.4f}\n")
            f.write(f"  Noisy mean R²:          {t['noisy_mean_r2']:.4f}\n")

            if t.get("p_value") is not None:
                f.write(f"  Statistic: {t['statistic']:.4f}\n")
                f.write(f"  P-value:   {t['p_value']:.6f}\n")
                f.write(f"  Significant at 0.05: {t['significant_0.05']}\n")
                f.write(f"  Significant at 0.01: {t['significant_0.01']}\n")
            else:
                f.write(f"  Note: {t.get('note', 'Not performed')}\n")
            f.write("\n")

    print(f"Saved: {txt_path}")

    # Wilcoxon tests — JSON
    json_path = output_dir / "noise_statistical_tests.json"
    import json as _json
    json_path.write_text(_json.dumps(test_results, indent=2))
    print(f"Saved: {json_path}")


# --- Console summary ---

def print_noise_summary(agg_df: pd.DataFrame, test_results: list):
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("NOISE ROBUSTNESS SUMMARY")
    print("=" * 70)

    if agg_df.empty:
        print("No results to summarise.")
        return

    for train_size in sorted(agg_df["train_size"].unique()):
        print(f"\nTrain size: {train_size}")
        print(f"  {'p':>8}  {'R² mean':>10}  {'± std':>8}  "
              f"{'RMSE mean':>10}  {'± std':>8}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")

        subset = agg_df[agg_df["train_size"] == train_size].sort_values("noise_p")
        for _, row in subset.iterrows():
            p_label = "0 (ideal)" if row["noise_p"] == 0.0 else str(row["noise_p"])
            print(
                f"  {p_label:>8}  "
                f"{row['mean_test_r2']:>10.4f}  "
                f"{row['std_test_r2']:>8.4f}  "
                f"{row['mean_test_rmse']:>10.2f}  "
                f"{row['std_test_rmse']:>8.2f}"
            )

    print("\n" + "-" * 70)
    print("Wilcoxon tests (noisy vs p=0 baseline):")
    current_size = None
    for t in test_results:
        if t["train_size"] != current_size:
            current_size = t["train_size"]
            print(f"\n  Train size = {current_size}")
        if t.get("p_value") is not None:
            sig = "*" if t["significant_0.05"] else ""
            sig += "*" if t["significant_0.01"] else ""
            print(f"    p={t['noise_p']}: p-val={t['p_value']:.4f} {sig}")
        else:
            print(f"    p={t['noise_p']}: {t.get('note', 'N/A')}")

    print("\n  * p<0.05  ** p<0.01")
    print("  95% CIs saved to noise_confidence_intervals.csv")
    print("=" * 70)


# --- Main ---

def main():
    print("=" * 70)
    print("NOISE RESULTS AGGREGATION")
    print("=" * 70)

    print(f"\nCollecting results from {config.RESULTS_DIR} ...")
    results = collect_noise_results()
    print(f"Found {len(results)} result entries "
          f"({len(set(r['seed'] for r in results))} seeds × "
          f"{len(set(r['train_size'] for r in results))} sizes × "
          f"{len(set(r['noise_p'] for r in results))} noise levels)")

    check_coverage(results)

    if not results:
        print("\nNo results to aggregate. Run experiments first:")
        print("  1. python -m src.generate_splits --all")
        print("  2. bash run_noise_pipeline.sh")
        return

    print("\nAggregating across seeds...")
    agg_df = aggregate_noise_results(results)

    print("Running Wilcoxon tests (noisy vs baseline)...")
    test_results = run_noise_wilcoxon_tests(results)

    print("Generating LaTeX table...")
    latex = generate_noise_latex_table(agg_df)

    print("Saving results...")
    save_noise_results(agg_df, test_results, latex)

    print_noise_summary(agg_df, test_results)


if __name__ == "__main__":
    main()
