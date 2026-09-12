from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

INPUT_PATH = Path("outputs/statistical_analysis/statistical_tests.csv")
OUTPUT_PATH = Path("outputs/statistical_analysis/paired_difference_ci.csv")


def main():
    df = pd.read_csv(INPUT_PATH)
    results = []
    for _, row in df.iterrows():
        mean_diff = float(row["mean_difference"])
        sd_diff = float(row["sd_difference"])
        n = int(row["n_seeds"])
        if n > 1 and sd_diff > 0:
            se = sd_diff / np.sqrt(n)
            t_critical = stats.t.ppf(0.975, df=n - 1)
            margin = t_critical * se
            ci_low = mean_diff - margin
            ci_high = mean_diff + margin
        else:
            ci_low = np.nan
            ci_high = np.nan
        results.append({
            "dataset": row["dataset"],
            "experiment": row["experiment"],
            "adaptive_method": row["adaptive_method"],
            "baseline_method": row["baseline_method"],
            "metric": row["metric"],
            "n_seeds": n,
            "mean_difference": mean_diff,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "paired_t": row["paired_t"],
            "paired_t_p": row["paired_t_p"],
            "wilcoxon_p": row["wilcoxon_p"],
            "cohens_dz": row["cohens_dz"],
        })
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print()
    print("=" * 100)
    print("PAIRED DIFFERENCE 95% CI")
    print("=" * 100)
    for _, row in result_df.iterrows():
        print(
            f"{row['dataset']:10s} | "
            f"{row['adaptive_method']} vs {row['baseline_method']} | "
            f"{row['metric']:25s} | "
            f"Diff={row['mean_difference']:+.6f} | "
            f"95% CI=[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | "
            f"p={row['paired_t_p']:.6g}"
        )
    print()
    print("=" * 100)
    print("Saved:")
    print(OUTPUT_PATH)
    print("=" * 100)


if __name__ == "__main__":
    main()
