import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [42, 123, 2024, 3407, 7777]
OUTPUT_DIR = Path("outputs/statistical_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def mean_sd_ci(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan

    n = len(values)
    mean = np.mean(values)

    if n == 1:
        return mean, 0.0, mean, mean

    sd = np.std(values, ddof=1)

    if np.isclose(sd, 0.0):
        return mean, 0.0, mean, mean

    sem = sd / np.sqrt(n)

    ci_low, ci_high = stats.t.interval(
        0.95,
        df=n - 1,
        loc=mean,
        scale=sem,
    )

    return mean, sd, ci_low, ci_high


def paired_statistics(adaptive, baseline):
    adaptive = np.asarray(adaptive, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    if len(adaptive) != len(baseline):
        raise ValueError("Paired samples must have the same length.")

    if len(adaptive) < 2:
        raise ValueError("At least two paired observations are required.")

    diff = adaptive - baseline

    n = len(diff)
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)

    sem_diff = stats.sem(diff)

    ci_low, ci_high = stats.t.interval(
        0.95,
        df=n - 1,
        loc=mean_diff,
        scale=sem_diff,
    )

    t_stat, t_p = stats.ttest_rel(
        adaptive,
        baseline,
    )

    try:
        w_stat, w_p = stats.wilcoxon(
            adaptive,
            baseline,
            zero_method="wilcox",
            alternative="two-sided",
        )
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    if sd_diff > 0:
        cohens_dz = mean_diff / sd_diff
    else:
        cohens_dz = np.nan

    return {
        "mean_difference": mean_diff,
        "sd_difference": sd_diff,
        "ci95_difference_low": ci_low,
        "ci95_difference_high": ci_high,
        "paired_t": t_stat,
        "paired_t_p": t_p,
        "wilcoxon": w_stat,
        "wilcoxon_p": w_p,
        "cohens_dz": cohens_dz,
    }


def load_mab_results(dataset):
    if dataset == "Amazon":
        base = Path("outputs/experiments/mab/multiseed/details")
        prefix = "mab_results_seed_"
    else:
        base = Path("outputs/experiments/kuairand_mab/multiseed/details")
        prefix = "kuairand_mab_seed_"
    frames = []
    for seed in SEEDS:
        path = base / f"{prefix}{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_mab(df):
    rows = []
    for seed in SEEDS:
        seed_df = df[df["seed"] == seed]
        for method in seed_df["method"].unique():
            method_df = seed_df[seed_df["method"] == method]
            row = {"seed": seed, "method": method}
            observed = method_df["observed_reward"].dropna()
            if len(observed) > 0:
                row["mean_reward"] = observed.mean()
            else:
                row["mean_reward"] = np.nan
            row["target_selection_rate"] = method_df["selected_target"].mean()
            row["non_greedy_rate"] = method_df["non_greedy"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def load_mechanism_results(dataset):
    if dataset == "Amazon":
        base = Path("outputs/experiments/mechanism_ablation")
    else:
        base = Path("outputs/experiments/kuairand_mechanism_ablation")
    frames = []
    for seed in SEEDS:
        path = base / f"mechanism_ablation_seed_{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def load_uncertainty_ablation_results(dataset):
    base = (
        Path("outputs/experiments/uncertainty_ablation")
        / dataset.lower()
    )

    frames = []

    for seed in SEEDS:
        path = base / f"seed_{seed}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing uncertainty-ablation file: {path}"
            )

        df = pd.read_csv(path)

        required = {
            "seed",
            "method",
            "observed_reward",
            "selected_target",
            "non_greedy",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def aggregate_mechanism(df):
    rows = []
    for seed in SEEDS:
        seed_df = df[df["seed"] == seed]
        for method in seed_df["method"].unique():
            method_df = seed_df[seed_df["method"] == method]
            observed = method_df["observed_reward"].dropna()
            if len(observed) > 0:
                mean_reward = observed.mean()
            else:
                mean_reward = np.nan
            rows.append({
                "seed": seed,
                "method": method,
                "mean_reward": mean_reward,
                "target_selection_rate": method_df["selected_target"].mean(),
                "non_greedy_rate": method_df["non_greedy"].mean(),
            })
    return pd.DataFrame(rows)

def aggregate_uncertainty_ablation(df):
    rows = []

    expected_methods = {
        "Adaptive-MAB",
        "Adaptive-MAB-w/o-Uncertainty",
    }

    for seed in SEEDS:
        seed_df = df[df["seed"] == seed]

        available_methods = set(
            seed_df["method"].dropna().unique()
        )

        missing_methods = (
            expected_methods - available_methods
        )

        if missing_methods:
            raise ValueError(
                f"Uncertainty ablation seed {seed} "
                f"is missing methods: {sorted(missing_methods)}"
            )

        for method in [
            "Adaptive-MAB",
            "Adaptive-MAB-w/o-Uncertainty",
        ]:
            method_df = seed_df[
                seed_df["method"] == method
            ]

            observed = (
                method_df["observed_reward"]
                .dropna()
            )

            mean_reward = (
                observed.mean()
                if len(observed) > 0
                else np.nan
            )

            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "mean_reward": mean_reward,
                    "target_selection_rate":
                        method_df["selected_target"].mean(),
                    "non_greedy_rate":
                        method_df["non_greedy"].mean(),
                }
            )

    return pd.DataFrame(rows)


def summarize_dataset(seed_df, dataset, experiment, comparisons=None,):
    summary_rows = []
    test_rows = []
    methods = seed_df["method"].unique()
    for method in methods:
        method_df = seed_df[seed_df["method"] == method]
        for metric in ["mean_reward", "target_selection_rate", "non_greedy_rate"]:
            values = method_df[metric].dropna().to_numpy()
            mean, sd, ci_low, ci_high = mean_sd_ci(values)
            summary_rows.append({
                "dataset": dataset,
                "experiment": experiment,
                "method": method,
                "metric": metric,
                "n_seeds": len(values),
                "mean": mean,
                "sd": sd,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            })
    if comparisons is None:
        comparisons = [
            (
                "Adaptive-MAB",
                "Transformer-Greedy",
            ),
            (
                "Adaptive-MAB",
                "Fixed-Beta-MAB",
            ),
        ]
    for adaptive_method, baseline_method in comparisons:
        if adaptive_method not in methods or baseline_method not in methods:
            continue
        for metric in ["mean_reward", "target_selection_rate", "non_greedy_rate"]:
            adaptive_values = []
            baseline_values = []
            for seed in SEEDS:
                adaptive_row = seed_df[(seed_df["seed"] == seed) & (seed_df["method"] == adaptive_method)]
                baseline_row = seed_df[(seed_df["seed"] == seed) & (seed_df["method"] == baseline_method)]
                if len(adaptive_row) == 1 and len(baseline_row) == 1:
                    a = adaptive_row.iloc[0][metric]
                    b = baseline_row.iloc[0][metric]
                    if not pd.isna(a) and not pd.isna(b):
                        adaptive_values.append(a)
                        baseline_values.append(b)
            if len(adaptive_values) < 2:
                continue
            result = paired_statistics(adaptive_values, baseline_values)
            test_rows.append({
                "dataset": dataset,
                "experiment": experiment,
                "adaptive_method": adaptive_method,
                "baseline_method": baseline_method,
                "metric": metric,
                "n_seeds": len(adaptive_values),
                **result,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def run_mab_analysis(dataset):
    df = load_mab_results(dataset)
    seed_df = aggregate_mab(df)
    seed_path = OUTPUT_DIR / f"{dataset.lower()}_mab_seed_metrics.csv"
    seed_df.to_csv(seed_path, index=False)
    summary, tests = summarize_dataset(seed_df, dataset, "MAB")
    return summary, tests


def run_mechanism_analysis(dataset):
    df = load_mechanism_results(dataset)
    seed_df = aggregate_mechanism(df)
    seed_path = OUTPUT_DIR / f"{dataset.lower()}_mechanism_seed_metrics.csv"
    seed_df.to_csv(seed_path, index=False)
    summary, tests = summarize_dataset(seed_df, dataset, "Mechanism Ablation")
    return summary, tests

def run_uncertainty_ablation_analysis(dataset):
    df = load_uncertainty_ablation_results(
        dataset
    )

    seed_df = aggregate_uncertainty_ablation(
        df
    )

    seed_path = (
        OUTPUT_DIR
        / f"{dataset.lower()}_uncertainty_ablation_seed_metrics.csv"
    )

    seed_df.to_csv(
        seed_path,
        index=False,
    )

    summary, tests = summarize_dataset(
        seed_df=seed_df,
        dataset=dataset,
        experiment="Uncertainty Ablation",
        comparisons=[
            (
                "Adaptive-MAB",
                "Adaptive-MAB-w/o-Uncertainty",
            )
        ],
    )

    return summary, tests


def print_summary(summary, title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    for _, row in summary.iterrows():
        print(f"{row['method']:25s} {row['metric']:25s} {row['mean']:.6f} ± {row['sd']:.6f} [95% CI: {row['ci95_low']:.6f}, {row['ci95_high']:.6f}]")


def print_tests(tests, title):
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)
    for _, row in tests.iterrows():
        print(f"{row['adaptive_method']} vs {row['baseline_method']} | {row['metric']}")
        print(f"  Difference: {row['mean_difference']:.6f}")
        print(f"  Difference 95% CI: "f"[{row['ci95_difference_low']:.6f}, "f"{row['ci95_difference_high']:.6f}]")
        print(f"  Paired t-test: t={row['paired_t']:.4f}, p={row['paired_t_p']:.6f}")
        print(f"  Wilcoxon: W={row['wilcoxon']}, p={row['wilcoxon_p']:.6f}")
        print(f"  Cohen's dz: {row['cohens_dz']:.4f}")
        print()


def main():
    all_summaries = []
    all_tests = []
    for dataset in ["Amazon", "KuaiRand"]:
        summary, tests = run_mab_analysis(dataset)
        all_summaries.append(summary)
        all_tests.append(tests)

        print_summary(
            summary,
            f"{dataset} - MAB",
        )

        print_tests(
            tests,
            f"{dataset} - MAB Statistical Tests",
        )

    for dataset in ["Amazon", "KuaiRand"]:
        summary, tests = run_mechanism_analysis(
            dataset
        )

        all_summaries.append(summary)
        all_tests.append(tests)

        print_summary(
            summary,
            f"{dataset} - Mechanism Ablation",
        )

        print_tests(
            tests,
            f"{dataset} - Mechanism Ablation Statistical Tests",
        )

    for dataset in ["Amazon", "KuaiRand"]:
        summary, tests = (
            run_uncertainty_ablation_analysis(
                dataset
            )
        )

        all_summaries.append(summary)
        all_tests.append(tests)

        print_summary(
            summary,
            f"{dataset} - Uncertainty Ablation",
        )

        print_tests(
            tests,
            f"{dataset} - Uncertainty Ablation Statistical Tests",
        )
    summary_df = pd.concat(all_summaries, ignore_index=True)
    tests_df = pd.concat(all_tests, ignore_index=True)
    summary_df.to_csv(OUTPUT_DIR / "statistical_summary.csv", index=False)
    tests_df.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
    results = {
        "summary": summary_df.to_dict(orient="records"),
        "tests": tests_df.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "statistical_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print()
    print("=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated:")
    print("  statistical_summary.csv")
    print("  statistical_tests.csv")
    print("  statistical_results.json")


if __name__ == "__main__":
    main()
