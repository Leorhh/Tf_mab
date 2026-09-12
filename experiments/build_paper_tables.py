import os
import json
import glob
import pandas as pd
import numpy as np

OUTPUT_DIR = "outputs/paper_tables"
SEEDS = [42, 123, 2024, 3407, 7777]


def make_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[OK] {path} | rows={len(df)}")


def load_csv(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None
    return pd.read_csv(path)


# Table 1: Dataset Statistics
def build_table_1():
    rows = [
        {"Dataset": "Amazon Electronics", "Users": 728489, "Items": 159729, "Interactions": 6737580, "Max Sequence Length": 20},
        {"Dataset": "KuaiRand", "Users": 27285, "Items": 7583, "Interactions": 1186059, "Max Sequence Length": 50},
    ]
    df = pd.DataFrame(rows)
    save(df, "table_1_dataset_statistics.csv")


# Table 2: Transformer Prediction
def build_table_2():
    rows = [
        {"Dataset": "Amazon Electronics", "MSE": 0.101362, "MAE": 0.234860, "RMSE": 0.318374, "Prediction Mean": 0.824217, "Target Mean": 0.794412},
        {"Dataset": "KuaiRand", "MSE": 0.025981, "MAE": 0.116276, "RMSE": 0.161187, "Prediction Mean": 0.072534, "Target Mean": 0.075807},
    ]
    df = pd.DataFrame(rows)
    save(df, "table_2_transformer_prediction.csv")


# Table 3: Main MAB Comparison
def build_table_3():
    sources = [
        (
            "Amazon",
            "outputs/experiments/mab/multiseed/mab_multiseed_summary.csv",
        ),
        (
            "KuaiRand",
            "outputs/experiments/kuairand_mab/multiseed/kuairand_mab_multiseed_summary.csv",
        ),
    ]

    frames = []

    for dataset_name, path in sources:
        df = load_csv(path)

        if df is None:
            raise FileNotFoundError(
                f"Missing main MAB summary for {dataset_name}: {path}"
            )

        required = [
            "method",
            "mean_reward_mean",
            "mean_reward_std",
            "target_selection_rate_mean",
            "target_selection_rate_std",
            "non_greedy_rate_mean",
            "non_greedy_rate_std",
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset_name} main MAB summary is missing columns: "
                f"{missing}\nAvailable columns: {list(df.columns)}"
            )

        out = df[required].copy()

        out.insert(
            0,
            "Dataset",
            dataset_name,
        )

        out = out.rename(
            columns={
                "method": "Method",
                "mean_reward_mean": "Reward Mean",
                "mean_reward_std": "Reward SD",
                "target_selection_rate_mean":
                    "Target Selection Rate Mean",
                "target_selection_rate_std":
                    "Target Selection Rate SD",
                "non_greedy_rate_mean":
                    "Non-Greedy Selection Rate Mean",
                "non_greedy_rate_std":
                    "Non-Greedy Selection Rate SD",
            }
        )

        frames.append(out)

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    expected_methods = {
        "Random",
        "Transformer-Greedy",
        "Epsilon-Greedy",
        "UCB",
        "Thompson",
        "Adaptive-MAB",
    }

    for dataset_name in [
        "Amazon",
        "KuaiRand",
    ]:
        methods = set(
            df.loc[
                df["Dataset"] == dataset_name,
                "Method",
            ]
        )

        missing_methods = (
            expected_methods - methods
        )

        if missing_methods:
            raise ValueError(
                f"{dataset_name} Table 3 is missing methods: "
                f"{sorted(missing_methods)}"
            )

    method_name_map = {
        "UCB": "Predictive UCB",
        "Thompson": "Hybrid Thompson Sampling",
    }

    df["Method"] = (
        df["Method"]
        .replace(method_name_map)
    )

    save(
        df,
        "table_3_main_mab.csv",
    )


# Table 4: Mechanism Ablation
def build_table_4():
    sources = [
        (
            "Amazon",
            "outputs/experiments/mechanism_ablation/"
            "mechanism_ablation_summary.csv",
        ),
        (
            "KuaiRand",
            "outputs/experiments/kuairand_mechanism_ablation/"
            "mechanism_ablation_summary.csv",
        ),
    ]

    frames = []

    for dataset_name, path in sources:
        df = load_csv(path)

        if df is None:
            raise FileNotFoundError(
                f"Missing mechanism ablation summary "
                f"for {dataset_name}: {path}"
            )

        required = [
            "method",
            "mean_reward_mean",
            "mean_reward_std",
            "target_selection_rate_mean",
            "target_selection_rate_std",
            "non_greedy_rate_mean",
            "non_greedy_rate_std",
            "mean_selected_uncertainty_mean",
            "mean_beta_mean",
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset_name} mechanism ablation "
                f"is missing columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        out = df[required].copy()

        out.insert(
            0,
            "Dataset",
            dataset_name,
        )

        out = out.rename(
            columns={
                "method": "Method",
                "mean_reward_mean": "Reward Mean",
                "mean_reward_std": "Reward SD",
                "target_selection_rate_mean":
                    "Target Selection Rate",
                "target_selection_rate_std":
                    "Target Selection Rate SD",
                "non_greedy_rate_mean":
                    "Non-Greedy Selection Rate",
                "non_greedy_rate_std":
                    "Non-Greedy Selection Rate SD",
                "mean_selected_uncertainty_mean":
                    "Mean Selected Uncertainty",
                "mean_beta_mean":
                    "Mean Beta",
            }
        )

        frames.append(out)

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    expected_methods = {
        "Transformer-Greedy",
        "Fixed-Beta-MAB",
        "Adaptive-MAB",
    }

    for dataset_name in [
        "Amazon",
        "KuaiRand",
    ]:
        methods = set(
            df.loc[
                df["Dataset"] == dataset_name,
                "Method",
            ]
        )

        missing_methods = (
            expected_methods - methods
        )

        if missing_methods:
            raise ValueError(
                f"{dataset_name} Table 4 is missing methods: "
                f"{sorted(missing_methods)}"
            )

    save(
        df,
        "table_4_mechanism_ablation.csv",
    )


# Table 5: Uncertainty Ablation
def build_table_5():
    path = "outputs/experiments/uncertainty_ablation/final_summary.csv"
    df = load_csv(path)
    if df is None:
        return
    rename = {
        "dataset": "Dataset",
        "method": "Method",
        "num_seeds": "Seeds",
        "mean_reward_mean": "Reward Mean",
        "mean_reward_std": "Reward SD",
        "target_selection_rate_mean": "Target Selection Rate",
        "target_selection_rate_std": "Target Selection Rate SD",
        "non_greedy_rate_mean": "Non-Greedy Selection Rate",
        "non_greedy_rate_std": "Non-Greedy Selection Rate SD",
        "mean_selected_uncertainty_mean": "Mean Selected Uncertainty",
        "mean_beta_mean": "Mean Beta",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    save(df, "table_5_uncertainty_ablation.csv")


# Table 6: Candidate Size Sensitivity
def build_table_6():
    path = "outputs/experiments/candidate_size_sensitivity/final_summary.csv"
    df = load_csv(path)
    if df is None:
        return
    columns = [
        "dataset", "candidate_size", "num_seeds",
        "mean_reward_mean", "mean_reward_std",
        "target_selection_rate_mean", "target_selection_rate_std",
        "non_greedy_rate_mean", "non_greedy_rate_std",
        "mean_selected_uncertainty_mean", "mean_beta_mean",
    ]
    columns = [col for col in columns if col in df.columns]
    df = df[columns]
    rename = {
        "dataset": "Dataset",
        "candidate_size": "Candidate Size",
        "num_seeds": "Seeds",
        "mean_reward_mean": "Reward Mean",
        "mean_reward_std": "Reward SD",
        "target_selection_rate_mean": "Target Selection Rate",
        "target_selection_rate_std": "Target Selection Rate SD",
        "non_greedy_rate_mean": "Exploration Rate",
        "non_greedy_rate_std": "Exploration Rate SD",
        "mean_selected_uncertainty_mean": "Mean Selected Uncertainty",
        "mean_beta_mean": "Mean Beta",
    }
    df = df.rename(columns=rename)
    save(df, "table_6_candidate_size.csv")


# Table 7: MC Dropout Sensitivity
def build_table_7():
    path = "outputs/experiments/mc_dropout_sensitivity/final_summary.csv"
    df = load_csv(path)
    if df is None:
        return
    columns = [
        "dataset", "mc_samples", "candidate_size", "num_seeds",
        "mean_reward_mean", "mean_reward_std",
        "target_selection_rate_mean", "target_selection_rate_std",
        "non_greedy_rate_mean", "non_greedy_rate_std",
        "mean_selected_prediction_mean", "mean_selected_uncertainty_mean",
        "mean_beta_mean",
    ]
    columns = [col for col in columns if col in df.columns]
    df = df[columns]
    rename = {
        "dataset": "Dataset",
        "mc_samples": "MC Samples",
        "candidate_size": "Candidate Size",
        "num_seeds": "Seeds",
        "mean_reward_mean": "Reward Mean",
        "mean_reward_std": "Reward SD",
        "target_selection_rate_mean": "Target Selection Rate",
        "target_selection_rate_std": "Target Selection Rate SD",
        "non_greedy_rate_mean": "Exploration Rate",
        "non_greedy_rate_std": "Exploration Rate SD",
        "mean_selected_prediction_mean": "Mean Selected Prediction",
        "mean_selected_uncertainty_mean": "Mean Selected Uncertainty",
        "mean_beta_mean": "Mean Beta",
    }
    df = df.rename(columns=rename)
    save(df, "table_7_mc_dropout.csv")


# Table 8: Statistical Significance
def build_table_8():
    path = "outputs/statistical_analysis/statistical_tests.csv"

    df = load_csv(path)

    if df is None:
        raise FileNotFoundError(
            f"Statistical test results not found: {path}\n"
            "Run first:\n"
            "PYTHONPATH=. python experiments/statistical_analysis.py"
        )

    required_columns = [
        "dataset",
        "experiment",
        "adaptive_method",
        "baseline_method",
        "metric",
        "n_seeds",
        "mean_difference",
        "ci95_difference_low",
        "ci95_difference_high",
        "paired_t",
        "paired_t_p",
        "wilcoxon",
        "wilcoxon_p",
        "cohens_dz",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "statistical_tests.csv is missing required columns: "
            f"{missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Keep only experiments intended for the paper.
    allowed_experiments = [
        "MAB",
        "Mechanism Ablation",
        "Uncertainty Ablation",
    ]
    comparison_mask = (
            (
                    (df["experiment"] == "MAB")
                    & (df["baseline_method"] == "Transformer-Greedy")
            )
            |
            (
                    (df["experiment"] == "Mechanism Ablation")
                    & (df["baseline_method"] == "Fixed-Beta-MAB")
            )
            |
            (
                    (df["experiment"] == "Uncertainty Ablation")
                    & (
                            df["baseline_method"]
                            == "Adaptive-MAB-w/o-Uncertainty"
                    )
            )
    )

    df = df[comparison_mask].copy()

    df = df[
        df["experiment"].isin(allowed_experiments)
    ].copy()

    if df.empty:
        raise ValueError(
            "No supported statistical-test rows were found."
        )

    # Friendly names for the paper.
    experiment_map = {
        "MAB": "Main MAB",
        "Mechanism Ablation": "Mechanism Ablation",
        "Uncertainty Ablation": "Uncertainty Ablation",
    }

    metric_map = {
        "mean_reward": "Mean Observed Reward",
        "target_selection_rate": "Target Selection Rate",
        "non_greedy_rate": "Non-Greedy Selection Rate",
    }

    method_map = {
        "Transformer-Greedy": "Greedy",
        "Fixed-Beta-MAB": "Fixed-Beta MAB",
        "Adaptive-MAB": "Adaptive-MAB",
        "Adaptive-MAB-w/o-Uncertainty":
            "Adaptive-MAB w/o Uncertainty",
    }

    df["Experiment"] = (
        df["experiment"]
        .map(experiment_map)
        .fillna(df["experiment"])
    )

    df["Metric"] = (
        df["metric"]
        .map(metric_map)
        .fillna(df["metric"])
    )

    df["Method"] = (
        df["adaptive_method"]
        .map(method_map)
        .fillna(df["adaptive_method"])
    )

    df["Baseline"] = (
        df["baseline_method"]
        .map(method_map)
        .fillna(df["baseline_method"])
    )

    # --------------------------------------------------------
    # Convert rate differences to percentage points.
    #
    # Example:
    # 0.008700 -> 0.870 percentage points
    #
    # Reward differences remain on the original reward scale.
    # --------------------------------------------------------

    rate_metrics = {
        "target_selection_rate",
        "non_greedy_rate",
    }

    df["Unit"] = np.where(
        df["metric"].isin(rate_metrics),
        "pp",
        "reward",
    )

    rate_mask = df["metric"].isin(rate_metrics)

    df["Difference"] = df["mean_difference"].astype(float)
    df["CI95 Low"] = df["ci95_difference_low"].astype(float)
    df["CI95 High"] = df["ci95_difference_high"].astype(float)

    df.loc[rate_mask, "Difference"] *= 100.0
    df.loc[rate_mask, "CI95 Low"] *= 100.0
    df.loc[rate_mask, "CI95 High"] *= 100.0

    # --------------------------------------------------------
    # Final paper table
    # --------------------------------------------------------

    out = pd.DataFrame({
        "Dataset": df["dataset"],
        "Experiment": df["Experiment"],
        "Method": df["Method"],
        "Baseline": df["Baseline"],
        "Metric": df["Metric"],
        "N Seeds": df["n_seeds"].astype(int),

        "Difference": df["Difference"],
        "95% CI Low": df["CI95 Low"],
        "95% CI High": df["CI95 High"],
        "Unit": df["Unit"],

        "Paired t": df["paired_t"],
        "Paired t p": df["paired_t_p"],

        "Wilcoxon W": df["wilcoxon"],
        "Wilcoxon p": df["wilcoxon_p"],

        "Cohen dz": df["cohens_dz"],
    })

    # Logical ordering in the paper.
    experiment_order = {
        "Main MAB": 0,
        "Mechanism Ablation": 1,
        "Uncertainty Ablation": 2,
    }

    metric_order = {
        "Mean Observed Reward": 0,
        "Target Selection Rate": 1,
        "Non-Greedy Selection Rate": 2,
    }

    dataset_order = {
        "Amazon": 0,
        "KuaiRand": 1,
    }

    out["_dataset_order"] = (
        out["Dataset"]
        .map(dataset_order)
        .fillna(99)
    )

    out["_experiment_order"] = (
        out["Experiment"]
        .map(experiment_order)
        .fillna(99)
    )

    out["_metric_order"] = (
        out["Metric"]
        .map(metric_order)
        .fillna(99)
    )

    out = (
        out.sort_values(
            [
                "_dataset_order",
                "_experiment_order",
                "Baseline",
                "_metric_order",
            ]
        )
        .drop(
            columns=[
                "_dataset_order",
                "_experiment_order",
                "_metric_order",
            ]
        )
        .reset_index(drop=True)
    )

    save(
        out,
        "table_8_statistical_significance.csv",
    )


# Table 9: KuaiRand Ranking
def build_ranking_table():
    sources = [
        (
            "Amazon",
            "outputs/experiments/amazon_ranking/"
            "amazon_ranking_summary.csv",
        ),
        (
            "KuaiRand",
            "outputs/experiments/kuairand_ranking/"
            "kuairand_ranking_summary.csv",
        ),
    ]

    frames = []

    for dataset_name, path in sources:
        df = load_csv(path)

        if df is None:
            raise FileNotFoundError(
                f"Missing ranking summary for {dataset_name}: {path}"
            )

        row = df.copy()

        if "dataset" not in row.columns:
            row.insert(
                0,
                "Dataset",
                dataset_name,
            )
        else:
            row = row.rename(
                columns={
                    "dataset": "Dataset",
                }
            )
            row["Dataset"] = dataset_name

        frames.append(row)

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    wanted = [
        "Dataset",
        "num_seeds",
        "samples_per_seed",
        "hit@1_mean",
        "hit@1_std",
        "hit@5_mean",
        "hit@5_std",
        "hit@10_mean",
        "hit@10_std",
        "ndcg@10_mean",
        "ndcg@10_std",
        "mrr_mean",
        "mrr_std",
    ]

    missing = [
        c for c in wanted
        if c not in out.columns
    ]

    if missing:
        raise ValueError(
            f"Ranking table missing columns: {missing}\n"
            f"Available: {list(out.columns)}"
        )

    out = out[wanted]

    save(
        out,
        "table_9_ranking.csv",
    )


def main():
    print("=" * 80)
    print("BUILDING PAPER RESULT TABLES")
    print("=" * 80)
    make_dir()
    print("\n[1/9] Dataset statistics")
    build_table_1()
    print("\n[2/9] Transformer prediction")
    build_table_2()
    print("\n[3/9] Main MAB comparison")
    build_table_3()
    print("\n[4/9] Mechanism ablation")
    build_table_4()
    print("\n[5/9] Uncertainty ablation")
    build_table_5()
    print("\n[6/9] Candidate size sensitivity")
    build_table_6()
    print("\n[7/9] MC Dropout sensitivity")
    build_table_7()
    print("\n[8/9] Statistical significance")
    build_table_8()
    print("\n[9/9] Ranking evaluation")
    build_ranking_table()
    print("\n" + "=" * 80)
    print("PAPER TABLE GENERATION FINISHED")
    print("=" * 80)
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.csv")))
    print(f"\nGenerated {len(files)} files:")
    for path in files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
