import os
import json
import glob
import pandas as pd
import numpy as np
ROOT = "outputs"
def title(text):
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)
def check_file(path):
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[OK] {path} | rows={len(df)}")
        return df
    except Exception as e:
        print(f"[ERROR] {path} | {e}")
        return None
def check_seed_files(directory, expected_seeds):
    files = glob.glob(
        os.path.join(directory, "*.csv")
    )
    found = []
    for path in files:
        name = os.path.basename(path)
        for seed in expected_seeds:
            if f"seed_{seed}" in name:
                found.append(seed)
    found = sorted(set(found))
    missing = sorted(
        set(expected_seeds) - set(found)
    )
    print(f"Directory: {directory}")
    print(f"Found seeds: {found}")
    if missing:
        print(f"[WARN] Missing seeds: {missing}")
    else:
        print("[OK] All 5 seeds present")
    return found, missing
def check_numeric_columns(df, name):
    if df is None or df.empty:
        return
    print(f"\n{name}")
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue
        if not np.isfinite(values).all():
            print(
                f"[WARN] {col}: contains NaN/Inf"
            )
        vmin = values.min()
        vmax = values.max()
        if (
            "rate" in col
            or "selection" in col
            or "uncertainty" in col
            or "reward" in col
        ):
            if vmin < 0 or vmax > 1:
                print(
                    f"[WARN] {col}: "
                    f"range={vmin:.6f} ~ {vmax:.6f}"
                )
def main():
    title(
        "FULL EXPERIMENT RESULT VERIFICATION"
    )
    print(
        f"Working directory: "
        f"{os.path.abspath('.')}"
    )
    print(
        f"Output directory exists: "
        f"{os.path.exists(ROOT)}"
    )
    seeds = [
        123,
        2024,
        3407,
        42,
        7777,
    ]
    # 1. Main MAB - Amazon
    title("1. AMAZON MAIN MAB")
    amazon_mab_dir = (
        "outputs/experiments/mab/"
        "multiseed/details"
    )
    check_seed_files(
        amazon_mab_dir,
        seeds,
    )
    amazon_mab_files = glob.glob(
        os.path.join(
            amazon_mab_dir,
            "mab_results_seed_*.csv"
        )
    )
    print(
        f"Total seed files: "
        f"{len(amazon_mab_files)}"
    )
    # 2. Main MAB - KuaiRand
    title("2. KAUIRAND MAIN MAB")
    kuai_mab_dir = (
        "outputs/experiments/kuairand_mab/"
        "multiseed/details"
    )
    check_seed_files(
        kuai_mab_dir,
        seeds,
    )
    kuai_mab_files = glob.glob(
        os.path.join(
            kuai_mab_dir,
            "kuairand_mab_seed_*.csv"
        )
    )
    print(
        f"Total seed files: "
        f"{len(kuai_mab_files)}"
    )
    # 3. Candidate Size
    title("3. CANDIDATE SIZE SENSITIVITY")
    candidate_expected = {
        "Amazon": [20, 50, 100, 200],
        "KuaiRand": [20, 50, 100, 200],
    }
    candidate_root = (
        "outputs/experiments/"
        "candidate_size_sensitivity"
    )
    for dataset, sizes in candidate_expected.items():
        dataset_dir = os.path.join(
            candidate_root,
            dataset.lower(),
        )
        print()
        print(
            f"--- {dataset} ---"
        )
        for size in sizes:
            files = glob.glob(
                os.path.join(
                    dataset_dir,
                    f"candidates_{size}_seed_*.csv",
                )
            )
            found_seeds = []
            for path in files:
                name = os.path.basename(path)
                for seed in seeds:
                    if f"seed_{seed}" in name:
                        found_seeds.append(seed)
            found_seeds = sorted(
                set(found_seeds)
            )
            missing = sorted(
                set(seeds) - set(found_seeds)
            )
            print(
                f"candidates={size}: "
                f"{len(found_seeds)}/5 seeds"
            )
            if missing:
                print(
                    f"  [WARN] missing={missing}"
                )
            else:
                print(
                    "  [OK] complete"
                )
    # 4. MC Dropout
    title("4. MC DROPOUT SENSITIVITY")
    mc_root = (
        "outputs/experiments/"
        "mc_dropout_sensitivity"
    )
    for dataset in [
        "Amazon",
        "KuaiRand",
    ]:
        dataset_dir = os.path.join(
            mc_root,
            dataset.lower(),
        )
        print()
        print(
            f"--- {dataset} ---"
        )
        for mc in [5, 10, 20]:
            files = glob.glob(
                os.path.join(
                    dataset_dir,
                    f"mc_{mc}_seed_*.csv",
                )
            )
            found_seeds = []
            for path in files:
                name = os.path.basename(path)
                for seed in seeds:
                    if f"seed_{seed}" in name:
                        found_seeds.append(seed)
            found_seeds = sorted(
                set(found_seeds)
            )
            missing = sorted(
                set(seeds) - set(found_seeds)
            )
            print(
                f"MC={mc}: "
                f"{len(found_seeds)}/5 seeds"
            )
            if missing:
                print(
                    f"  [WARN] missing={missing}"
                )
            else:
                print(
                    "  [OK] complete"
                )
    # 5. Candidate Size Data Quality
    title(
        "5. CANDIDATE SIZE DATA QUALITY"
    )
    candidate_summary = (
        "outputs/experiments/"
        "candidate_size_sensitivity/"
        "final_summary.csv"
    )
    df_candidate = check_file(
        candidate_summary
    )
    if df_candidate is not None:
        print()
        print(
            df_candidate[
                [
                    "dataset",
                    "candidate_size",
                    "num_seeds",
                    "mean_reward_mean",
                    "mean_reward_std",
                    "target_selection_rate_mean",
                    "non_greedy_rate_mean",
                ]
            ].to_string(index=False)
        )
        bad_seed_counts = df_candidate[
            df_candidate["num_seeds"] != 5
        ]
        if len(bad_seed_counts) > 0:
            print(
                "\n[WARN] Some candidate-size "
                "settings do not have 5 seeds:"
            )
            print(
                bad_seed_counts.to_string(
                    index=False
                )
            )
        else:
            print(
                "\n[OK] All candidate-size "
                "settings have 5 seeds."
            )
        check_numeric_columns(
            df_candidate,
            "Candidate-size numeric checks",
        )
    # 6. MC Data Quality
    title(
        "6. MC DROPOUT DATA QUALITY"
    )
    mc_summary = (
        "outputs/experiments/"
        "mc_dropout_sensitivity/"
        "final_summary.csv"
    )
    df_mc = check_file(
        mc_summary
    )
    if df_mc is not None:
        print()
        print(
            df_mc[
                [
                    "dataset",
                    "mc_samples",
                    "candidate_size",
                    "num_seeds",
                    "mean_reward_mean",
                    "mean_reward_std",
                    "non_greedy_rate_mean",
                    "mean_selected_uncertainty_mean",
                    "mean_beta_mean",
                ]
            ].to_string(index=False)
        )
        bad_seed_counts = df_mc[
            df_mc["num_seeds"] != 5
        ]
        if len(bad_seed_counts) > 0:
            print(
                "\n[WARN] Some MC settings "
                "do not have 5 seeds."
            )
        else:
            print(
                "\n[OK] All MC settings "
                "have 5 seeds."
            )
        check_numeric_columns(
            df_mc,
            "MC numeric checks",
        )
    # 7. Uncertainty Ablation
    title(
        "7. UNCERTAINTY ABLATION"
    )
    uncertainty_summary = (
        "outputs/experiments/"
        "uncertainty_ablation/"
        "final_summary.csv"
    )
    df_uncertainty = check_file(
        uncertainty_summary
    )
    if df_uncertainty is not None:
        print()
        print(
            df_uncertainty.to_string(
                index=False
            )
        )
        check_numeric_columns(
            df_uncertainty,
            "Uncertainty ablation checks",
        )
    # 8. Statistical Analysis
    title(
        "8. STATISTICAL ANALYSIS"
    )
    statistical_files = [
        "outputs/statistical_analysis/"
        "statistical_summary.csv",
        "outputs/statistical_analysis/"
        "statistical_tests.csv",
        "outputs/statistical_analysis/"
        "paired_difference_ci.csv",
        "outputs/statistical_analysis/"
        "statistical_results.json",
    ]
    for path in statistical_files:
        if os.path.exists(path):
            print(
                f"[OK] {path}"
            )
        else:
            print(
                f"[MISSING] {path}"
            )
    # 9. Ranking
    title(
        "9. KUairand RANKING EVALUATION"
    )
    ranking_files = glob.glob(
        "outputs/experiments/"
        "**/*ranking*.csv",
        recursive=True,
    )
    if ranking_files:
        for path in ranking_files:
            print(
                f"[OK] {path}"
            )
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    print(
                        df.tail().to_string(
                            index=False
                        )
                    )
            except Exception as e:
                print(
                    f"[WARN] Could not read: "
                    f"{e}"
                )
    else:
        print(
            "[INFO] No ranking CSV found "
            "with filename containing "
            "'ranking'."
        )
    # 10. Search for suspicious zeros
    title(
        "10. SUSPICIOUS ZERO CHECK"
    )
    csv_files = glob.glob(
        os.path.join(
            ROOT,
            "**",
            "*.csv",
        ),
        recursive=True,
    )
    suspicious = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            for col in df.columns:
                if (
                    col.endswith("_mean")
                    or col.endswith("_rate")
                    or "uncertainty" in col
                    or "prediction" in col
                ):
                    if pd.api.types.is_numeric_dtype(
                        df[col]
                    ):
                        zero_count = (
                            df[col] == 0
                        ).sum()
                        if (
                            zero_count > 0
                            and zero_count
                            == len(df)
                        ):
                            suspicious.append(
                                (
                                    path,
                                    col,
                                    zero_count,
                                )
                            )
        except Exception:
            pass
    if suspicious:
        print(
            "[WARN] Columns containing "
            "only zero values:"
        )
        for path, col, count in suspicious:
            print(
                f"  {path}"
            )
            print(
                f"    column={col}, "
                f"rows={count}"
            )
    else:
        print(
            "[OK] No all-zero suspicious "
            "columns detected."
        )
    # 11. Final file inventory
    title(
        "11. OUTPUT FILE INVENTORY"
    )
    all_csv = glob.glob(
        os.path.join(
            ROOT,
            "**",
            "*.csv",
        ),
        recursive=True,
    )
    all_json = glob.glob(
        os.path.join(
            ROOT,
            "**",
            "*.json",
        ),
        recursive=True,
    )
    print(
        f"CSV files : {len(all_csv)}"
    )
    print(
        f"JSON files: {len(all_json)}"
    )
    # Final
    title(
        "VERIFICATION FINISHED"
    )
    print(
        "No files were modified."
    )
    print(
        "This script only reads experiment outputs."
    )
if __name__ == "__main__":
    main()
