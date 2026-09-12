import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from src.data.sequence_dataset import SequenceDataset
from src.models.candidate_uncertainty import score_candidates_with_uncertainty
from src.data.candidate_generator import CandidateGenerator
from src.models.transformer import TransformerRewardModel
from src.bandit.adaptive_mab import AdaptiveMAB

SEEDS = [42, 123, 2024, 3407, 7777]
NUM_CANDIDATES = 100
MC_SAMPLES = 10
BATCH_SIZE = 256
MAX_SAMPLES_AMAZON = 100000
MAX_SAMPLES_KUAIRAND = 100000
BETA_MIN = 0.1
BETA_MAX = 0.5
TAU = 0.01
HISTORY_WEIGHT = 0.1
OUTPUT_DIR = Path("outputs/uncertainty")
DATASETS = {
    "Amazon": {
        "train_path": "data/processed/amazon/train_sequences.csv",
        "test_path": "data/processed/amazon/test_sequences.csv",
        "mapping_path": "data/processed/amazon/item_mapping.json",
        "checkpoint": "checkpoints/best_transformer.pt",
        "max_seq_len": 20,
    },
    "KuaiRand": {
        "train_path": "data/processed/Kuairand/train_sequences.csv",
        "test_path": "data/processed/Kuairand/test_sequences.csv",
        "mapping_path": "data/processed/Kuairand/item_mapping.json",
        "checkpoint": "checkpoints/best_kuairand_transformer.pt",
        "max_seq_len": 50,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(dataset_name, config, device):
    with open(config["mapping_path"], "r", encoding="utf-8") as f:
        mapping = json.load(f)
    num_items = mapping["num_items"]
    model = TransformerRewardModel(
        num_items=num_items,
        max_seq_len=config["max_seq_len"],
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        padding_idx=0,
    )
    checkpoint = torch.load(config["checkpoint"], map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, num_items


def load_dataset(dataset_name, config):
    dataset = SequenceDataset(
        sequence_path=config["test_path"],
        mapping_path=config["mapping_path"],
        max_seq_len=config["max_seq_len"],
    )

    return dataset

def analyze_dataset(dataset_name, config, device):
    print()
    print("=" * 70)
    print(f"Adaptive Exploration Analysis: {dataset_name}")
    print("=" * 70)
    all_records = []
    for seed in SEEDS:
        print()
        print("-" * 70)
        print(f"[Seed {seed}]")
        print("-" * 70)
        set_seed(seed)
        dataset = load_dataset(dataset_name, config)
        max_samples = MAX_SAMPLES_AMAZON if dataset_name == "Amazon" else MAX_SAMPLES_KUAIRAND
        num_samples = min(len(dataset), max_samples)
        indices = np.arange(num_samples)
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        model, num_items = load_model(dataset_name, config, device)
        candidate_generator = CandidateGenerator(num_items=num_items, num_candidates=NUM_CANDIDATES, seed=seed)
        bandit = AdaptiveMAB(beta_min=BETA_MIN, beta_max=BETA_MAX, tau=TAU, history_weight=HISTORY_WEIGHT)
        processed = 0
        for batch_idx, batch in enumerate(loader):
            hist_items = batch["history_items"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            target_items = batch["target_item"].cpu().numpy()
            target_rewards = batch["target_reward"].cpu().numpy()
            batch_size = hist_items.size(0)
            candidate_list = []
            target_positions = []
            for i in range(batch_size):
                candidates, target_idx = candidate_generator.generate(
                    target_item=int(target_items[i]),
                    history_items=hist_items[i].cpu().numpy(),
                )
                candidate_list.append(candidates)
                target_positions.append(target_idx)
            candidates = torch.tensor(np.stack(candidate_list), dtype=torch.long, device=device)
            mean_prediction, uncertainty, _ = score_candidates_with_uncertainty(
                model=model,
                hist_items=hist_items,
                attn_mask=attn_mask,
                candidates=candidates,
                num_samples=MC_SAMPLES,
            )
            mean_prediction = mean_prediction.cpu().numpy()
            uncertainty = uncertainty.cpu().numpy()
            for i in range(batch_size):
                candidate_items = candidates[i].cpu().numpy()
                candidate_predictions = mean_prediction[i]
                candidate_uncertainty = uncertainty[i]
                candidate_mean_uncertainty = float(np.mean(candidate_uncertainty))
                beta = bandit.compute_beta(candidate_uncertainty)
                selected_arm, scores, _ = bandit.select_arm(
                    predicted_reward=candidate_predictions,
                    uncertainty=candidate_uncertainty,
                    candidate_items=candidate_items,
                )
                greedy_arm = int(np.argmax(candidate_predictions))
                target_idx = target_positions[i]
                selected_is_target = (selected_arm == target_idx)
                is_exploration = (selected_arm != greedy_arm)
                selected_prediction = float(candidate_predictions[selected_arm])
                selected_uncertainty = float(candidate_uncertainty[selected_arm])
                target_reward = float(target_rewards[i])
                if selected_is_target:
                    observed_reward = target_reward
                    bandit.update(item_id=int(candidate_items[selected_arm]), reward=observed_reward)
                else:
                    observed_reward = np.nan
                all_records.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "sample_index": processed + i,
                    "candidate_mean_uncertainty": candidate_mean_uncertainty,
                    "beta": beta,
                    "selected_uncertainty": selected_uncertainty,
                    "selected_prediction": selected_prediction,
                    "is_exploration": int(is_exploration),
                    "selected_is_target": int(selected_is_target),
                    "target_reward": target_reward,
                    "observed_reward": observed_reward,
                })
            processed += batch_size
            if batch_idx % 50 == 0:
                print(f"  processed {processed:,}/{num_samples:,}")
        seed_df = pd.DataFrame([r for r in all_records if r["seed"] == seed and r["dataset"] == dataset_name])
        seed_path = OUTPUT_DIR / f"{dataset_name.lower()}_adaptive_exploration_seed_{seed}.csv"
        seed_df.to_csv(seed_path, index=False)
        print(f"[INFO] Saved: {seed_path}")
    return pd.DataFrame(all_records)


def make_uncertainty_groups(df):
    values = df["candidate_mean_uncertainty"].to_numpy()
    quantiles = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    groups = np.zeros(len(df), dtype=int)
    for i in range(5):
        lower = quantiles[i]
        upper = quantiles[i + 1]
        if i == 4:
            mask = (values >= lower) & (values <= upper)
        else:
            mask = (values >= lower) & (values < upper)
        groups[mask] = i + 1
    result = df.copy()
    result["uncertainty_group"] = groups
    return result, quantiles


def summarize_groups(df):
    grouped = df.groupby(["dataset", "uncertainty_group"]).agg(
        samples=("candidate_mean_uncertainty", "size"),
        mean_uncertainty=("candidate_mean_uncertainty", "mean"),
        std_uncertainty=("candidate_mean_uncertainty", "std"),
        mean_beta=("beta", "mean"),
        std_beta=("beta", "std"),
        exploration_rate=("is_exploration", "mean"),
        target_selection_rate=("selected_is_target", "mean"),
        mean_selected_uncertainty=("selected_uncertainty", "mean"),
        mean_selected_prediction=("selected_prediction", "mean"),
    ).reset_index()
    grouped["exploration_rate"] *= 100
    grouped["target_selection_rate"] *= 100
    return grouped


def calculate_correlations(df):
    results = []
    for dataset_name in df["dataset"].unique():
        subset = df[df["dataset"] == dataset_name]
        uncertainty = subset["candidate_mean_uncertainty"].to_numpy()
        beta = subset["beta"].to_numpy()
        exploration = subset["is_exploration"].to_numpy()
        pearson_beta = pearsonr(uncertainty, beta)
        spearman_beta = spearmanr(uncertainty, beta)
        pearson_exploration = pearsonr(uncertainty, exploration)
        spearman_exploration = spearmanr(uncertainty, exploration)
        results.append({
            "dataset": dataset_name,
            "pearson_uncertainty_beta": pearson_beta.statistic,
            "pearson_uncertainty_beta_p": pearson_beta.pvalue,
            "spearman_uncertainty_beta": spearman_beta.statistic,
            "spearman_uncertainty_beta_p": spearman_beta.pvalue,
            "pearson_uncertainty_exploration": pearson_exploration.statistic,
            "pearson_uncertainty_exploration_p": pearson_exploration.pvalue,
            "spearman_uncertainty_exploration": spearman_exploration.statistic,
            "spearman_uncertainty_exploration_p": spearman_exploration.pvalue,
        })
    return pd.DataFrame(results)


def print_results(group_summary, correlations):
    print()
    print("=" * 80)
    print("UNCERTAINTY GROUP ANALYSIS")
    print("=" * 80)
    display_columns = [
        "dataset",
        "uncertainty_group",
        "samples",
        "mean_uncertainty",
        "mean_beta",
        "exploration_rate",
        "target_selection_rate",
    ]
    print(group_summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    print(correlations.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Adaptive Uncertainty-Driven Exploration Analysis")
    print("=" * 80)
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    all_results = []
    for dataset_name, config in DATASETS.items():
        df = analyze_dataset(dataset_name, config, device)
        all_results.append(df)
    all_results = pd.concat(all_results, ignore_index=True)
    raw_path = OUTPUT_DIR / "adaptive_exploration_analysis_raw.csv"
    all_results.to_csv(raw_path, index=False)
    grouped_frames = []
    for dataset_name in all_results["dataset"].unique():
        dataset_df = all_results[all_results["dataset"] == dataset_name].copy()
        grouped_df, quantiles = make_uncertainty_groups(dataset_df)
        grouped_frames.append(grouped_df)
        print()
        print(f"[INFO] {dataset_name} uncertainty quantiles:")
        for i, value in enumerate(quantiles):
            print(f"  Q{i * 20}: {value:.8f}")
    grouped_all = pd.concat(grouped_frames, ignore_index=True)
    group_summary = summarize_groups(grouped_all)
    correlations = calculate_correlations(all_results)
    group_path = OUTPUT_DIR / "adaptive_exploration_group_summary.csv"
    correlation_path = OUTPUT_DIR / "adaptive_exploration_correlations.csv"
    summary_json_path = OUTPUT_DIR / "adaptive_exploration_summary.json"
    group_summary.to_csv(group_path, index=False)
    correlations.to_csv(correlation_path, index=False)
    summary = {
        "config": {
            "seeds": SEEDS,
            "num_candidates": NUM_CANDIDATES,
            "mc_samples": MC_SAMPLES,
            "batch_size": BATCH_SIZE,
            "beta_min": BETA_MIN,
            "beta_max": BETA_MAX,
            "tau": TAU,
            "history_weight": HISTORY_WEIGHT,
            "max_samples_amazon": MAX_SAMPLES_AMAZON,
            "max_samples_kuairand": MAX_SAMPLES_KUAIRAND,
        },
        "group_summary": group_summary.to_dict(orient="records"),
        "correlations": correlations.to_dict(orient="records"),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print()
    print_results(group_summary, correlations)
    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Raw data:       {raw_path}")
    print(f"Group summary:  {group_path}")
    print(f"Correlations:   {correlation_path}")
    print(f"JSON summary:   {summary_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
