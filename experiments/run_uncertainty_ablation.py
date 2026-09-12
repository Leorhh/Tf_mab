import json
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.sequence_dataset import SequenceDataset
from src.models.transformer import TransformerRewardModel
from src.models.candidate_uncertainty import score_candidates_with_uncertainty
from src.data.candidate_generator import CandidateGenerator
from src.bandit.adaptive_mab import AdaptiveMAB

SEEDS = [42, 123, 2024, 3407, 7777]
BATCH_SIZE = 256
NUM_CANDIDATES = 100
MC_SAMPLES = 10
BETA_MIN = 0.1
BETA_MAX = 0.5
TAU = 0.01
HISTORY_WEIGHT = 0.1
AMAZON_MAX_SAMPLES = 100000
KUAIRAND_MAX_SAMPLES = 27285
OUTPUT_DIR = "outputs/experiments/uncertainty_ablation"
DATASETS = {
    "Amazon": {
        "test_path": "data/processed/amazon/test_sequences.csv",
        "mapping_path": "data/processed/amazon/item_mapping.json",
        "checkpoint": "checkpoints/best_transformer.pt",
        "max_seq_len": 20,
        "max_samples": AMAZON_MAX_SAMPLES,
    },
    "KuaiRand": {
        "test_path": "data/processed/Kuairand/test_sequences.csv",
        "mapping_path": "data/processed/Kuairand/item_mapping.json",
        "checkpoint": "checkpoints/best_kuairand_transformer.pt",
        "max_seq_len": 50,
        "max_samples": KUAIRAND_MAX_SAMPLES,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(mapping_path, checkpoint_path, device, max_seq_len):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    num_items = mapping["num_items"]

    model = TransformerRewardModel(
        num_items=num_items,
        max_seq_len=max_seq_len,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        padding_idx=0,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, num_items


def select_without_uncertainty(
    predicted_reward,
    candidate_items,
    mab,
):
    scores = []

    for reward, item_id in zip(
        predicted_reward,
        candidate_items,
    ):
        item_id = int(item_id)

        count = mab.item_counts.get(
            item_id,
            0,
        )

        if count == 0:
            historical_mean = 0.0
        else:
            historical_mean = (
                mab.item_reward_sums[item_id]
                / count
            )

        score = (
            float(reward)
            + HISTORY_WEIGHT
            * historical_mean
        )

        scores.append(score)

    return int(np.argmax(scores))


def run_experiment(dataset_name, config, seed, device):
    set_seed(seed)
    print()
    print("=" * 70)
    print(f"{dataset_name} | seed={seed}")
    print("=" * 70)
    dataset = SequenceDataset(
        sequence_path=config["test_path"],
        mapping_path=config["mapping_path"],
        max_seq_len=config["max_seq_len"],
    )
    max_samples = min(len(dataset), config["max_samples"])
    generator = torch.utils.data.Subset(dataset, range(max_samples))
    loader = DataLoader(generator, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model, num_items = load_model(
        config["mapping_path"],
        config["checkpoint"],
        device,
        config["max_seq_len"],
    )
    candidate_generator = CandidateGenerator(num_items=num_items, num_candidates=NUM_CANDIDATES, seed=seed)
    adaptive = AdaptiveMAB(beta_min=BETA_MIN, beta_max=BETA_MAX, tau=TAU, history_weight=HISTORY_WEIGHT)
    no_uncertainty = AdaptiveMAB(beta_min=BETA_MIN, beta_max=BETA_MAX, tau=TAU, history_weight=HISTORY_WEIGHT)
    results = []
    sample_index = 0
    for batch in loader:
        hist_items = batch["history_items"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        target_items = batch["target_item"].to(device)
        target_rewards = batch["target_reward"].to(device)
        batch_size = hist_items.size(0)
        with torch.no_grad():
            candidates_list = []
            target_indices = []
            for i in range(batch_size):
                candidates, target_idx = candidate_generator.generate(
                    target_item=int(target_items[i].item()),
                    history_items=hist_items[i].cpu().numpy(),
                )
                candidates_list.append(candidates)
                target_indices.append(target_idx)
            candidates = torch.tensor(np.stack(candidates_list), dtype=torch.long, device=device)
            mean_prediction, uncertainty, _ = score_candidates_with_uncertainty(
                model=model,
                hist_items=hist_items,
                attn_mask=attn_mask,
                candidates=candidates,
                num_samples=MC_SAMPLES,
            )
        mean_prediction_np = mean_prediction.cpu().numpy()
        uncertainty_np = uncertainty.cpu().numpy()
        candidates_np = candidates.cpu().numpy()
        target_items_np = target_items.cpu().numpy()
        target_rewards_np = target_rewards.cpu().numpy()
        for i in range(batch_size):
            candidate_items = candidates_np[i]
            predicted_reward = mean_prediction_np[i]
            candidate_uncertainty = uncertainty_np[i]
            target_item = int(target_items_np[i])
            target_reward = float(target_rewards_np[i])
            greedy_idx = int(np.argmax(predicted_reward))
            adaptive_idx, _, beta = adaptive.select_arm(
                predicted_reward=predicted_reward,
                uncertainty=candidate_uncertainty,
                candidate_items=candidate_items,
            )
            no_unc_idx = select_without_uncertainty(
                predicted_reward=predicted_reward,
                candidate_items=candidate_items,
                mab=no_uncertainty,
            )
            adaptive_item = int(candidate_items[adaptive_idx])
            no_unc_item = int(candidate_items[no_unc_idx])
            greedy_item = int(candidate_items[greedy_idx])
            adaptive_target = adaptive_item == target_item
            no_unc_target = no_unc_item == target_item
            greedy_target = greedy_item == target_item
            adaptive_non_greedy = adaptive_idx != greedy_idx
            no_unc_non_greedy = no_unc_idx != greedy_idx
            adaptive_observed_reward = target_reward if adaptive_target else np.nan
            no_unc_observed_reward = target_reward if no_unc_target else np.nan
            if adaptive_target:
                adaptive.update(adaptive_item, target_reward)
            if no_unc_target:
                no_uncertainty.update(no_unc_item, target_reward)
            results.append({
                "seed": seed,
                "sample_index": sample_index,
                "method": "Adaptive-MAB",
                "selected_item": adaptive_item,
                "target_item": target_item,
                "target_reward": target_reward,
                "observed_reward": adaptive_observed_reward,
                "selected_target": int(adaptive_target),
                "non_greedy": int(adaptive_non_greedy),
                "selected_prediction": float(predicted_reward[adaptive_idx]),
                "selected_uncertainty": float(candidate_uncertainty[adaptive_idx]),
                "beta": float(beta),
                "greedy_target": int(greedy_target),
            })
            results.append({
                "seed": seed,
                "sample_index": sample_index,
                "method": "Adaptive-MAB-w/o-Uncertainty",
                "selected_item": no_unc_item,
                "target_item": target_item,
                "target_reward": target_reward,
                "observed_reward": no_unc_observed_reward,
                "selected_target": int(no_unc_target),
                "non_greedy": int(no_unc_non_greedy),
                "selected_prediction": float(predicted_reward[no_unc_idx]),
                "selected_uncertainty": float(candidate_uncertainty[no_unc_idx]),
                "beta": 0.0,
                "greedy_target": int(greedy_target),
            })
            sample_index += 1
        if sample_index % 10000 < batch_size:
            print(f"[INFO] Processed {sample_index}/{max_samples}")
    return pd.DataFrame(results)


def summarize_seed_results(df):
    rows = []
    for method, group in df.groupby("method"):
        observed = group["observed_reward"].dropna()
        rows.append({
            "method": method,
            "samples": len(group),
            "mean_reward": observed.mean() if len(observed) > 0 else 0.0,
            "target_selection_rate": group["selected_target"].mean(),
            "non_greedy_rate": group["non_greedy"].mean(),
            "mean_selected_prediction": group["selected_prediction"].mean(),
            "mean_selected_uncertainty": group["selected_uncertainty"].mean(),
            "mean_beta": group["beta"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Uncertainty Ablation")
    print("=" * 70)
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_summary = []
    for dataset_name, config in DATASETS.items():
        dataset_dir = os.path.join(OUTPUT_DIR, dataset_name.lower())
        os.makedirs(dataset_dir, exist_ok=True)
        # for seed in SEEDS:
        #     df = run_experiment(
        #         dataset_name=dataset_name,
        #         config=config,
        #         seed=seed,
        #         device=device,
        #     )
        # 临时更改
        for seed in SEEDS:
            raw_path = os.path.join(
                dataset_dir,
                f"seed_{seed}.csv",
            )

            if os.path.exists(raw_path):
                print(f"[INFO] Loading existing result: {raw_path}")
                df = pd.read_csv(raw_path)

            else:
                df = run_experiment(
                    dataset_name=dataset_name,
                    config=config,
                    seed=seed,
                    device=device,
                )

                df.to_csv(
                    raw_path,
                    index=False,
                )

                print(f"[INFO] Saved: {raw_path}")

            seed_summary = summarize_seed_results(df)

            seed_summary.insert(
                0,
                "dataset",
                dataset_name,
            )

            seed_summary.insert(
                1,
                "seed",
                seed,
            )

            all_summary.append(seed_summary)
    if not all_summary:
        raise RuntimeError(
            "No uncertainty-ablation results were loaded. "
            "Check seed CSV files and experiment paths."
        )

    summary_df = pd.concat(
        all_summary,
        ignore_index=True,
    )
    summary_path = os.path.join(OUTPUT_DIR, "seed_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    final_summary = (
        summary_df
        .groupby(["dataset", "method"])
        .agg(
            num_seeds=("seed", "count"),
            mean_reward_mean=("mean_reward", "mean"),
            mean_reward_std=("mean_reward", "std"),
            target_selection_rate_mean=("target_selection_rate", "mean"),
            target_selection_rate_std=("target_selection_rate", "std"),
            non_greedy_rate_mean=("non_greedy_rate", "mean"),
            non_greedy_rate_std=("non_greedy_rate", "std"),
            mean_selected_uncertainty_mean=("mean_selected_uncertainty", "mean"),
            mean_beta_mean=("mean_beta", "mean"),
        )
        .reset_index()
    )
    final_path = os.path.join(OUTPUT_DIR, "final_summary.csv")
    final_summary.to_csv(final_path, index=False)
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(final_summary.to_string(index=False))
    print()
    print(f"[INFO] Saved: {summary_path}")
    print(f"[INFO] Saved: {final_path}")


if __name__ == "__main__":
    main()
