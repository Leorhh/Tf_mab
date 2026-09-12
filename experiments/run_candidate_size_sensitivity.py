import os
import json
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

# Configuration
DATASETS = {
    "Amazon": {
        "test_path": "data/processed/amazon/test_sequences.csv",
        "mapping_path": "data/processed/amazon/item_mapping.json",
        "checkpoint": "checkpoints/best_transformer.pt",
        "max_seq_len": 20,
        "max_samples": 100000,
    },
    "KuaiRand": {
        "test_path": "data/processed/Kuairand/test_sequences.csv",
        "mapping_path": "data/processed/Kuairand/item_mapping.json",
        "checkpoint": "checkpoints/best_kuairand_transformer.pt",
        "max_seq_len": 50,
        "max_samples": 27285,
    },
}
CANDIDATE_SIZES = [20, 50, 100, 200]
SEEDS = [123, 2024, 3407, 42, 7777]
MC_SAMPLES = 10
# Safe GPU inference batch size.
# If CUDA out-of-memory occurs, change this to 8.
INFERENCE_BATCH_SIZE = 16
BETA_MIN = 0.1
BETA_MAX = 0.5
TAU = 0.01
HISTORY_WEIGHT = 0.1
OUTPUT_ROOT = "outputs/experiments/candidate_size_sensitivity"
# Utilities
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
def load_model(checkpoint_path, num_items, max_seq_len, device):
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
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
def get_output_path(dataset_name, candidate_size, seed):
    dataset_dir = os.path.join(
        OUTPUT_ROOT,
        dataset_name.lower()
    )
    os.makedirs(dataset_dir, exist_ok=True)
    return os.path.join(
        dataset_dir,
        f"candidates_{candidate_size}_seed_{seed}.csv"
    )
def save_result(output_path, result):
    pd.DataFrame([result]).to_csv(
        output_path,
        index=False
    )
# Batched Candidate Evaluation
def evaluate_one_setting(
    model,
    dataset,
    candidate_size,
    seed,
    num_items,
    max_samples,
    device,
):
    candidate_generator = CandidateGenerator(
        num_items=num_items,
        num_candidates=candidate_size,
        seed=seed,
    )
    mab = AdaptiveMAB(
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        tau=TAU,
        history_weight=HISTORY_WEIGHT,
    )
    total_samples = min(len(dataset), max_samples)
    target_selected = 0
    non_greedy = 0
    selected_rewards = []
    selected_predictions = []
    selected_uncertainties = []
    beta_values = []
    num_batches = (
        total_samples + INFERENCE_BATCH_SIZE - 1
    ) // INFERENCE_BATCH_SIZE
    for batch_start in range(
        0,
        total_samples,
        INFERENCE_BATCH_SIZE
    ):
        batch_end = min(
            batch_start + INFERENCE_BATCH_SIZE,
            total_samples
        )
        batch_indices = range(
            batch_start,
            batch_end
        )
        histories = []
        masks = []
        candidates_list = []
        target_items = []
        target_rewards = []
        for idx in batch_indices:
            sample = dataset[idx]
            hist = sample["history_items"]
            mask = sample["attention_mask"]
            target_item = int(
                sample["target_item"].item()
            )
            target_reward = float(
                sample["target_reward"].item()
            )
            candidates, _ = candidate_generator.generate(
                target_item=target_item,
                history_items=hist.tolist(),
            )
            histories.append(hist)
            masks.append(mask)
            candidates_list.append(candidates)
            target_items.append(target_item)
            target_rewards.append(target_reward)
        hist_batch = torch.stack(histories).to(
            device,
            non_blocking=True
        )
        mask_batch = torch.stack(masks).to(
            device,
            non_blocking=True
        )
        candidate_batch = torch.tensor(
            np.stack(candidates_list),
            dtype=torch.long,
            device=device,
        )
        mean_predictions, uncertainties, _ = (
            score_candidates_with_uncertainty(
                model=model,
                hist_items=hist_batch,
                attn_mask=mask_batch,
                candidates=candidate_batch,
                num_samples=MC_SAMPLES,
            )
        )
        mean_predictions = mean_predictions.cpu().numpy()
        uncertainties = uncertainties.cpu().numpy()
        for local_idx in range(len(target_items)):
            candidate_items = candidates_list[local_idx]
            predictions = mean_predictions[local_idx]
            uncertainty = uncertainties[local_idx]
            selected_idx, scores, beta = mab.select_arm(
                predicted_reward=predictions,
                uncertainty=uncertainty,
                candidate_items=candidate_items,
            )
            selected_item = int(
                candidate_items[selected_idx]
            )
            greedy_idx = int(
                np.argmax(predictions)
            )
            greedy_item = int(
                candidate_items[greedy_idx]
            )
            if selected_item != greedy_item:
                non_greedy += 1
            if selected_item == target_items[local_idx]:
                target_selected += 1
                reward = target_rewards[local_idx]
                selected_rewards.append(reward)
                mab.update(
                    item_id=selected_item,
                    reward=reward,
                )
            selected_predictions.append(
                float(predictions[selected_idx])
            )
            selected_uncertainties.append(
                float(uncertainty[selected_idx])
            )
            beta_values.append(
                float(beta)
            )
        if (
            (batch_start // INFERENCE_BATCH_SIZE + 1) % 50 == 0
            or batch_end == total_samples
        ):
            progress = (
                batch_end / total_samples * 100
            )
            print(
                f"\r    Progress: "
                f"{batch_end:,}/{total_samples:,} "
                f"({progress:.1f}%)",
                end="",
                flush=True,
            )
    print()
    mean_reward = (
        float(np.mean(selected_rewards))
        if selected_rewards
        else 0.0
    )
    target_selection_rate = (
        target_selected / total_samples
    )
    non_greedy_rate = (
        non_greedy / total_samples
    )
    mean_selected_prediction = (
        float(np.mean(selected_predictions))
        if selected_predictions
        else 0.0
    )
    mean_selected_uncertainty = (
        float(np.mean(selected_uncertainties))
        if selected_uncertainties
        else 0.0
    )
    mean_beta = (
        float(np.mean(beta_values))
        if beta_values
        else 0.0
    )
    return {
        "dataset": None,
        "candidate_size": candidate_size,
        "seed": seed,
        "samples": total_samples,
        "mean_reward": mean_reward,
        "target_selection_rate": target_selection_rate,
        "non_greedy_rate": non_greedy_rate,
        "mean_selected_prediction": mean_selected_prediction,
        "mean_selected_uncertainty": mean_selected_uncertainty,
        "mean_beta": mean_beta,
    }

# Main
def main():
    print("=" * 70)
    print("Candidate Size Sensitivity - Fast Batched Version")
    print("=" * 70)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
    print(
        f"[INFO] MC Dropout samples: {MC_SAMPLES}"
    )
    print(
        f"[INFO] Inference batch size: "
        f"{INFERENCE_BATCH_SIZE}"
    )
    all_results = []
    for dataset_name, config in DATASETS.items():
        print()
        print("=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)
        with open(
            config["mapping_path"],
            "r",
            encoding="utf-8",
        ) as f:
            mapping = json.load(f)
        num_items = mapping["num_items"]
        print(
            f"[INFO] Number of items: "
            f"{num_items:,}"
        )
        dataset = SequenceDataset(
            sequence_path=config["test_path"],
            mapping_path=config["mapping_path"],
            max_seq_len=config["max_seq_len"],
        )
        print(
            f"[INFO] Test samples available: "
            f"{len(dataset):,}"
        )
        model = load_model(
            checkpoint_path=config["checkpoint"],
            num_items=num_items,
            max_seq_len=config["max_seq_len"],
            device=device,
        )
        for candidate_size in CANDIDATE_SIZES:
            print()
            print(
                "-" * 70
            )
            print(
                f"{dataset_name} | "
                f"candidates={candidate_size}"
            )
            print(
                "-" * 70
            )
            for seed in SEEDS:
                output_path = get_output_path(
                    dataset_name,
                    candidate_size,
                    seed,
                )
                if os.path.exists(output_path):
                    print(
                        f"[SKIP] "
                        f"candidates={candidate_size}, "
                        f"seed={seed}"
                    )
                    try:
                        existing = pd.read_csv(
                            output_path
                        )
                        if len(existing) > 0:
                            all_results.append(
                                existing.iloc[0].to_dict()
                            )
                    except Exception as e:
                        print(
                            f"[WARN] "
                            f"Could not read existing file: "
                            f"{e}"
                        )
                    continue
                print()
                print(
                    f"[RUN] {dataset_name} | "
                    f"candidates={candidate_size} | "
                    f"seed={seed}"
                )
                set_seed(seed)
                try:
                    result = evaluate_one_setting(
                        model=model,
                        dataset=dataset,
                        candidate_size=candidate_size,
                        seed=seed,
                        num_items=num_items,
                        max_samples=config["max_samples"],
                        device=device,
                    )
                    result["dataset"] = dataset_name
                    save_result(
                        output_path,
                        result,
                    )
                    all_results.append(result)
                    print(
                        f"[RESULT] "
                        f"reward={result['mean_reward']:.6f}, "
                        f"target={result['target_selection_rate']:.6f}, "
                        f"exploration={result['non_greedy_rate']:.6f}"
                    )
                    print(
                        f"[SAVED] {output_path}"
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print()
                        print(
                            "[ERROR] CUDA out of memory."
                        )
                        print(
                            "[ACTION] "
                            "Set INFERENCE_BATCH_SIZE = 8 "
                            "and run again."
                        )
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        raise
                    raise
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    if not all_results:
        print("[WARN] No results found.")
        return
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.drop_duplicates(
        subset=[
            "dataset",
            "candidate_size",
            "seed",
        ],
        keep="last",
    )
    summary_dir = OUTPUT_ROOT
    os.makedirs(
        summary_dir,
        exist_ok=True
    )
    raw_summary_path = os.path.join(
        summary_dir,
        "all_results.csv"
    )
    summary_df.to_csv(
        raw_summary_path,
        index=False
    )
    grouped = (
        summary_df
        .groupby(
            ["dataset", "candidate_size"]
        )
        .agg(
            num_seeds=("seed", "count"),
            mean_reward_mean=(
                "mean_reward",
                "mean"
            ),
            mean_reward_std=(
                "mean_reward",
                "std"
            ),
            target_selection_rate_mean=(
                "target_selection_rate",
                "mean"
            ),
            target_selection_rate_std=(
                "target_selection_rate",
                "std"
            ),
            non_greedy_rate_mean=(
                "non_greedy_rate",
                "mean"
            ),
            non_greedy_rate_std=(
                "non_greedy_rate",
                "std"
            ),
            mean_selected_prediction_mean=(
                "mean_selected_prediction",
                "mean"
            ),
            mean_selected_uncertainty_mean=(
                "mean_selected_uncertainty",
                "mean"
            ),
            mean_beta_mean=(
                "mean_beta",
                "mean"
            ),
        )
        .reset_index()
    )
    grouped = grouped.fillna(0.0)
    final_summary_path = os.path.join(
        summary_dir,
        "final_summary.csv"
    )
    grouped.to_csv(
        final_summary_path,
        index=False
    )
    print()
    print(
        grouped.to_string(
            index=False
        )
    )
    print()
    print(
        f"[SAVED] {raw_summary_path}"
    )
    print(
        f"[SAVED] {final_summary_path}"
    )
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
if __name__ == "__main__":
    main()

