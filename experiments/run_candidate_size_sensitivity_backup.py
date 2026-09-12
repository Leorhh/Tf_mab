import os
import json
import numpy as np
import pandas as pd
import torch
from src.data.sequence_dataset import SequenceDataset
from src.models.transformer import TransformerRewardModel
from src.models.candidate_uncertainty import score_candidates_with_uncertainty
from src.data.candidate_generator import CandidateGenerator
from src.bandit.adaptive_mab import AdaptiveMAB

SEEDS = [42, 123, 2024, 3407, 7777]
CANDIDATE_SIZES = [20, 50, 100, 200]
MC_SAMPLES = 10
BATCH_SIZE = 256
BETA_MIN = 0.1
BETA_MAX = 0.5
TAU = 0.01
HISTORY_WEIGHT = 0.1
AMAZON_MAX_SAMPLES = 100000
KUAIRAND_MAX_SAMPLES = 27285

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, num_items


def evaluate(dataset_name, config, seed, candidate_size, device):
    print("=" * 70)
    print(f"{dataset_name} | seed={seed} | candidates={candidate_size}")
    print("=" * 70)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, num_items = load_model(config["mapping_path"], config["checkpoint"], device, config["max_seq_len"])
    dataset = SequenceDataset(
        sequence_path=config["test_path"],
        mapping_path=config["mapping_path"],
        max_seq_len=config["max_seq_len"],
    )
    max_samples = min(config["max_samples"], len(dataset))
    rng = np.random.default_rng(seed)
    mab = AdaptiveMAB(
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        tau=TAU,
        history_weight=HISTORY_WEIGHT,
    )

    rewards = []
    target_selected = []
    non_greedy = []
    selected_uncertainties = []
    betas = []

    for start in range(0, max_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, max_samples)
        batch_indices = np.arange(start, end)
        histories = []
        masks = []
        targets = []
        target_rewards = []
        for idx in batch_indices:
            sample = dataset[idx]
            histories.append(sample["history_items"])
            masks.append(sample["attention_mask"])
            targets.append(int(sample["target_item"]))
            target_rewards.append(float(sample["target_reward"]))

        hist_items = torch.stack(histories).to(device)
        attn_mask = torch.stack(masks).to(device)

        for i in range(len(batch_indices)):
            target_item = targets[i]
            target_reward = target_rewards[i]
            generator = CandidateGenerator(
                num_items=num_items,
                num_candidates=candidate_size,
                seed=seed,
            )
            candidates, target_idx = generator.generate(
                target_item=target_item,
                history_items=hist_items[i].cpu().numpy(),
            )
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device).unsqueeze(0)
            single_hist = hist_items[i:i + 1]
            single_mask = attn_mask[i:i + 1]

            mean_pred, uncertainty, _ = score_candidates_with_uncertainty(
                model=model,
                hist_items=single_hist,
                attn_mask=single_mask,
                candidates=candidate_tensor,
                num_samples=MC_SAMPLES,
            )
            mean_pred = mean_pred[0].cpu().numpy()
            uncertainty = uncertainty[0].cpu().numpy()

            selected_idx, scores, beta = mab.select_arm(
                predicted_reward=mean_pred,
                uncertainty=uncertainty,
                candidate_items=candidates,
            )
            selected_item = int(candidates[selected_idx])
            greedy_idx = int(np.argmax(mean_pred))

            selected_uncertainties.append(float(uncertainty[selected_idx]))
            betas.append(float(beta))
            is_target = selected_item == target_item
            target_selected.append(1 if is_target else 0)
            non_greedy.append(1 if selected_idx != greedy_idx else 0)

            if is_target:
                rewards.append(target_reward)
                mab.update(selected_item, target_reward)

    result = {
        "dataset": dataset_name,
        "seed": seed,
        "candidate_size": candidate_size,
        "samples": max_samples,
        "mean_reward": float(np.mean(rewards)),
        "target_selection_rate": float(np.mean(target_selected)),
        "non_greedy_rate": float(np.mean(non_greedy)),
        "mean_selected_uncertainty": float(np.mean(selected_uncertainties)),
        "mean_beta": float(np.mean(betas)),
    }
    print(f"[RESULT] reward={result['mean_reward']:.6f}, target={result['target_selection_rate']:.6f}, exploration={result['non_greedy_rate']:.6f}")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Candidate Size Sensitivity")
    print("=" * 70)
    print(f"[INFO] Device: {device}")

    output_dir = "outputs/experiments/candidate_size_sensitivity"
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for dataset_name, config in DATASETS.items():
        dataset_dir = os.path.join(output_dir, dataset_name.lower())
        os.makedirs(dataset_dir, exist_ok=True)
        for candidate_size in CANDIDATE_SIZES:
            for seed in SEEDS:
                raw_path = os.path.join(dataset_dir, f"candidates_{candidate_size}_seed_{seed}.csv")
                if os.path.exists(raw_path):
                    print(f"[INFO] Skip existing: {raw_path}")
                    existing = pd.read_csv(raw_path)
                    all_results.extend(existing.to_dict(orient="records"))
                    continue
                result = evaluate(
                    dataset_name=dataset_name,
                    config=config,
                    seed=seed,
                    candidate_size=candidate_size,
                    device=device,
                )
                pd.DataFrame([result]).to_csv(raw_path, index=False)
                all_results.append(result)

    raw_df = pd.DataFrame(all_results)
    raw_output = os.path.join(output_dir, "raw_results.csv")
    raw_df.to_csv(raw_output, index=False)

    summary = raw_df.groupby(["dataset", "candidate_size"]).agg(
        num_seeds=("seed", "count"),
        mean_reward_mean=("mean_reward", "mean"),
        mean_reward_std=("mean_reward", "std"),
        target_selection_rate_mean=("target_selection_rate", "mean"),
        target_selection_rate_std=("target_selection_rate", "std"),
        non_greedy_rate_mean=("non_greedy_rate", "mean"),
        non_greedy_rate_std=("non_greedy_rate", "std"),
        mean_selected_uncertainty_mean=("mean_selected_uncertainty", "mean"),
        mean_beta_mean=("mean_beta", "mean"),
    ).reset_index()

    summary_output = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_output, index=False)

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()
    print(f"[INFO] Saved: {raw_output}")
    print(f"[INFO] Saved: {summary_output}")


if __name__ == "__main__":
    main()
