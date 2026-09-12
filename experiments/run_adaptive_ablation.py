import os
import json
import random
import numpy as np
import pandas as pd
import torch
from src.data.sequence_dataset import AmazonSequenceDataset
from src.data.candidate_generator import CandidateGenerator
from src.models.transformer import TransformerRewardModel
from src.models.candidate_uncertainty import score_candidates_with_uncertainty
from src.bandit.adaptive_mab import AdaptiveMAB

SEED = 42
MAX_SAMPLES = 10000
NUM_CANDIDATES = 100
MC_SAMPLES = 10
SEQUENCE_PATH = "data/processed/amazon/test_sequences.csv"
MAPPING_PATH = "data/processed/amazon/item_mapping.json"
CHECKPOINT_PATH = "checkpoints/best_transformer.pt"
OUTPUT_DIR = "outputs/experiments/adaptive_ablation"
CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "adaptive_history_weight_10000.csv"
)
JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "adaptive_history_weight_10000.json"
)
HISTORY_WEIGHTS = [0.0, 0.1, 0.2, 0.5]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(device):
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    num_items = mapping["num_items"]
    model = TransformerRewardModel(
        num_items=num_items,
        max_seq_len=20,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        padding_idx=0,
    )
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def main():
    set_seed(SEED)
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("=" * 70)
    print("Adaptive-MAB History Weight Ablation")
    print("=" * 70)
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
    print("\n[1/5] Loading dataset...")
    print("-" * 70)
    dataset = AmazonSequenceDataset(
        sequence_path=SEQUENCE_PATH,
        mapping_path=MAPPING_PATH,
        max_seq_len=20,
    )
    total_samples = min(
        len(dataset),
        MAX_SAMPLES
    )
    print(
        f"[INFO] Test samples: "
        f"{total_samples}"
    )
    print(
        f"[INFO] Candidates: "
        f"{NUM_CANDIDATES}"
    )
    print(
        f"[INFO] MC samples: "
        f"{MC_SAMPLES}"
    )
    print(
        f"[INFO] History weights: "
        f"{HISTORY_WEIGHTS}"
    )
    print("\n[2/5] Loading model...")
    print("-" * 70)
    model = load_model(device)
    print(
        "[INFO] Model loaded successfully."
    )
    candidate_generator = CandidateGenerator(
        num_items=dataset.num_items,
        num_candidates=NUM_CANDIDATES,
        seed=SEED,
    )
    bandits = {}
    for weight in HISTORY_WEIGHTS:
        bandits[weight] = AdaptiveMAB(
            beta_min=0.1,
            beta_max=1.0,
            tau=0.01,
            history_weight=weight,
        )
    results = []
    print("\n[3/5] Running ablation...")
    print("-" * 70)
    for sample_count in range(1, total_samples + 1):
        dataset_index = sample_count - 1
        sample = dataset[dataset_index]
        hist_items = (
            sample["history_items"]
            .unsqueeze(0)
            .to(device)
        )
        attn_mask = (
            sample["attention_mask"]
            .unsqueeze(0)
            .to(device)
        )
        target_item = int(
            sample["target_item"].item()
        )
        target_reward = float(
            sample["target_reward"].item()
        )
        history = (
            sample["history_items"]
            .numpy()
        )
        history = (
            history[history != 0]
            .tolist()
        )
        candidates, target_index = (
            candidate_generator.generate(
                target_item=target_item,
                history_items=history,
            )
        )
        candidates_np = np.asarray(
            candidates,
            dtype=np.int64
        )
        candidates_tensor = (
            torch.tensor(
                candidates_np,
                dtype=torch.long
            )
            .unsqueeze(0)
            .to(device)
        )
        target_index = int(target_index)
        mean_pred, uncertainty, _ = (
            score_candidates_with_uncertainty(
                model=model,
                hist_items=hist_items,
                attn_mask=attn_mask,
                candidates=candidates_tensor,
                num_samples=MC_SAMPLES,
            )
        )
        predicted_reward = (
            mean_pred[0]
            .cpu()
            .numpy()
        )
        candidate_uncertainty = (
            uncertainty[0]
            .cpu()
            .numpy()
        )
        greedy_arm = int(
            np.argmax(predicted_reward)
        )
        for weight in HISTORY_WEIGHTS:
            bandit = bandits[weight]
            selected_arm, scores, beta = (
                bandit.select_arm(
                    predicted_reward,
                    candidate_uncertainty,
                    candidates_np,
                )
            )
            selected_arm = int(selected_arm)
            selected_item = int(
                candidates_np[selected_arm]
            )
            selected_target = int(
                selected_arm == target_index
            )
            if selected_target:
                observed_reward = target_reward
            else:
                observed_reward = None
            non_greedy = int(
                selected_arm != greedy_arm
            )
            if observed_reward is not None:
                bandit.update(
                    selected_item,
                    observed_reward
                )
            results.append({
                "sample_index": dataset_index,
                "history_weight": weight,
                "selected_item": selected_item,
                "target_item": target_item,
                "target_reward": target_reward,
                "observed_reward": (
                    observed_reward
                    if observed_reward is not None
                    else np.nan
                ),
                "selected_target": selected_target,
                "non_greedy": non_greedy,
                "selected_prediction": float(
                    predicted_reward[selected_arm]
                ),
                "selected_uncertainty": float(
                    candidate_uncertainty[selected_arm]
                ),
                "beta": float(beta),
            })
        if sample_count % 100 == 0:
            print(
                f"[INFO] Processed "
                f"{sample_count}/{total_samples}"
            )
    print("\n[4/5] Saving results...")
    print("-" * 70)
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        CSV_PATH,
        index=False
    )
    print(
        f"[INFO] Results: {CSV_PATH}"
    )
    print("\n[5/5] Calculating summary...")
    print("-" * 70)
    summary = {}
    for weight in HISTORY_WEIGHTS:
        df = results_df[
            results_df["history_weight"] == weight
        ]
        valid_reward = (
            df["observed_reward"]
            .dropna()
        )
        if len(valid_reward) > 0:
            mean_reward = float(
                valid_reward.mean()
            )
        else:
            mean_reward = 0.0
        summary[str(weight)] = {
            "samples": int(len(df)),
            "history_weight": weight,
            "mean_reward": mean_reward,
            "target_selection_rate": float(
                df["selected_target"].mean()
            ),
            "non_greedy_rate": float(
                df["non_greedy"].mean()
            ),
            "mean_selected_prediction": float(
                df["selected_prediction"].mean()
            ),
            "mean_selected_uncertainty": float(
                df["selected_uncertainty"].mean()
            ),
            "mean_beta": float(
                df["beta"].mean()
            ),
        }
    with open(JSON_PATH, "w") as f:
        json.dump(
            {
                "experiment": {
                    "dataset": "Amazon Electronics",
                    "split": "test",
                    "samples": total_samples,
                    "candidates": NUM_CANDIDATES,
                    "mc_samples": MC_SAMPLES,
                    "seed": SEED,
                    "beta_min": 0.1,
                    "beta_max": 1.0,
                    "tau": 0.01,
                    "history_weights": HISTORY_WEIGHTS,
                    "feedback_protocol": (
                        "logged-target-only"
                    ),
                },
                "results": summary,
            },
            f,
            indent=2
        )
    print(
        f"[INFO] Summary: {JSON_PATH}"
    )
    print("\n" + "=" * 70)
    print("HISTORY WEIGHT ABLATION")
    print("=" * 70)
    summary_rows = []
    for weight in HISTORY_WEIGHTS:
        row = summary[str(weight)]
        summary_rows.append({
            "history_weight": weight,
            "mean_reward": row["mean_reward"],
            "target_selection_rate": (
                row["target_selection_rate"]
            ),
            "non_greedy_rate": (
                row["non_greedy_rate"]
            ),
            "mean_selected_uncertainty": (
                row["mean_selected_uncertainty"]
            ),
            "mean_beta": row["mean_beta"],
        })
    summary_df = pd.DataFrame(
        summary_rows
    ).set_index("history_weight")
    print(
        summary_df.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )
    print("\nDONE")

if __name__ == "__main__":
    main()
