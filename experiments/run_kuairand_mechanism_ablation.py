import os
import json
import random
import numpy as np
import pandas as pd
import torch
from src.data.sequence_dataset import SequenceDataset
from src.data.candidate_generator import CandidateGenerator
from src.models.transformer import TransformerRewardModel
from src.models.candidate_uncertainty import score_candidates_with_uncertainty
from src.bandit.adaptive_mab import AdaptiveMAB

SEEDS = [42, 123, 2024, 3407, 7777]
MAX_SAMPLES = 100000
NUM_CANDIDATES = 100
MC_SAMPLES = 10
MAX_SEQ_LEN = 50
SEQUENCE_PATH = "data/processed/Kuairand/test_sequences.csv"
MAPPING_PATH = "data/processed/Kuairand/item_mapping.json"
CHECKPOINT_PATH = "checkpoints/best_kuairand_transformer.pt"
OUTPUT_DIR = "outputs/experiments/kuairand_mechanism_ablation"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(device):
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    num_items = mapping["num_items"]
    model = TransformerRewardModel(
        num_items=num_items,
        max_seq_len=MAX_SEQ_LEN,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        padding_idx=0,
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model



def calculate_summary(results_df):
    rows = []
    methods = ["Transformer-Greedy", "Fixed-Beta-MAB", "Adaptive-MAB"]
    for method in methods:
        df = results_df[results_df["method"] == method]
        valid_reward = df["observed_reward"].dropna()
        rows.append({
            "method": method,
            "mean_reward": float(valid_reward.mean()) if len(valid_reward) > 0 else 0.0,
            "target_selection_rate": float(df["selected_target"].mean()),
            "non_greedy_rate": float(df["non_greedy"].mean()),
            "mean_selected_prediction": float(df["selected_prediction"].mean()),
            "mean_selected_uncertainty": float(df["selected_uncertainty"].mean()),
            "mean_beta": float(df["beta"].mean()),
        })
    return pd.DataFrame(rows)


def run_one_seed(seed, dataset, model, device):
    fixed_mab = AdaptiveMAB(
        beta_min=0.5,
        beta_max=0.5,
        tau=0.01,
        history_weight=0.1,
    )
    set_seed(seed)
    candidate_generator = CandidateGenerator(
        num_items=dataset.num_items,
        num_candidates=NUM_CANDIDATES,
        seed=seed,
    )
    adaptive_mab = AdaptiveMAB(
        beta_min=0.1,
        beta_max=0.5,
        tau=0.01,
        history_weight=0.1,
    )
    total_samples = min(len(dataset), MAX_SAMPLES)
    results = []

    for sample_index in range(total_samples):
        sample = dataset[sample_index]
        hist_items = sample["history_items"].unsqueeze(0).to(device)
        attn_mask = sample["attention_mask"].unsqueeze(0).to(device)
        target_item = int(sample["target_item"].item())
        target_reward = float(sample["target_reward"].item())
        history = sample["history_items"].numpy()
        history = history[history != 0].tolist()

        candidates, target_index = candidate_generator.generate(
            target_item=target_item,
            history_items=history,
        )
        candidates_np = np.asarray(candidates, dtype=np.int64)
        candidates_tensor = torch.tensor(candidates_np, dtype=torch.long).unsqueeze(0).to(device)

        mean_pred, uncertainty, _ = score_candidates_with_uncertainty(
            model=model,
            hist_items=hist_items,
            attn_mask=attn_mask,
            candidates=candidates_tensor,
            num_samples=MC_SAMPLES,
        )
        predicted_reward = mean_pred[0].detach().cpu().numpy()
        candidate_uncertainty = uncertainty[0].detach().cpu().numpy()
        greedy_arm = int(np.argmax(predicted_reward))

        # Transformer-Greedy
        selected_arm = greedy_arm
        selected_target = int(selected_arm == target_index)
        observed_reward = target_reward if selected_target else np.nan
        results.append({
            "seed": seed,
            "sample_index": sample_index,
            "method": "Transformer-Greedy",
            "selected_target": selected_target,
            "observed_reward": observed_reward,
            "non_greedy": 0,
            "selected_prediction": float(predicted_reward[selected_arm]),
            "selected_uncertainty": float(candidate_uncertainty[selected_arm]),
            "beta": 0.0,
        })

        # Fixed-Beta-MAB
        selected_arm, scores, fixed_beta = fixed_mab.select_arm(
            predicted_reward,
            candidate_uncertainty,
            candidates_np,
        )

        selected_arm = int(selected_arm)
        selected_item = int(candidates_np[selected_arm])
        selected_target = int(selected_arm == target_index)
        observed_reward = target_reward if selected_target else np.nan

        if selected_target:
            fixed_mab.update(
                selected_item,
                target_reward,
            )

        results.append({
            "seed": seed,
            "sample_index": sample_index,
            "method": "Fixed-Beta-MAB",
            "selected_target": selected_target,
            "observed_reward": observed_reward,
            "non_greedy": int(selected_arm != greedy_arm),
            "selected_prediction": float(predicted_reward[selected_arm]),
            "selected_uncertainty": float(candidate_uncertainty[selected_arm]),
            "beta": float(fixed_beta),
        })

        # Adaptive-MAB
        selected_arm, scores, beta = adaptive_mab.select_arm(
            predicted_reward,
            candidate_uncertainty,
            candidates_np,
        )
        selected_arm = int(selected_arm)
        selected_target = int(selected_arm == target_index)
        observed_reward = target_reward if selected_target else np.nan
        if selected_target:
            adaptive_mab.update(int(candidates_np[selected_arm]), target_reward)
        results.append({
            "seed": seed,
            "sample_index": sample_index,
            "method": "Adaptive-MAB",
            "selected_target": selected_target,
            "observed_reward": observed_reward,
            "non_greedy": int(selected_arm != greedy_arm),
            "selected_prediction": float(predicted_reward[selected_arm]),
            "selected_uncertainty": float(candidate_uncertainty[selected_arm]),
            "beta": float(beta),
        })

        if (sample_index + 1) % 5000 == 0:
            print(f"[Seed {seed}] {sample_index + 1}/{total_samples}")
    return pd.DataFrame(results)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("KuaiRand Mechanism Ablation")
    print("=" * 70)
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("[1/4] Loading dataset...")
    dataset = SequenceDataset(
        sequence_path=SEQUENCE_PATH,
        mapping_path=MAPPING_PATH,
        max_seq_len=MAX_SEQ_LEN,
    )
    total_samples = min(len(dataset), MAX_SAMPLES)
    print(f"[INFO] Samples per seed: {total_samples}")
    print(f"[INFO] Candidates: {NUM_CANDIDATES}")
    print(f"[INFO] MC samples: {MC_SAMPLES}")
    print(f"[INFO] Seeds: {SEEDS}")
    print()
    print("[2/4] Loading model...")
    model = load_model(device)
    print("[INFO] Model loaded successfully.")
    print()
    print("[3/4] Running mechanism ablation...")
    print("-" * 70)
    all_results = []
    for seed in SEEDS:
        print()
        print(f"{'=' * 20} SEED {seed} {'=' * 20}")
        seed_results = run_one_seed(seed=seed, dataset=dataset, model=model, device=device)
        all_results.append(seed_results)
        seed_path = os.path.join(OUTPUT_DIR, f"mechanism_ablation_seed_{seed}.csv")
        seed_results.to_csv(seed_path, index=False)
        print(f"[INFO] Saved: {seed_path}")
    results_df = pd.concat(all_results, ignore_index=True)
    print()
    print("[4/4] Calculating summary...")
    print("-" * 70)
    seed_summary_list = []
    for seed in SEEDS:
        seed_df = results_df[results_df["seed"] == seed]
        seed_summary = calculate_summary(seed_df)
        seed_summary.insert(0, "seed", seed)
        seed_summary_list.append(seed_summary)
    seed_summary_df = pd.concat(seed_summary_list, ignore_index=True)
    seed_summary_path = os.path.join(OUTPUT_DIR, "mechanism_ablation_seed_summary.csv")
    seed_summary_df.to_csv(seed_summary_path, index=False)

    final_rows = []
    for method in ["Transformer-Greedy", "Fixed-Beta-MAB", "Adaptive-MAB"]:
        df = seed_summary_df[seed_summary_df["method"] == method]
        final_rows.append({
            "method": method,
            "num_seeds": len(df),
            "mean_reward_mean": float(df["mean_reward"].mean()),
            "mean_reward_std": float(df["mean_reward"].std(ddof=1)),
            "target_selection_rate_mean": float(df["target_selection_rate"].mean()),
            "target_selection_rate_std": float(df["target_selection_rate"].std(ddof=1)),
            "non_greedy_rate_mean": float(df["non_greedy_rate"].mean()),
            "non_greedy_rate_std": float(df["non_greedy_rate"].std(ddof=1)),
            "mean_selected_prediction_mean": float(df["mean_selected_prediction"].mean()),
            "mean_selected_uncertainty_mean": float(df["mean_selected_uncertainty"].mean()),
            "mean_beta_mean": float(df["mean_beta"].mean()),
        })
    final_summary_df = pd.DataFrame(final_rows)
    final_csv_path = os.path.join(OUTPUT_DIR, "mechanism_ablation_summary.csv")
    final_json_path = os.path.join(OUTPUT_DIR, "mechanism_ablation_summary.json")
    final_summary_df.to_csv(final_csv_path, index=False)
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": {
                "dataset": "KuaiRand",
                "split": "test",
                "samples_per_seed": total_samples,
                "candidates": NUM_CANDIDATES,
                "mc_samples": MC_SAMPLES,
                "seeds": SEEDS,
                "adaptive_beta_min": 0.1,
                "adaptive_beta_max": 0.5,
                "adaptive_tau": 0.01,
                "adaptive_history_weight": 0.1,
                "fixed_beta": 0.5,
                "methods": ["Transformer-Greedy", "Fixed-Beta-MAB", "Adaptive-MAB"],
                "feedback_protocol": "logged-target-only",
            },
            "results": final_rows,
        }, f, indent=2, ensure_ascii=False)
    print()
    print("=" * 70)
    print("KUAIRAND MECHANISM ABLATION SUMMARY")
    print("=" * 70)
    print(final_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("[INFO] Seed summary:")
    print(seed_summary_path)
    print("[INFO] Final summary:")
    print(final_csv_path)
    print()
    print("DONE")


if __name__ == "__main__":
    main()
