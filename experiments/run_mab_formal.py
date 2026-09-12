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
from src.bandit.epsilon_greedy import EpsilonGreedy
from src.bandit.ucb import UCB
from src.bandit.thompson import ThompsonSampling

SEED = 42
MAX_SAMPLES = 100000
NUM_CANDIDATES = 100
MC_SAMPLES = 10
SEQUENCE_PATH = "data/processed/amazon/test_sequences.csv"
MAPPING_PATH = "data/processed/amazon/item_mapping.json"
CHECKPOINT_PATH = "checkpoints/best_transformer.pt"
OUTPUT_DIR = "outputs/experiments/mab"
CSV_PATH = os.path.join(OUTPUT_DIR,"mab_results_100000_final_config.csv")
JSON_PATH = os.path.join(OUTPUT_DIR,"mab_summary_100000_final_config.json")


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
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def calculate_rank(scores, target_index):
    order = torch.argsort(scores, descending=True)
    rank = (order == target_index).nonzero(as_tuple=True)[0].item() + 1
    return rank


def ranking_metrics(rank):
    hit1 = int(rank <= 1)
    hit5 = int(rank <= 5)
    hit10 = int(rank <= 10)
    if rank <= 10:
        ndcg10 = 1.0 / np.log2(rank + 1)
    else:
        ndcg10 = 0.0
    mrr = 1.0 / rank
    return hit1, hit5, hit10, ndcg10, mrr


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Transformer + Sequential Contextual Bandit Experiment")
    print("=" * 70)
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    print("\n[1/5] Loading dataset...")
    print("-" * 70)
    dataset = AmazonSequenceDataset(
        sequence_path=SEQUENCE_PATH,
        mapping_path=MAPPING_PATH,
        max_seq_len=20,
    )
    total_samples = min(len(dataset), MAX_SAMPLES)
    indices = range(total_samples)
    print(f"[INFO] Total test samples used: {total_samples}")
    print(f"[INFO] Candidates per sample: {NUM_CANDIDATES}")

    print("\n[2/5] Loading model...")
    print("-" * 70)
    model = load_model(device)
    print("[INFO] Model loaded successfully.")

    candidate_generator = CandidateGenerator(
        num_items=dataset.num_items,
        num_candidates=NUM_CANDIDATES,
        seed=SEED,
    )

    methods = [
        "Random",
        "Transformer-Greedy",
        "Epsilon-Greedy",
        "UCB",
        "Thompson",
        "Adaptive-MAB",
    ]
    results = []

    epsilon_bandit = EpsilonGreedy(epsilon=0.1)
    ucb_bandit = UCB(c=1.0)
    ts_bandit = ThompsonSampling(seed=SEED)
    adaptive_bandit = AdaptiveMAB(
        beta_min=0.1,
        beta_max=0.5,
        tau=0.01,
        history_weight=0.1,
    )

    exp_rng = np.random.default_rng(SEED)
    print("\n[3/5] Running experiment...")
    print("-" * 70)

    for sample_count, dataset_index in enumerate(indices, start=1):
        sample = dataset[dataset_index]
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
        candidates = torch.tensor(candidates_np, dtype=torch.long).unsqueeze(0).to(device)
        target_index = int(target_index)

        mean_pred, uncertainty, _ = score_candidates_with_uncertainty(
            model=model,
            hist_items=hist_items,
            attn_mask=attn_mask,
            candidates=candidates,
            num_samples=MC_SAMPLES,
        )
        predicted_reward = mean_pred[0].cpu().numpy()
        candidate_uncertainty = uncertainty[0].cpu().numpy()

        greedy_arm = int(np.argmax(predicted_reward))
        strategies = {}

        strategies["Random"] = exp_rng.integers(0, NUM_CANDIDATES)
        strategies["Transformer-Greedy"] = greedy_arm
        strategies["Epsilon-Greedy"] = epsilon_bandit.select_arm(
            predicted_reward,
            candidate_uncertainty,
            candidates_np,
        )
        strategies["UCB"] = ucb_bandit.select_arm(
            predicted_reward,
            candidate_uncertainty,
            candidates_np,
        )
        strategies["Thompson"] = ts_bandit.select_arm(
            predicted_reward,
            candidate_uncertainty,
            candidates_np,
        )
        adaptive_arm, adaptive_scores, beta = adaptive_bandit.select_arm(
            predicted_reward,
            candidate_uncertainty,
            candidates_np,
        )
        strategies["Adaptive-MAB"] = adaptive_arm

        transformer_rank = calculate_rank(torch.tensor(predicted_reward), target_index)
        transformer_hit1, transformer_hit5, transformer_hit10, transformer_ndcg10, transformer_mrr = ranking_metrics(transformer_rank)

        for method, selected_arm in strategies.items():
            selected_arm = int(selected_arm)
            selected_item = int(candidates_np[selected_arm])
            selected_is_target = int(selected_arm == target_index)

            if selected_is_target:
                observed_reward = target_reward
            else:
                observed_reward = None

            selection_hit = selected_is_target
            exploration = int(selected_arm != greedy_arm)

            if observed_reward is not None:
                if method == "Epsilon-Greedy":
                    epsilon_bandit.update(selected_item, observed_reward)
                elif method == "UCB":
                    ucb_bandit.update(selected_item, observed_reward)
                elif method == "Thompson":
                    ts_bandit.update(selected_item, observed_reward)
                elif method == "Adaptive-MAB":
                    adaptive_bandit.update(selected_item, observed_reward)

            results.append({
                "sample_index": dataset_index,
                "method": method,
                "selected_item": selected_item,
                "target_item": target_item,
                "target_reward": target_reward,
                "observed_reward": observed_reward if observed_reward is not None else np.nan,
                "selected_target": selected_is_target,
                "transformer_rank": transformer_rank,
                "transformer_hit@1": transformer_hit1,
                "transformer_hit@5": transformer_hit5,
                "transformer_hit@10": transformer_hit10,
                "transformer_ndcg@10": transformer_ndcg10,
                "transformer_mrr": transformer_mrr,
                "selection_hit": selection_hit,
                "non_greedy": exploration,
                "selected_prediction": float(predicted_reward[selected_arm]),
                "selected_uncertainty": float(candidate_uncertainty[selected_arm]),
                "beta": float(beta) if method == "Adaptive-MAB" else np.nan,
            })

        if sample_count % 100 == 0:
            print(f"[INFO] Processed {sample_count}/{total_samples}")

    print("\n[4/5] Saving detailed results...")
    print("-" * 70)
    results_df = pd.DataFrame(results)
    results_df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] Results: {CSV_PATH}")

    print("\n[5/5] Calculating summary...")
    print("-" * 70)
    summary = {}
    for method in methods:
        df = results_df[results_df["method"] == method]
        valid_reward = df["observed_reward"].dropna()
        if len(valid_reward) > 0:
            mean_reward = float(valid_reward.mean())
        else:
            mean_reward = 0.0

        method_summary = {
            "samples": int(len(df)),
            "mean_reward": mean_reward,
            "target_selection_rate": float(df["selected_target"].mean()),
            "selection_hit": float(df["selection_hit"].mean()),
            "non_greedy_rate": float(df["non_greedy"].mean()),
            "mean_selected_prediction": float(df["selected_prediction"].mean()),
            "mean_selected_uncertainty": float(df["selected_uncertainty"].mean()),
        }

        if method == "Transformer-Greedy":
            method_summary.update({
                "transformer_hit@1": float(df["transformer_hit@1"].mean()),
                "transformer_hit@5": float(df["transformer_hit@5"].mean()),
                "transformer_hit@10": float(df["transformer_hit@10"].mean()),
                "transformer_ndcg@10": float(df["transformer_ndcg@10"].mean()),
                "transformer_mrr": float(df["transformer_mrr"].mean()),
            })
        if method == "Adaptive-MAB":
            method_summary["mean_beta"] = float(df["beta"].mean())
        summary[method] = method_summary

    with open(JSON_PATH, "w") as f:
        json.dump({
            "experiment": {
                "dataset": "Amazon Electronics",
                "split": "test",
                "samples": total_samples,
                "candidates": NUM_CANDIDATES,
                "mc_samples": MC_SAMPLES,
                "seed": SEED,
                "feedback_protocol": "logged-target-only",
            },
            "methods": summary,
        }, f, indent=2)
    print(f"[INFO] Summary: {JSON_PATH}")

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    summary_rows = []
    for method in methods:
        row = summary[method]
        summary_rows.append({
            "method": method,
            "mean_reward": row["mean_reward"],
            "target_selection_rate": row["target_selection_rate"],
            "non_greedy_rate": row["non_greedy_rate"],
            "mean_selected_uncertainty": row["mean_selected_uncertainty"],
        })
    summary_df = pd.DataFrame(summary_rows).set_index("method")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "-" * 70)
    print("TRANSFORMER RANKING")
    print("-" * 70)
    greedy_df = results_df[results_df["method"] == "Transformer-Greedy"]
    print(f"Hit@1    : {greedy_df['transformer_hit@1'].mean():.4f}")
    print(f"Hit@5    : {greedy_df['transformer_hit@5'].mean():.4f}")
    print(f"Hit@10   : {greedy_df['transformer_hit@10'].mean():.4f}")
    print(f"NDCG@10  : {greedy_df['transformer_ndcg@10'].mean():.4f}")
    print(f"MRR      : {greedy_df['transformer_mrr'].mean():.4f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
