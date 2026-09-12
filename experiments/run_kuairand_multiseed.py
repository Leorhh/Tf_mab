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
from src.bandit.epsilon_greedy import EpsilonGreedy
from src.bandit.ucb import UCB
from src.bandit.thompson import ThompsonSampling

SEEDS = [42, 123, 2024, 3407, 7777]
MAX_SAMPLES = 100000
NUM_CANDIDATES = 100
MC_SAMPLES = 10
MAX_SEQ_LEN = 50
SEQUENCE_PATH = "data/processed/Kuairand/test_sequences.csv"
MAPPING_PATH = "data/processed/Kuairand/item_mapping.json"
CHECKPOINT_PATH = "checkpoints/best_kuairand_transformer.pt"
OUTPUT_DIR = "outputs/experiments/kuairand_mab/multiseed"
DETAIL_DIR = os.path.join(OUTPUT_DIR, "details")
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "kuairand_mab_multiseed_summary.csv")
SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "kuairand_mab_multiseed_summary.json")


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


def create_bandits(seed):
    epsilon_bandit = EpsilonGreedy(epsilon=0.1)
    ucb_bandit = UCB(c=1.0)
    ts_bandit = ThompsonSampling(seed=seed)
    adaptive_bandit = AdaptiveMAB(beta_min=0.1, beta_max=0.5, tau=0.01, history_weight=0.1)
    return epsilon_bandit, ucb_bandit, ts_bandit, adaptive_bandit


def run_single_seed(seed, dataset, model, device, total_samples):
    print()
    print("=" * 70)
    print(f"RUNNING SEED: {seed}")
    print("=" * 70)
    set_seed(seed)
    candidate_generator = CandidateGenerator(num_items=dataset.num_items, num_candidates=NUM_CANDIDATES, seed=seed)
    epsilon_bandit, ucb_bandit, ts_bandit, adaptive_bandit = create_bandits(seed)
    exp_rng = np.random.default_rng(seed)
    methods = ["Random", "Transformer-Greedy", "Epsilon-Greedy", "UCB", "Thompson", "Adaptive-MAB"]
    results = []

    for sample_count in range(total_samples):
        sample = dataset[sample_count]
        hist_items = sample["history_items"].unsqueeze(0).to(device)
        attn_mask = sample["attention_mask"].unsqueeze(0).to(device)
        target_item = int(sample["target_item"].item())
        target_reward = float(sample["target_reward"].item())
        history = sample["history_items"].numpy()
        history = history[history != 0].tolist()

        candidates, target_index = candidate_generator.generate(target_item=target_item, history_items=history)
        candidates_np = np.asarray(candidates, dtype=np.int64)
        candidates_tensor = torch.tensor(candidates_np, dtype=torch.long).unsqueeze(0).to(device)
        target_index = int(target_index)

        mean_pred, uncertainty, _ = score_candidates_with_uncertainty(
            model=model,
            hist_items=hist_items,
            attn_mask=attn_mask,
            candidates=candidates_tensor,
            num_samples=MC_SAMPLES,
        )
        predicted_reward = mean_pred[0].cpu().numpy()
        candidate_uncertainty = uncertainty[0].cpu().numpy()
        greedy_arm = int(np.argmax(predicted_reward))

        strategies = {}
        strategies["Random"] = int(exp_rng.integers(0, NUM_CANDIDATES))
        strategies["Transformer-Greedy"] = greedy_arm
        strategies["Epsilon-Greedy"] = epsilon_bandit.select_arm(predicted_reward, candidate_uncertainty, candidates_np)
        strategies["UCB"] = ucb_bandit.select_arm(predicted_reward, candidate_uncertainty, candidates_np)
        strategies["Thompson"] = ts_bandit.select_arm(predicted_reward, candidate_uncertainty, candidates_np)
        adaptive_arm, adaptive_scores, beta = adaptive_bandit.select_arm(predicted_reward, candidate_uncertainty, candidates_np)
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
                "seed": seed,
                "sample_index": sample_count,
                "method": method,
                "selected_item": selected_item,
                "target_item": target_item,
                "target_reward": target_reward,
                "observed_reward": observed_reward if observed_reward is not None else np.nan,
                "selected_target": selected_is_target,
                "selection_hit": selected_is_target,
                "non_greedy": exploration,
                "transformer_rank": transformer_rank,
                "transformer_hit@1": transformer_hit1,
                "transformer_hit@5": transformer_hit5,
                "transformer_hit@10": transformer_hit10,
                "transformer_ndcg@10": transformer_ndcg10,
                "transformer_mrr": transformer_mrr,
                "selected_prediction": float(predicted_reward[selected_arm]),
                "selected_uncertainty": float(candidate_uncertainty[selected_arm]),
                "beta": float(beta) if method == "Adaptive-MAB" else np.nan,
            })
        if (sample_count + 1) % 1000 == 0:
            print(f"[Seed {seed}] {sample_count + 1}/{total_samples}")
    return pd.DataFrame(results)


def calculate_seed_summary(results_df):
    methods = ["Random", "Transformer-Greedy", "Epsilon-Greedy", "UCB", "Thompson", "Adaptive-MAB"]
    rows = []
    for method in methods:
        df = results_df[results_df["method"] == method]
        valid_reward = df["observed_reward"].dropna()
        if len(valid_reward) > 0:
            mean_reward = float(valid_reward.mean())
        else:
            mean_reward = 0.0
        row = {
            "seed": int(df["seed"].iloc[0]),
            "method": method,
            "mean_reward": mean_reward,
            "target_selection_rate": float(df["selected_target"].mean()),
            "non_greedy_rate": float(df["non_greedy"].mean()),
            "mean_selected_prediction": float(df["selected_prediction"].mean()),
            "mean_selected_uncertainty": float(df["selected_uncertainty"].mean()),
        }
        if method == "Adaptive-MAB":
            row["mean_beta"] = float(df["beta"].mean())
        else:
            row["mean_beta"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_final_summary(seed_summary):
    metrics = [
        "mean_reward",
        "target_selection_rate",
        "non_greedy_rate",
        "mean_selected_prediction",
        "mean_selected_uncertainty",
        "mean_beta",
    ]
    rows = []
    for method in seed_summary["method"].unique():
        df = seed_summary[seed_summary["method"] == method]
        row = {
            "method": method,
            "num_seeds": len(df),
        }
        for metric in metrics:
            values = df[metric].dropna()
            if len(values) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
            else:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1) if len(values) > 1 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DETAIL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("KuaiRand Transformer + Sequential Contextual Bandit")
    print("=" * 70)
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("[1/5] Loading dataset...")
    print("-" * 70)
    dataset = SequenceDataset(
        sequence_path=SEQUENCE_PATH,
        mapping_path=MAPPING_PATH,
        max_seq_len=MAX_SEQ_LEN,
    )
    total_samples = min(len(dataset), MAX_SAMPLES)
    print(f"[INFO] Total test samples used: {total_samples}")
    print(f"[INFO] Candidates per sample: {NUM_CANDIDATES}")
    print()
    print("[2/5] Loading model...")
    print("-" * 70)
    model = load_model(device)
    print("[INFO] Model loaded successfully.")
    all_seed_summaries = []
    print()
    print("[3/5] Running multi-seed experiment...")
    print("-" * 70)
    for seed in SEEDS:
        results_df = run_single_seed(
            seed=seed,
            dataset=dataset,
            model=model,
            device=device,
            total_samples=total_samples,
        )
        detail_path = os.path.join(DETAIL_DIR, f"kuairand_mab_seed_{seed}.csv")
        results_df.to_csv(detail_path, index=False)
        seed_summary = calculate_seed_summary(results_df)
        all_seed_summaries.append(seed_summary)
        print(f"[INFO] Saved: {detail_path}")
    seed_summary_df = pd.concat(all_seed_summaries, ignore_index=True)
    final_summary_df = calculate_final_summary(seed_summary_df)
    print()
    print("[4/5] Saving results...")
    print("-" * 70)
    seed_summary_path = os.path.join(OUTPUT_DIR, "kuairand_mab_seed_summary.csv")
    seed_summary_df.to_csv(seed_summary_path, index=False)
    final_summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    summary_dict = final_summary_df.to_dict(orient="records")
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Seed summary: {seed_summary_path}")
    print(f"[INFO] Final summary: {SUMMARY_CSV_PATH}")
    print()
    print("[5/5] Final summary")
    print("-" * 70)
    print(final_summary_df.to_string(index=False))
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
