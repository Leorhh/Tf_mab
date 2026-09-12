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

SEEDS = [42, 123, 2024, 3407, 7777]
MAX_SAMPLES = 100000
NUM_CANDIDATES = 100
MC_SAMPLES = 10
MAX_SEQ_LEN = 50
SEQUENCE_PATH = "data/processed/Kuairand/test_sequences.csv"
MAPPING_PATH = "data/processed/Kuairand/item_mapping.json"
CHECKPOINT_PATH = "checkpoints/best_kuairand_transformer.pt"
OUTPUT_DIR = "outputs/experiments/kuairand_ranking"
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "kuairand_ranking_summary.csv")
SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "kuairand_ranking_summary.json")


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


def run_single_seed(seed, dataset, model, device, total_samples):
    print()
    print("=" * 70)
    print(f"RUNNING SEED: {seed}")
    print("=" * 70)
    set_seed(seed)
    candidate_generator = CandidateGenerator(
        num_items=dataset.num_items,
        num_candidates=NUM_CANDIDATES,
        seed=seed,
    )
    metric_results = []
    for sample_count in range(total_samples):
        sample = dataset[sample_count]
        hist_items = sample["history_items"].unsqueeze(0).to(device)
        attn_mask = sample["attention_mask"].unsqueeze(0).to(device)
        target_item = int(sample["target_item"].item())
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
        predicted_reward = mean_pred[0].cpu().numpy()
        target_index = int(target_index)
        rank = calculate_rank(torch.tensor(predicted_reward), target_index)
        hit1, hit5, hit10, ndcg10, mrr = ranking_metrics(rank)
        metric_results.append({
            "seed": seed,
            "sample_index": sample_count,
            "target_item": target_item,
            "target_index": target_index,
            "rank": rank,
            "hit@1": hit1,
            "hit@5": hit5,
            "hit@10": hit10,
            "ndcg@10": ndcg10,
            "mrr": mrr,
        })
        if (sample_count + 1) % 1000 == 0:
            print(f"[Seed {seed}] {sample_count + 1}/{total_samples}")
    return pd.DataFrame(metric_results)


def calculate_seed_summary(results_df):
    return {
        "seed": int(results_df["seed"].iloc[0]),
        "samples": len(results_df),
        "hit@1": float(results_df["hit@1"].mean()),
        "hit@5": float(results_df["hit@5"].mean()),
        "hit@10": float(results_df["hit@10"].mean()),
        "ndcg@10": float(results_df["ndcg@10"].mean()),
        "mrr": float(results_df["mrr"].mean()),
    }


def calculate_final_summary(seed_summaries):
    df = pd.DataFrame(seed_summaries)
    metrics = ["hit@1", "hit@5", "hit@10", "ndcg@10", "mrr"]
    row = {
        "num_seeds": len(df),
        "samples_per_seed": int(df["samples"].iloc[0]),
    }
    for metric in metrics:
        row[f"{metric}_mean"] = float(df[metric].mean())
        row[f"{metric}_std"] = float(df[metric].std(ddof=1))
    return df, pd.DataFrame([row])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("KuaiRand Transformer Ranking Evaluation")
    print("=" * 70)
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("[1/4] Loading dataset...")
    print("-" * 70)
    dataset = SequenceDataset(
        sequence_path=SEQUENCE_PATH,
        mapping_path=MAPPING_PATH,
        max_seq_len=MAX_SEQ_LEN,
    )
    total_samples = min(len(dataset), MAX_SAMPLES)
    print(f"[INFO] Test samples: {total_samples}")
    print(f"[INFO] Candidates: {NUM_CANDIDATES}")
    print()
    print("[2/4] Loading model...")
    print("-" * 70)
    model = load_model(device)
    print("[INFO] Model loaded successfully.")
    print()
    print("[3/4] Running ranking evaluation...")
    print("-" * 70)
    seed_summaries = []
    for seed in SEEDS:
        results_df = run_single_seed(
            seed=seed,
            dataset=dataset,
            model=model,
            device=device,
            total_samples=total_samples,
        )
        detail_path = os.path.join(OUTPUT_DIR, f"ranking_seed_{seed}.csv")
        results_df.to_csv(detail_path, index=False)
        seed_summary = calculate_seed_summary(results_df)
        seed_summaries.append(seed_summary)
        print(f"[INFO] Saved: {detail_path}")
        print(f"[Seed {seed}] Hit@1={seed_summary['hit@1']:.4f}, Hit@5={seed_summary['hit@5']:.4f}, Hit@10={seed_summary['hit@10']:.4f}, NDCG@10={seed_summary['ndcg@10']:.4f}, MRR={seed_summary['mrr']:.4f}")
    seed_summary_df, final_summary_df = calculate_final_summary(seed_summaries)
    print()
    print("[4/4] Saving summary...")
    print("-" * 70)
    seed_summary_path = os.path.join(OUTPUT_DIR, "kuairand_ranking_seed_summary.csv")
    seed_summary_df.to_csv(seed_summary_path, index=False)
    final_summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    summary_dict = final_summary_df.to_dict(orient="records")
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Seed summary: {seed_summary_path}")
    print(f"[INFO] Final summary: {SUMMARY_CSV_PATH}")
    print()
    print("=" * 70)
    print("KUAIRAND RANKING SUMMARY")
    print("=" * 70)
    print(final_summary_df.to_string(index=False))
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
