import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.sequence_dataset import AmazonSequenceDataset
from src.models.transformer import TransformerRewardModel
from src.models.candidate_scorer import score_candidates

DATA_PATH = "data/processed/amazon/test_sequences.csv"
MAPPING_PATH = "data/processed/amazon/item_mapping.json"
CHECKPOINT_PATH = "checkpoints/best_transformer.pt"
BATCH_SIZE = 256
NUM_CANDIDATES = 100
MAX_SEQ_LEN = 20
MAX_SAMPLES = 10000
SEED = 42

def generate_candidates(target_items, history_items, num_items, num_candidates=100, start_index=0, seed=42):
    all_candidates = []
    for i in range(len(target_items)):
        sample_index = start_index + i
        target = int(target_items[i].item())
        history_set = {int(x) for x in history_items[i].cpu().tolist() if int(x) > 0}
        rng = np.random.default_rng(seed + sample_index)
        candidates = set()
        while len(candidates) < num_candidates - 1:
            remaining = num_candidates - 1 - len(candidates)
            sampled = rng.integers(1, num_items + 1, size=remaining * 2)
            for item in sampled:
                item = int(item)
                if item == target or item in history_set:
                    continue
                candidates.add(item)
                if len(candidates) == num_candidates - 1:
                    break
        candidates = list(candidates)
        candidates.append(target)
        rng.shuffle(candidates)
        all_candidates.append(candidates)
    return torch.tensor(all_candidates, dtype=torch.long)

def calculate_metrics(target_ranks, num_candidates):
    target_ranks = np.asarray(target_ranks)
    n = len(target_ranks)
    hit1 = np.mean(target_ranks <= 1)
    hit5 = np.mean(target_ranks <= 5)
    hit10 = np.mean(target_ranks <= 10)
    ndcg10 = np.mean(np.where(target_ranks <= 10, 1.0 / np.log2(target_ranks + 1), 0.0))
    mrr = np.mean(1.0 / target_ranks)
    return {
        "Hit@1": hit1,
        "Hit@5": hit5,
        "Hit@10": hit10,
        "NDCG@10": ndcg10,
        "MRR": mrr,
    }

def main():
    print("=" * 70)
    print("100-Candidate Ranking Evaluation")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    print("\n[1/5] Loading mapping...")
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    num_items = mapping["num_items"]
    print(f"[INFO] Number of items: {num_items}")

    print("\n[2/5] Loading dataset...")
    dataset = AmazonSequenceDataset(
        sequence_path=DATA_PATH,
        mapping_path=MAPPING_PATH,
        max_seq_len=MAX_SEQ_LEN,
    )
    eval_size = min(MAX_SAMPLES, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(eval_size))
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[INFO] Evaluation samples: {eval_size}")

    print("\n[3/5] Loading Transformer...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("[INFO] Model loaded.")

    print("\n[4/5] Evaluating ranking...")
    target_ranks = []
    total_batches = math.ceil(eval_size / BATCH_SIZE)
    for batch_idx, batch in enumerate(loader):
        hist_items = batch["history_items"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        target_items = batch["target_item"].to(device)
        start_index = batch_idx * BATCH_SIZE
        candidates = generate_candidates(
            target_items=target_items,
            history_items=hist_items,
            num_items=num_items,
            num_candidates=NUM_CANDIDATES,
            start_index=start_index,
            seed=SEED,
        ).to(device)
        predictions = score_candidates(model=model, hist_items=hist_items, attn_mask=attn_mask, candidates=candidates)
        sorted_indices = torch.argsort(predictions, dim=1, descending=True)
        for i in range(len(target_items)):
            target = target_items[i]
            target_position = (candidates[i] == target).nonzero(as_tuple=True)[0].item()
            rank = (sorted_indices[i] == target_position).nonzero(as_tuple=True)[0].item() + 1
            target_ranks.append(rank)
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx + 1 == total_batches:
            print(f"[INFO] Batch {batch_idx + 1}/{total_batches} | Samples: {len(target_ranks)}")

    print("\n[5/5] Calculating metrics...")
    metrics = calculate_metrics(target_ranks, NUM_CANDIDATES)
    ranks = np.asarray(target_ranks)

    print("\n" + "=" * 70)
    print("Ranking Results")
    print("=" * 70)
    print(f"Samples: {len(ranks)}")
    print(f"Candidates per sample: {NUM_CANDIDATES}")

    print("\nTarget Rank:")
    print(f"Mean:   {ranks.mean():.2f}")
    print(f"Median: {np.median(ranks):.2f}")
    print(f"P90:    {np.percentile(ranks, 90):.2f}")
    print(f"P95:    {np.percentile(ranks, 95):.2f}")
    print(f"Best:   {ranks.min()}")
    print(f"Worst:  {ranks.max()}")

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"{name:10s}: {value:.6f} ({value * 100:.2f}%)")

    print("\nRank Distribution:")
    for threshold in [1, 5, 10, 20, 50, 100]:
        count = np.sum(ranks <= threshold)
        print(f"Rank <= {threshold:3d}: {count:5d} ({count / len(ranks) * 100:.2f}%)")

    print("\n" + "=" * 70)
    print("Evaluation completed.")
    print("=" * 70)

if __name__ == "__main__":
    main()
