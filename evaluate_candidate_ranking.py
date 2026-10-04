"""Candidate ranking evaluation: Hit@K / NDCG@10 / MRR over 100 candidates.

Each test sample is scored against 99 sampled negatives plus the true
target. Sampling is seeded per sample index so the candidate sets are
reproducible.
"""
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.sequence_dataset import SequenceDataset
from src.models.transformer import RewardTransformer
from src.models.candidate_scorer import score_candidates

DATA_PATH = "data/processed/amazon/test_sequences.csv"
MAPPING_PATH = "data/processed/amazon/item_mapping.json"
CHECKPOINT = "checkpoints/best_transformer.pt"
BATCH_SIZE = 256
NUM_CANDIDATES = 100
MAX_SEQ_LEN = 20
MAX_SAMPLES = 10000
SEED = 42


def sample_candidates(target_items, history_items, num_items,
                      num_candidates=100, start_index=0, seed=42):
    """99 negatives (excluding history and target) + the target, shuffled."""
    all_candidates = []
    for i in range(len(target_items)):
        target = int(target_items[i].item())
        seen = {int(x) for x in history_items[i].cpu().tolist() if int(x) > 0}
        rng = np.random.default_rng(seed + start_index + i)

        negatives = set()
        while len(negatives) < num_candidates - 1:
            remaining = num_candidates - 1 - len(negatives)
            draws = rng.integers(1, num_items + 1, size=remaining * 2)
            for x in draws:
                x = int(x)
                if x != target and x not in seen:
                    negatives.add(x)
                if len(negatives) == num_candidates - 1:
                    break

        candidates = list(negatives) + [target]
        rng.shuffle(candidates)
        all_candidates.append(candidates)
    return torch.tensor(all_candidates, dtype=torch.long)


def ranking_metrics(ranks):
    ranks = np.asarray(ranks)
    return {
        "Hit@1": np.mean(ranks <= 1),
        "Hit@5": np.mean(ranks <= 5),
        "Hit@10": np.mean(ranks <= 10),
        "NDCG@10": np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)),
        "MRR": np.mean(1.0 / ranks),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("=" * 70)
    print(f"100-Candidate Ranking Evaluation  (device: {device})")
    print("=" * 70)

    with open(MAPPING_PATH, "r") as f:
        num_items = json.load(f)["num_items"]

    dataset = SequenceDataset(DATA_PATH, MAPPING_PATH, max_seq_len=MAX_SEQ_LEN)
    eval_size = min(MAX_SAMPLES, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(eval_size))
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=(device.type == "cuda"))
    print(f"[INFO] Evaluation samples: {eval_size}")

    model = RewardTransformer(num_items=num_items, max_seq_len=MAX_SEQ_LEN).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ranks = []
    total_batches = math.ceil(eval_size / BATCH_SIZE)
    for batch_idx, batch in enumerate(loader):
        hist = batch["history_items"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["target_item"].to(device)

        candidates = sample_candidates(
            targets, hist, num_items,
            num_candidates=NUM_CANDIDATES,
            start_index=batch_idx * BATCH_SIZE,
            seed=SEED,
        ).to(device)

        preds = score_candidates(model, hist, mask, candidates)
        order = torch.argsort(preds, dim=1, descending=True)
        for i in range(len(targets)):
            pos = (candidates[i] == targets[i]).nonzero(as_tuple=True)[0].item()
            rank = (order[i] == pos).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
            print(f"[INFO] batch {batch_idx + 1}/{total_batches}")

    ranks = np.asarray(ranks)
    print("\n" + "=" * 70)
    print(f"Samples: {len(ranks)}   candidates/sample: {NUM_CANDIDATES}")
    print(f"\nTarget rank  mean {ranks.mean():.2f}  median {np.median(ranks):.0f}  "
          f"p90 {np.percentile(ranks, 90):.0f}  p95 {np.percentile(ranks, 95):.0f}")

    print("\nMetrics:")
    for name, value in ranking_metrics(ranks).items():
        print(f"  {name:8s} {value:.6f}")

    print("\nRank distribution:")
    for t in [1, 5, 10, 20, 50, 100]:
        print(f"  rank <= {t:3d}: {(ranks <= t).mean() * 100:.2f}%")
    print("DONE")


if __name__ == "__main__":
    main()
