"""Reward prediction metrics (MSE / MAE / RMSE) on the Amazon test set."""
import json
import math

import torch
from torch.utils.data import DataLoader

from src.data.sequence_dataset import SequenceDataset
from src.models.transformer import RewardTransformer

DATA_PATH = "data/processed/amazon/test_sequences.csv"
MAPPING_PATH = "data/processed/amazon/item_mapping.json"
CHECKPOINT = "checkpoints/best_transformer.pt"
BATCH_SIZE = 256


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("=" * 70)
    print(f"Reward Prediction - Test Evaluation  (device: {device})")
    print("=" * 70)

    test_ds = SequenceDataset(DATA_PATH, MAPPING_PATH, max_seq_len=20)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=(device.type == "cuda"))
    print(f"[INFO] Test samples: {len(test_ds):,}")

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        num_items = json.load(f)["num_items"]

    model = RewardTransformer(num_items=num_items, max_seq_len=20).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    sq_err = abs_err = pred_sum = tgt_sum = 0.0
    n = 0
    pred_lo, pred_hi = float("inf"), float("-inf")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            hist = batch["history_items"].to(device)
            mask = batch["attention_mask"].to(device)
            item = batch["target_item"].to(device)
            rew = batch["target_reward"].to(device)

            pred = model(hist_items=hist, attn_mask=mask, tgt_item=item)
            sq_err += ((pred - rew) ** 2).sum().item()
            abs_err += (pred - rew).abs().sum().item()
            pred_sum += pred.sum().item()
            tgt_sum += rew.sum().item()
            n += rew.size(0)
            pred_lo = min(pred_lo, pred.min().item())
            pred_hi = max(pred_hi, pred.max().item())

            if (i + 1) % 500 == 0:
                print(f"  batch {i + 1}/{len(loader)}")

    mse = sq_err / n
    print("\n" + "=" * 70)
    print(f"Samples:         {n:,}")
    print(f"MSE:             {mse:.6f}")
    print(f"MAE:             {abs_err / n:.6f}")
    print(f"RMSE:            {math.sqrt(mse):.6f}")
    print(f"Prediction mean: {pred_sum / n:.6f}   range [{pred_lo:.4f}, {pred_hi:.4f}]")
    print(f"Target mean:     {tgt_sum / n:.6f}")
    print("DONE")


if __name__ == "__main__":
    main()
