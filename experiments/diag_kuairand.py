"""Quick health check for the KuaiRand reward model.

Loads the checkpoint and reports val MSE + prediction mean.
A healthy model: pred_mean around 0.06-0.08, MSE below ~0.028.
Collapsed model: pred_mean 0.000000, MSE stuck at ~0.0326.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sequence_dataset import SequenceDataset
from src.models.transformer import RewardTransformer

VAL_PATH = "data/processed/Kuairand/val_sequences.csv"
PACKED_VAL = "data/processed/Kuairand/packed_val"
MAPPING_PATH = "data/processed/Kuairand/item_mapping.json"
CHECKPOINT = "checkpoints/best_kuairand_transformer.pt"
MAX_SEQ_LEN = 50


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        num_items = json.load(f)["num_items"]

    val_src = PACKED_VAL if os.path.isdir(PACKED_VAL) else VAL_PATH
    val_ds = SequenceDataset(val_src, MAPPING_PATH, max_seq_len=MAX_SEQ_LEN)
    loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    print(f"[INFO] Val samples: {len(val_ds):,}")

    model = RewardTransformer(
        num_items=num_items, max_seq_len=MAX_SEQ_LEN,
        d_model=128, nhead=4, num_layers=2,
        dim_feedforward=256, dropout=0.1, padding_idx=0,
    ).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    criterion = nn.MSELoss()
    total_loss, total_n = 0.0, 0
    preds = []
    with torch.no_grad():
        for batch in loader:
            hist = batch["history_items"].to(device)
            mask = batch["attention_mask"].to(device)
            tgt = batch["target_item"].to(device)
            reward = batch["target_reward"].to(device)
            out = model(hist, mask, tgt)
            total_loss += criterion(out, reward).item() * len(reward)
            total_n += len(reward)
            preds.append(out.cpu())

    preds = torch.cat(preds)
    print(f"\n[RESULT] MSE:       {total_loss / total_n:.6f}")
    print(f"[RESULT] pred_mean: {preds.mean().item():.6f}")
    print(f"[RESULT] pred_std:  {preds.std().item():.6f}")
    if preds.mean().item() < 0.01:
        print("[WARN] pred_mean near 0 — model looks collapsed, retrain needed")
    else:
        print("[OK] model looks healthy")


if __name__ == "__main__":
    main()
