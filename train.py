"""Train the reward model on Amazon Electronics sequences.

If data/processed/amazon/packed_train (etc.) exist — produced by
src/data/pack_sequences.py — they are used instead of the CSVs so the
full dataset trains in constant memory.
"""
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sequence_dataset import SequenceDataset
from src.models.transformer import RewardTransformer

# fixed seed so training runs are reproducible
torch.manual_seed(42)

TRAIN_PATH = "data/processed/amazon/train_sequences.csv"
VAL_PATH = "data/processed/amazon/val_sequences.csv"
PACKED_TRAIN = "data/processed/amazon/packed_train"
PACKED_VAL = "data/processed/amazon/packed_val"
MAPPING_PATH = "data/processed/amazon/item_mapping.json"

MAX_SEQ_LEN = 20
BATCH_SIZE = 256
D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 5
NUM_WORKERS = 0

CHECKPOINT = "checkpoints/best_transformer.pt"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prefer_packed(packed_dir, csv_path):
    if os.path.isdir(packed_dir):
        print(f"[INFO] Using packed dataset: {packed_dir}")
        return packed_dir
    return csv_path


def main():
    device = pick_device()
    print("=" * 70)
    print("Reward Model Training - Amazon Electronics")
    print("=" * 70)
    print(f"\n[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        num_items = json.load(f)["num_items"]
    print(f"\n[INFO] Items: {num_items:,}")

    train_ds = SequenceDataset(
        prefer_packed(PACKED_TRAIN, TRAIN_PATH), MAPPING_PATH, max_seq_len=MAX_SEQ_LEN
    )
    val_ds = SequenceDataset(
        prefer_packed(PACKED_VAL, VAL_PATH), MAPPING_PATH, max_seq_len=MAX_SEQ_LEN
    )
    print(f"[INFO] Train samples: {len(train_ds):,}")
    print(f"[INFO] Val samples:   {len(val_ds):,}")

    pin = device.type == "cuda"  # pin_memory only helps CUDA
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )

    model = RewardTransformer(
        num_items=num_items,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        padding_idx=0,
    ).to(device)
    print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            hist = batch["history_items"].to(device)
            mask = batch["attention_mask"].to(device)
            item = batch["target_item"].to(device)
            rew = batch["target_reward"].to(device)

            loss = criterion(model(hist_items=hist, attn_mask=mask, tgt_item=item), rew)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}")
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                hist = batch["history_items"].to(device)
                mask = batch["attention_mask"].to(device)
                item = batch["target_item"].to(device)
                rew = batch["target_reward"].to(device)
                val_loss += criterion(
                    model(hist_items=hist, attn_mask=mask, tgt_item=item), rew
                ).item()
        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch}/{EPOCHS}  train {train_loss:.6f}  val {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, CHECKPOINT)
            print("[INFO] Best model saved.")

    print(f"\n[INFO] Best val loss: {best_val_loss:.6f}")
    print(f"[INFO] Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
