import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.sequence_dataset import AmazonSequenceDataset
from src.models.transformer import TransformerRewardModel

# Config
TRAIN_PATH = "data/processed/amazon/train_sequences.csv"
VAL_PATH = "data/processed/amazon/val_sequences.csv"
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("Transformer Reward Predictor Training")
print("=" * 70)
print(f"\n[INFO] Device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

print("\n[1/6] Loading item mapping...")
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)
num_items = mapping["num_items"]
print(f"[INFO] Number of items: {num_items:,}")

print("\n[2/6] Loading datasets...")
train_ds = AmazonSequenceDataset(
    sequence_path=TRAIN_PATH,
    mapping_path=MAPPING_PATH,
    max_seq_len=MAX_SEQ_LEN,
)
val_ds = AmazonSequenceDataset(
    sequence_path=VAL_PATH,
    mapping_path=MAPPING_PATH,
    max_seq_len=MAX_SEQ_LEN,
)
print(f"[INFO] Train samples: {len(train_ds):,}")
print(f"[INFO] Validation samples: {len(val_ds):,}")

print("\n[3/6] Creating DataLoaders...")
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

print("\n[4/6] Creating model...")
model = TransformerRewardModel(
    num_items=num_items,
    max_seq_len=MAX_SEQ_LEN,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    padding_idx=0,
)
model = model.to(device)
print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
opt = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

print("\n[5/6] Starting training...")
best_val_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        hist_items = batch["history_items"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        tgt_item = batch["target_item"].to(device)
        tgt_rew = batch["target_reward"].to(device)

        pred = model(hist_items=hist_items, attn_mask=attn_mask, tgt_item=tgt_item)
        loss = criterion(pred, tgt_rew)

        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()

        if (batch_idx + 1) % 500 == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            hist_items = batch["history_items"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            tgt_item = batch["target_item"].to(device)
            tgt_rew = batch["target_reward"].to(device)

            pred = model(hist_items=hist_items, attn_mask=attn_mask, tgt_item=tgt_item)
            loss = criterion(pred, tgt_rew)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print("\n" + "-" * 70)
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss:   {val_loss:.6f}")
    print("-" * 70)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        import os
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, "checkpoints/best_transformer.pt")
        print("[INFO] Best model saved.")

print("\n[6/6] Training completed.")
print(f"[INFO] Best validation loss: {best_val_loss:.6f}")
print("[INFO] Checkpoint: checkpoints/best_transformer.pt")
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
