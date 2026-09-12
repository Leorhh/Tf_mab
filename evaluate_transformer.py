import os
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.sequence_dataset import AmazonSequenceDataset
from src.models.transformer import TransformerRewardModel

DATA_PATH = "data/processed/amazon/test_sequences.csv"
ITEM_MAPPING_PATH = "data/processed/amazon/item_mapping.json"
CHECKPOINT_PATH = "checkpoints/best_transformer.pt"
BATCH_SIZE = 256
NUM_WORKERS = 0


def main():
    print("=" * 70)
    print("Transformer Reward Predictor - Test Evaluation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # load test dataset
    print("\n[1/4] Loading test dataset...")
    test_dataset = AmazonSequenceDataset(
        sequence_path=DATA_PATH,
        mapping_path=ITEM_MAPPING_PATH,
        max_seq_len=20
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )
    print(f"[INFO] Test samples: {len(test_dataset):,}")
    print(f"[INFO] Test batches: {len(test_loader):,}")

    # initialize model and load checkpoint
    print("\n[2/4] Loading model...")
    with open(ITEM_MAPPING_PATH, "r", encoding="utf-8") as f:
        item_mapping = json.load(f)

    num_items = item_mapping["num_items"]
    model = TransformerRewardModel(
        num_items=num_items,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=20
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"[INFO] Number of items: {num_items:,}")
    print(f"[INFO] Checkpoint: {CHECKPOINT_PATH}")
    print("[INFO] Model loaded successfully.")

    # run evaluation loop
    print("\n[3/4] Evaluating on test set...")
    mse_criterion = nn.MSELoss(reduction="sum")
    total_sq_err = 0.0
    total_abs_err = 0.0
    total_samp = 0
    pred_sum = 0.0
    target_sum = 0.0
    pred_min = float("inf")
    pred_max = float("-inf")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            hist_items = batch["history_items"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            tgt_item = batch["target_item"].to(device)
            tgt_reward = batch["target_reward"].to(device)

            reward_pred = model(
                hist_items=hist_items,
                attn_mask=attn_mask,
                tgt_item=tgt_item
            )


            sq_err = (reward_pred - tgt_reward) ** 2
            abs_err = torch.abs(reward_pred - tgt_reward)

            total_sq_err += sq_err.sum().item()
            total_abs_err += abs_err.sum().item()
            pred_sum += reward_pred.sum().item()
            target_sum += tgt_reward.sum().item()
            total_samp += tgt_reward.size(0)

            pred_min = min(pred_min, reward_pred.min().item())
            pred_max = max(pred_max, reward_pred.max().item())

            if (batch_idx + 1) % 500 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(test_loader)}]")

    # compute metrics
    mse = total_sq_err / total_samp
    mae = total_abs_err / total_samp
    rmse = math.sqrt(mse)
    pred_mean = pred_sum / total_samp
    tgt_mean = target_sum / total_samp

    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    print(f"Test Samples:      {total_samp:,}")
    print(f"MSE:               {mse:.6f}")
    print(f"MAE:               {mae:.6f}")
    print(f"RMSE:              {rmse:.6f}")
    print(f"Prediction Mean:   {pred_mean:.6f}")
    print(f"Target Mean:       {tgt_mean:.6f}")
    print(f"Prediction Min:    {pred_min:.6f}")
    print(f"Prediction Max:    {pred_max:.6f}")
    print("=" * 70)
    print("\nDONE")


if __name__ == "__main__":
    main()
