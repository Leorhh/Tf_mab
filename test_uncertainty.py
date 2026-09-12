import json
import torch
from torch.utils.data import DataLoader
from src.data.sequence_dataset import AmazonSequenceDataset
from src.models.transformer import TransformerRewardModel
from src.models.uncertainty import mc_dropout_predict

DATA_PATH = "data/processed/amazon/test_sequences.csv"
ITEM_MAPPING_PATH = "data/processed/amazon/item_mapping.json"
CHECKPOINT_PATH = "checkpoints/best_transformer.pt"
BATCH_SIZE = 16
MC_SAMPLES = 10


def main():
    print("=" * 70)
    print("MC Dropout Uncertainty Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # load item mapping
    print("\n[1/4] Loading item mapping...")
    with open(ITEM_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    num_items = mapping["num_items"]
    print(f"[INFO] Number of items: {num_items:,}")

    # prepare dataset and take one batch
    print("\n[2/4] Loading test dataset...")
    dataset = AmazonSequenceDataset(
        sequence_path=DATA_PATH,
        mapping_path=ITEM_MAPPING_PATH,
        max_seq_len=20,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    batch = next(iter(loader))
    hist_items = batch["history_items"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    tgt_item = batch["target_item"].to(device)
    tgt_reward = batch["target_reward"].to(device)

    print(f"[INFO] Batch size: {hist_items.size(0)}")
    print(f"[INFO] History shape: {hist_items.shape}")

    # load transformer model and checkpoint
    print("\n[3/4] Loading trained model...")
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
    model = model.to(device)
    print("[INFO] Model loaded successfully.")

    # run mc dropout sampling
    print("\n[4/4] Running MC Dropout...")
    print(f"[INFO] MC samples: {MC_SAMPLES}")
    mean_pred, uncertainty, all_predictions = mc_dropout_predict(
        model=model,
        hist_items=hist_items,
        attn_mask=attn_mask,
        tgt_item=tgt_item,
        num_samples=MC_SAMPLES,
    )

    print("\n" + "=" * 70)
    print("MC Dropout Results")
    print("=" * 70)
    show_count = min(10, hist_items.size(0))
    for i in range(show_count):
        print(f"\nSample {i + 1}")
        print(f"Target reward:       {tgt_reward[i].item():.6f}")
        print(f"Mean prediction:     {mean_pred[i].item():.6f}")
        print(f"Uncertainty (std):   {uncertainty[i].item():.6f}")
        err = abs(mean_pred[i].item() - tgt_reward[i].item())
        print(f"Prediction error:    {err:.6f}")
        mc_vals = [f"{x:.4f}" for x in all_predictions[:, i].cpu().tolist()]
        print("MC predictions:      " + ", ".join(mc_vals))

    print("\n" + "=" * 70)
    print(f"Mean uncertainty:    {uncertainty.mean().item():.6f}")
    print(f"Min uncertainty:     {uncertainty.min().item():.6f}")
    print(f"Max uncertainty:     {uncertainty.max().item():.6f}")
    print("=" * 70)
    print("\nDONE")


if __name__ == "__main__":
    main()
