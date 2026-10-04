"""Sanity check for MC Dropout uncertainty on one test batch."""
import json

import torch
from torch.utils.data import DataLoader

from src.data.sequence_dataset import SequenceDataset
from src.models.transformer import RewardTransformer
from src.models.uncertainty import predict_with_uncertainty

DATA_PATH = "data/processed/amazon/test_sequences.csv"
MAPPING_PATH = "data/processed/amazon/item_mapping.json"
CHECKPOINT = "checkpoints/best_transformer.pt"
BATCH_SIZE = 16
MC_SAMPLES = 10


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("=" * 70)
    print(f"MC Dropout Uncertainty Check  (device: {device})")
    print("=" * 70)

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        num_items = json.load(f)["num_items"]

    dataset = SequenceDataset(DATA_PATH, MAPPING_PATH, max_seq_len=20)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    batch = next(iter(loader))
    hist = batch["history_items"].to(device)
    mask = batch["attention_mask"].to(device)
    item = batch["target_item"].to(device)
    rew = batch["target_reward"].to(device)

    model = RewardTransformer(num_items=num_items, max_seq_len=20).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    mean_pred, unc, all_preds = predict_with_uncertainty(
        model, hist, mask, item, num_samples=MC_SAMPLES
    )

    for i in range(min(10, hist.size(0))):
        err = abs(mean_pred[i].item() - rew[i].item())
        draws = ", ".join(f"{x:.4f}" for x in all_preds[:, i].cpu().tolist())
        print(f"\nSample {i + 1}")
        print(f"  target {rew[i].item():.6f}   mean {mean_pred[i].item():.6f}   "
              f"unc {unc[i].item():.6f}   err {err:.6f}")
        print(f"  draws: {draws}")

    print("\n" + "=" * 70)
    print(f"uncertainty  mean {unc.mean():.6f}  min {unc.min():.6f}  max {unc.max():.6f}")
    print("DONE")


if __name__ == "__main__":
    main()
