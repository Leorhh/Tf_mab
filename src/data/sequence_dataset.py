import json
import pandas as pd
import torch
from torch.utils.data import Dataset


class AmazonSequenceDataset(Dataset):
    def __init__(self, sequence_path, mapping_path, max_seq_len=20):
        self.sequence_path = sequence_path
        self.max_seq_len = max_seq_len
        print("=" * 60)
        print("Loading Sequence Dataset")
        print("=" * 60)

        print("\n[1/3] Loading item mapping...")
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.item2idx = mapping["item_to_idx"]
        self.num_items = mapping["num_items"]
        self.pad_idx = mapping["padding_idx"]
        print(f"[INFO] Number of items: {self.num_items:,}")
        print(f"[INFO] Padding index: {self.pad_idx}")

        print("\n[2/3] Loading sequence data...")
        self.df = pd.read_csv(sequence_path)
        print(f"[INFO] Samples: {len(self.df):,}")
        print(f"[INFO] Columns: {list(self.df.columns)}")

        print("\n[3/3] Validating dataset...")
        required_cols = [
            "user_id",
            "history_items",
            "target_item",
            "target_reward",
            "history_length",
        ]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        print("[INFO] Dataset validation passed.")
        print("\n" + "=" * 60)
        print("Dataset loaded successfully.")
        print("=" * 60)

    def __len__(self):
        return len(self.df)

    def _convert_item(self, item_id):
        item_id = str(item_id)
        if item_id not in self.item2idx:
            raise KeyError(f"Unknown item ID: {item_id}")
        return self.item2idx[item_id]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        history_raw = json.loads(row["history_items"])
        history_idx = [self._convert_item(it) for it in history_raw]
        history_idx = history_idx[-self.max_seq_len:]
        hist_len = len(history_idx)

        padded_hist = [self.pad_idx] * self.max_seq_len
        padded_hist[:hist_len] = history_idx

        attn_mask = [0] * self.max_seq_len
        attn_mask[:hist_len] = [1] * hist_len

        tgt_item = self._convert_item(row["target_item"])
        tgt_rew = float(row["target_reward"])

        return {
            "history_items": torch.tensor(padded_hist, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "target_item": torch.tensor(tgt_item, dtype=torch.long),
            "target_reward": torch.tensor(tgt_rew, dtype=torch.float32),
            "history_length": torch.tensor(hist_len, dtype=torch.long),
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = AmazonSequenceDataset(
        sequence_path="data/processed/amazon/train_sequences.csv",
        mapping_path="data/processed/amazon/item_mapping.json",
        max_seq_len=20,
    )
    print("\n[TEST] Dataset length:")
    print(len(ds))

    dataloader = DataLoader(
        ds,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )
    print("\n[TEST] Loading one batch...")
    batch = next(iter(dataloader))
    print("\n[TEST] Batch information:")
    for k, v in batch.items():
        print(f"{k:20s}: shape={tuple(v.shape)}, dtype={v.dtype}")

    print("\n[TEST] Batch example:")
    print("history_items[0]:", batch["history_items"][0])
    print("attention_mask[0]:", batch["attention_mask"][0])
    print("target_item[0]:", batch["target_item"][0])
    print("target_reward[0]:", batch["target_reward"][0])
    print("history_length[0]:", batch["history_length"][0])

    print("\n" + "=" * 60)
    print("DataLoader test completed.")
    print("=" * 60)
