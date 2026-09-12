import json
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequence_path,
        mapping_path,
        max_seq_len=20,
    ):
        self.sequence_path = sequence_path
        self.mapping_path = mapping_path
        self.max_seq_len = max_seq_len
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.item_to_idx = mapping["item_to_idx"]
        self.num_items = mapping["num_items"]
        self.padding_idx = mapping.get("padding_idx", 0)
        self.data = pd.read_csv(sequence_path)

    def __len__(self):
        return len(self.data)

    def _parse_history(self, history):
        if pd.isna(history):
            return []
        history = str(history).strip()
        if history.startswith("["):
            return json.loads(history)
        if not history:
            return []
        return history.split(",")

    def _convert_item(self, item_id):
        item_id = str(item_id).strip()
        if item_id in self.item_to_idx:
            return self.item_to_idx[item_id]
        try:
            mapped_id = int(item_id)
            if 1 <= mapped_id <= self.num_items:
                return mapped_id
        except ValueError:
            pass
        raise KeyError(f"Unknown item ID: {item_id}")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        history = self._parse_history(row["history_items"])
        history = [
            self._convert_item(item_id)
            for item_id in history
        ]
        if len(history) > self.max_seq_len:
            history = history[-self.max_seq_len:]
        history_length = len(history)
        padded_history = [self.padding_idx] * (self.max_seq_len - history_length) + history
        attention_mask = [0] * (self.max_seq_len - history_length) + [1] * history_length
        target_item = self._convert_item(row["target_item"])
        target_reward = float(row["target_reward"])
        return {
            "history_items": torch.tensor(padded_history, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "target_item": torch.tensor(target_item, dtype=torch.long),
            "target_reward": torch.tensor(target_reward, dtype=torch.float32),
            "history_length": torch.tensor(history_length, dtype=torch.long),
        }


if __name__ == "__main__":
    print("SequenceDataset test")
