"""Sequence dataset with two storage backends.

- CSV (one sample per row, history as a delimited string): convenient for
  small data, but pandas holds everything in RAM.
- Packed (.npy memmap): produced by pack_sequences.py, constant memory
  regardless of dataset size — this is what makes the full KuaiRand fit
  on a 16 GB machine.

Both paths yield identical tensors.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequence_path, mapping_path, max_seq_len=20):
        self.sequence_path = str(sequence_path)
        self.max_seq_len = max_seq_len
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.item_to_idx = mapping["item_to_idx"]
        self.num_items = mapping["num_items"]
        self.padding_idx = mapping.get("padding_idx", 0)

        if os.path.isdir(self.sequence_path):
            self._init_packed(self.sequence_path)
        else:
            self._packed = False
            self.data = pd.read_csv(self.sequence_path)

    def _init_packed(self, directory):
        self._packed = True
        # mmap_mode='r': pages are read from disk on demand
        self.hist = np.load(os.path.join(directory, "history.npy"), mmap_mode="r")
        self.target = np.load(os.path.join(directory, "target.npy"), mmap_mode="r")
        self.reward = np.load(os.path.join(directory, "reward.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.hist) if self._packed else len(self.data)

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

    def _encode_history(self, history):
        """Map raw ids to indices and left-pad to max_seq_len."""
        ids = [self._convert_item(x) for x in history][-self.max_seq_len:]
        n = len(ids)
        padded = [self.padding_idx] * (self.max_seq_len - n) + ids
        mask = [0] * (self.max_seq_len - n) + [1] * n
        return padded, mask, n

    def __getitem__(self, idx):
        if self._packed:
            padded = self.hist[idx].tolist()
            mask = [0 if x == self.padding_idx else 1 for x in padded]
            n = int(sum(mask))
            target_item = int(self.target[idx])
            target_reward = float(self.reward[idx])
        else:
            row = self.data.iloc[idx]
            padded, mask, n = self._encode_history(self._parse_history(row["history_items"]))
            target_item = self._convert_item(row["target_item"])
            target_reward = float(row["target_reward"])

        return {
            "history_items": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "target_item": torch.tensor(target_item, dtype=torch.long),
            "target_reward": torch.tensor(target_reward, dtype=torch.float32),
            "history_length": torch.tensor(n, dtype=torch.long),
        }


if __name__ == "__main__":
    print("SequenceDataset test")
