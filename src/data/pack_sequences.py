"""Pack sequence CSVs into memory-mappable .npy arrays.

The CSV layout (history as a delimited string per row) forces pandas to
materialize every row in RAM, which does not fit full KuaiRand on a
16 GB machine. This script streams the CSV in chunks, encodes each
history once, and writes three flat arrays:

    history.npy  int32   (N, max_seq_len)  left-padded item indices
    target.npy   int32   (N,)              target item index
    reward.npy   float32 (N,)              target reward

Point SequenceDataset at the output directory instead of the CSV and
memory use stays flat no matter how large N gets. The encoding rules
(parse, map, truncate, left-pad) mirror SequenceDataset exactly, so a
packed dataset and its CSV source produce identical tensors.

Usage:
    python -m src.data.pack_sequences \
        data/processed/KuaiRand/train_sequences.csv \
        data/processed/KuaiRand/item_mapping.json \
        data/processed/KuaiRand/packed_train --max-seq-len 50
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

CHUNK_ROWS = 200_000


def count_rows(csv_path):
    with open(csv_path, "rb") as f:
        return sum(1 for _ in f) - 1  # minus header


def parse_history(raw):
    if pd.isna(raw):
        return []
    raw = str(raw).strip()
    if raw.startswith("["):
        return json.loads(raw)
    if not raw:
        return []
    return raw.split(",")


def make_encoder(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    item_to_idx = mapping["item_to_idx"]
    num_items = mapping["num_items"]
    padding_idx = mapping.get("padding_idx", 0)

    def convert(item_id):
        item_id = str(item_id).strip()
        if item_id in item_to_idx:
            return item_to_idx[item_id]
        mapped = int(item_id)
        if 1 <= mapped <= num_items:
            return mapped
        raise KeyError(f"Unknown item ID: {item_id}")

    return convert, padding_idx


def pack(csv_path, mapping_path, out_dir, max_seq_len):
    convert, padding_idx = make_encoder(mapping_path)
    n_rows = count_rows(csv_path)
    print(f"[pack] {n_rows:,} rows -> {out_dir} (max_seq_len={max_seq_len})")

    os.makedirs(out_dir, exist_ok=True)
    hist = np.lib.format.open_memmap(
        os.path.join(out_dir, "history.npy"), mode="w+",
        dtype=np.int32, shape=(n_rows, max_seq_len),
    )
    target = np.lib.format.open_memmap(
        os.path.join(out_dir, "target.npy"), mode="w+",
        dtype=np.int32, shape=(n_rows,),
    )
    reward = np.lib.format.open_memmap(
        os.path.join(out_dir, "reward.npy"), mode="w+",
        dtype=np.float32, shape=(n_rows,),
    )

    offset = 0
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_ROWS):
        b = len(chunk)
        buf = np.full((b, max_seq_len), padding_idx, dtype=np.int32)
        tgt = np.empty(b, dtype=np.int32)
        rew = np.empty(b, dtype=np.float32)

        for i, row in enumerate(chunk.itertuples(index=False)):
            ids = [convert(x) for x in parse_history(row.history_items)][-max_seq_len:]
            if ids:
                buf[i, max_seq_len - len(ids):] = ids
            tgt[i] = convert(row.target_item)
            rew[i] = row.target_reward

        hist[offset:offset + b] = buf
        target[offset:offset + b] = tgt
        reward[offset:offset + b] = rew
        offset += b
        print(f"[pack] {offset:,}/{n_rows:,}", end="\r")

    hist.flush(); target.flush(); reward.flush()
    print(f"\n[pack] done. {offset:,} rows written.")


def main():
    ap = argparse.ArgumentParser(description="Pack sequence CSV into memmap arrays")
    ap.add_argument("csv_path")
    ap.add_argument("mapping_path")
    ap.add_argument("out_dir")
    ap.add_argument("--max-seq-len", type=int, default=20)
    args = ap.parse_args()
    pack(args.csv_path, args.mapping_path, args.out_dir, args.max_seq_len)


if __name__ == "__main__":
    main()
