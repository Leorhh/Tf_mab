"""Build per-user interaction sequences from the cleaned KuaiRand logs.

Split is chronological per user: last interaction -> test, second to
last -> val, everything before -> train.

Memory note: the cleaned CSV carries 20 columns but we only need four,
so read with usecols and downcast dtypes. On the full 27K logs this
keeps peak RAM within reach of a 16 GB machine; packing the resulting
CSVs (pack_sequences.py) takes care of the training side.
"""
import json
import os

import pandas as pd

INPUT_PATH = "data/processed/Kuairand/random_interactions.csv"
OUTPUT_DIR = "data/processed/Kuairand"
MAX_SEQ_LEN = 50

NEEDED_COLS = ["user_id", "video_id", "time_ms", "reward"]


def build_item_mapping(df):
    video_ids = sorted(df["video_id"].unique())
    item_to_idx = {str(v): i + 1 for i, v in enumerate(video_ids)}  # 0 = padding
    idx_to_item = {str(i): v for v, i in item_to_idx.items()}
    return {
        "num_items": len(video_ids),
        "padding_idx": 0,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
    }


def build_sequences(df, item_to_idx):
    train_rows, val_rows, test_rows = [], [], []
    for user_id, user_df in df.groupby("user_id", sort=False):
        user_df = user_df.sort_values("time_ms")
        items = [item_to_idx[str(v)] for v in user_df["video_id"].tolist()]
        rewards = user_df["reward"].astype(float).tolist()

        n = len(items)
        if n < 3:
            continue
        for pos in range(1, n):
            history = items[max(0, pos - MAX_SEQ_LEN):pos]
            row = {
                "user_id": user_id,
                "history_items": ",".join(map(str, history)),
                "target_item": items[pos],
                "target_reward": rewards[pos],
                "history_length": len(history),
            }
            if pos == n - 1:
                test_rows.append(row)
            elif pos == n - 2:
                val_rows.append(row)
            else:
                train_rows.append(row)
    return train_rows, val_rows, test_rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("KuaiRand Sequence Preprocessing")
    print("=" * 60)

    print("\n[1/4] Loading data (4 of 20 columns)...")
    df = pd.read_csv(
        INPUT_PATH,
        usecols=NEEDED_COLS,
        dtype={"user_id": "int32", "video_id": "int32", "time_ms": "int64", "reward": "float32"},
    )
    print(f"[INFO] Interactions: {len(df):,}")
    print(f"[INFO] Users: {df['user_id'].nunique():,}")
    print(f"[INFO] Videos: {df['video_id'].nunique():,}")

    print("\n[2/4] Sorting by user and time...")
    df = df.sort_values(["user_id", "time_ms"], kind="mergesort").reset_index(drop=True)

    print("\n[3/4] Building sequences...")
    mapping = build_item_mapping(df)
    train_rows, val_rows, test_rows = build_sequences(df, mapping["item_to_idx"])
    print(f"[INFO] Train: {len(train_rows):,}  Val: {len(val_rows):,}  Test: {len(test_rows):,}")

    print("\n[4/4] Saving...")
    pd.DataFrame(train_rows).to_csv(os.path.join(OUTPUT_DIR, "train_sequences.csv"), index=False)
    pd.DataFrame(val_rows).to_csv(os.path.join(OUTPUT_DIR, "val_sequences.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(OUTPUT_DIR, "test_sequences.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "item_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    stats = {
        "dataset": "KuaiRand",
        "source": "random_interactions.csv",
        "num_interactions": int(len(df)),
        "num_users": int(df["user_id"].nunique()),
        "num_videos": int(df["video_id"].nunique()),
        "max_seq_len": MAX_SEQ_LEN,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "split_strategy": "chronological",
        "test_strategy": "last interaction per user",
        "validation_strategy": "second-last interaction per user",
    }
    with open(os.path.join(OUTPUT_DIR, "sequence_split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nDone. Files written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
