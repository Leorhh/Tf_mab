import json
import os
import pandas as pd

INPUT_PATH = "data/processed/Kuairand/random_interactions.csv"
OUTPUT_DIR = "data/processed/Kuairand"
MAX_SEQ_LEN = 50


def build_item_mapping(df):
    video_ids = sorted(df["video_id"].unique())
    item_to_idx = {
        str(video_id): idx + 1
        for idx, video_id in enumerate(video_ids)
    }
    idx_to_item = {
        str(idx): video_id
        for video_id, idx in item_to_idx.items()
    }
    return {
        "num_items": len(video_ids),
        "padding_idx": 0,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
    }


def build_sequences(df, item_to_idx):
    train_rows = []
    val_rows = []
    test_rows = []
    grouped = df.groupby("user_id", sort=False)
    for user_id, user_df in grouped:
        user_df = user_df.sort_values("time_ms")
        items = [
            item_to_idx[str(video_id)]
            for video_id in user_df["video_id"].tolist()
        ]
        rewards = user_df["reward"].astype(float).tolist()
        n = len(items)
        if n < 3:
            continue
        for target_pos in range(1, n):
            start = max(0, target_pos - MAX_SEQ_LEN)
            history = items[start:target_pos]
            target_item = items[target_pos]
            target_reward = rewards[target_pos]
            row = {
                "user_id": user_id,
                "history_items": ",".join(map(str, history)),
                "target_item": target_item,
                "target_reward": target_reward,
                "history_length": len(history),
            }
            if target_pos == n - 1:
                test_rows.append(row)
            elif target_pos == n - 2:
                val_rows.append(row)
            else:
                train_rows.append(row)
    return train_rows, val_rows, test_rows


def save_sequences(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("KuaiRand Sequence Preprocessing")
    print("=" * 60)
    print("\n[1/5] Loading data...")
    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Interactions: {len(df):,}")
    print(f"[INFO] Users: {df['user_id'].nunique():,}")
    print(f"[INFO] Videos: {df['video_id'].nunique():,}")
    print("\n[2/5] Sorting by user and time...")
    df = df.sort_values(
        ["user_id", "time_ms"],
        kind="mergesort"
    ).reset_index(drop=True)
    print("[INFO] Sorting completed.")
    print("\n[3/5] Building video mapping...")
    mapping = build_item_mapping(df)
    mapping_path = os.path.join(
        OUTPUT_DIR,
        "item_mapping.json"
    )
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            mapping,
            f,
            ensure_ascii=False,
            indent=2
        )
    print(f"[INFO] Number of videos: {mapping['num_items']:,}")
    print(f"[INFO] Padding index: {mapping['padding_idx']}")
    print("\n[4/5] Building sequences...")
    train_rows, val_rows, test_rows = build_sequences(
        df,
        mapping["item_to_idx"]
    )
    print(f"[INFO] Train samples: {len(train_rows):,}")
    print(f"[INFO] Val samples:   {len(val_rows):,}")
    print(f"[INFO] Test samples:  {len(test_rows):,}")
    print("\n[5/5] Saving files...")
    train_df = save_sequences(
        train_rows,
        os.path.join(OUTPUT_DIR, "train_sequences.csv")
    )
    val_df = save_sequences(
        val_rows,
        os.path.join(OUTPUT_DIR, "val_sequences.csv")
    )
    test_df = save_sequences(
        test_rows,
        os.path.join(OUTPUT_DIR, "test_sequences.csv")
    )
    stats = {
        "dataset": "KuaiRand",
        "source": "random_interactions.csv",
        "num_interactions": int(len(df)),
        "num_users": int(df["user_id"].nunique()),
        "num_videos": int(df["video_id"].nunique()),
        "max_seq_len": MAX_SEQ_LEN,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "split_strategy": "chronological",
        "test_strategy": "last interaction per user",
        "validation_strategy": "second-last interaction per user",
    }
    stats_path = os.path.join(
        OUTPUT_DIR,
        "sequence_split_stats.json"
    )
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2
        )
    print("\n" + "=" * 60)
    print("Preprocessing completed")
    print("=" * 60)
    print(f"Train: {len(train_df):,}")
    print(f"Val:   {len(val_df):,}")
    print(f"Test:  {len(test_df):,}")
    print("\nFiles:")
    print("  train_sequences.csv")
    print("  val_sequences.csv")
    print("  test_sequences.csv")
    print("  item_mapping.json")
    print("  sequence_split_stats.json")


if __name__ == "__main__":
    main()
