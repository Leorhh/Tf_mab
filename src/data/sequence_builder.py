import os
import json
import pandas as pd

INPUT_PATH = "data/processed/amazon/interactions_5core.csv"
OUTPUT_DIR = "data/processed/amazon"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train_sequences.csv")
VAL_PATH = os.path.join(OUTPUT_DIR, "val_sequences.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test_sequences.csv")
STATS_PATH = os.path.join(OUTPUT_DIR, "sequence_split_stats.json")
MAX_SEQ_LEN = 20


def normalize_reward(rating):
    return (rating - 1.0) / 4.0


def make_sample(uid, item_list, rew_list, tgt_idx):
    st_idx = max(0, tgt_idx - MAX_SEQ_LEN)
    hist = item_list[st_idx:tgt_idx]
    return {
        "user_id": uid,
        "history_items": json.dumps(hist),
        "target_item": item_list[tgt_idx],
        "target_reward": rew_list[tgt_idx],
        "history_length": len(hist),
    }


def main():
    print("=" * 70)
    print("Amazon Electronics Sequence Builder")
    print("Chronological Train / Validation / Test Split")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(
        INPUT_PATH,
        usecols=[
            "user_id",
            "item_id",
            "timestamp",
            "rating",
        ],
    )
    print(f"[INFO] Interactions: {len(df):,}")
    print(f"[INFO] Users: {df['user_id'].nunique():,}")
    print(f"[INFO] Items: {df['item_id'].nunique():,}")

    print("\n[2/6] Sorting by user and timestamp...")
    df = df.sort_values(
        ["user_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)
    print("[INFO] Sorting completed.")

    print("\n[3/6] Normalizing reward...")
    df["reward"] = normalize_reward(df["rating"])
    print("[INFO] Reward values:", sorted(df["reward"].unique().tolist()))

    print("\n[4/6] Building chronological splits...")
    train_samples = []
    val_samples = []
    test_samples = []
    user_cnt = 0

    for uid, grp in df.groupby("user_id", sort=False):
        item_list = grp["item_id"].tolist()
        rew_list = grp["reward"].tolist()
        n = len(item_list)
        user_cnt += 1

        test_idx = n - 1
        val_idx = n - 2

        for tgt_idx in range(1, val_idx):
            train_samples.append(make_sample(uid, item_list, rew_list, tgt_idx))

        if val_idx >= 1:
            hist_st = max(0, val_idx - MAX_SEQ_LEN)
            val_samples.append({
                "user_id": uid,
                "history_items": json.dumps(item_list[hist_st:val_idx]),
                "target_item": item_list[val_idx],
                "target_reward": rew_list[val_idx],
                "history_length": val_idx - hist_st,
            })

        if test_idx >= 1:
            hist_st = max(0, test_idx - MAX_SEQ_LEN)
            test_samples.append({
                "user_id": uid,
                "history_items": json.dumps(item_list[hist_st:test_idx]),
                "target_item": item_list[test_idx],
                "target_reward": rew_list[test_idx],
                "history_length": test_idx - hist_st,
            })

        if user_cnt % 100000 == 0:
            print(f"[INFO] Processed users: {user_cnt:,}")

    print("\n[5/6] Converting samples to DataFrames...")
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    test_df = pd.DataFrame(test_samples)
    print(f"[INFO] Train samples: {len(train_df):,}")
    print(f"[INFO] Validation samples: {len(val_df):,}")
    print(f"[INFO] Test samples: {len(test_df):,}")

    print("\n[6/6] Saving datasets...")
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    stats = {
        "dataset": "Amazon Electronics 5-core",
        "input_path": INPUT_PATH,
        "max_seq_len": MAX_SEQ_LEN,
        "split_strategy": "chronological per-user split: last interaction=test, second-last=validation, earlier interactions=train",
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
        "interactions": int(len(df)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "train_users": int(train_df["user_id"].nunique()),
        "val_users": int(val_df["user_id"].nunique()),
        "test_users": int(test_df["user_id"].nunique()),
        "train_history_mean": float(train_df["history_length"].mean()),
        "val_history_mean": float(val_df["history_length"].mean()),
        "test_history_mean": float(test_df["history_length"].mean()),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("BUILD COMPLETED")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  Train: {TRAIN_PATH}")
    print(f"  Val:   {VAL_PATH}")
    print(f"  Test:  {TEST_PATH}")
    print(f"  Stats: {STATS_PATH}")
    print("\nSample counts:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    print("\nFirst train samples:")
    print(train_df.head().to_string(index=False))
    print("\nFirst validation sample:")
    print(val_df.head(1).to_string(index=False))
    print("\nFirst test sample:")
    print(test_df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
