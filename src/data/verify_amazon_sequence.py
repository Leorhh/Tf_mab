import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "splits" / "amazon"
TRAIN_FILE = SPLIT_DIR / "train.csv"
VALID_FILE = SPLIT_DIR / "valid.csv"
TEST_FILE = SPLIT_DIR / "test.csv"
ITEM2ID_FILE = SPLIT_DIR / "item2id.json"


def load_data():
    print("[INFO] Loading split files...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VALID_FILE)
    test_df = pd.read_csv(TEST_FILE)
    with open(ITEM2ID_FILE, "r", encoding="utf-8") as f:
        item_map = json.load(f)
    return train_df, val_df, test_df, item_map


def check_basic_structure(train_df, val_df, test_df, item_map):
    print("\n========== Basic Structure ==========")
    print(f"Train samples: {len(train_df):,}")
    print(f"Valid samples: {len(val_df):,}")
    print(f"Test samples : {len(test_df):,}")
    print(f"Item mapping : {len(item_map):,}")
    required_cols = {"user_id", "history", "target"}
    for tag, df in [("train", train_df), ("valid", val_df), ("test", test_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"[ERROR] {tag}.csv missing columns: {missing}")
        print(f"[OK] {tag}.csv columns correct")


def parse_history(raw_str):
    if pd.isna(raw_str) or str(raw_str).strip() == "":
        return []
    parts = str(raw_str).split()
    return [int(p) for p in parts]


def check_history_length(train_df, val_df, test_df):
    print("\n========== History Length ==========")
    for tag, df in [("train", train_df), ("valid", val_df), ("test", test_df)]:
        len_series = df["history"].apply(parse_history).apply(len)
        print(f"{tag:5s}: min={len_series.min()}, max={len_series.max()}, mean={len_series.mean():.2f}")
        if len_series.max() > 50:
            raise ValueError(f"[ERROR] {tag} contains history longer than 50")
        if len_series.min() < 1:
            raise ValueError(f"[ERROR] {tag} contains empty history")
        print(f"[OK] {tag} history length valid")


def check_user_counts(train_df, val_df, test_df):
    print("\n========== User Counts ==========")
    train_users = set(train_df["user_id"])
    val_users = set(val_df["user_id"])
    test_users = set(test_df["user_id"])
    print(f"Train users: {len(train_users):,}")
    print(f"Valid users: {len(val_users):,}")
    print(f"Test users : {len(test_users):,}")
    if train_users != val_users:
        print("[WARNING] Train and valid users are not identical")
    if train_users != test_users:
        print("[WARNING] Train and test users are not identical")
    print("[OK] User overlap checked")


def check_valid_test_one_sample(val_df, test_df):
    print("\n========== Valid/Test Sample Count ==========")
    val_group_cnt = val_df.groupby("user_id").size()
    test_group_cnt = test_df.groupby("user_id").size()
    bad_val = val_group_cnt[val_group_cnt != 1]
    bad_test = test_group_cnt[test_group_cnt != 1]
    print(f"Users with !=1 valid sample: {len(bad_val)}")
    print(f"Users with !=1 test sample : {len(bad_test)}")
    if len(bad_val) > 0:
        raise ValueError("[ERROR] Some users do not have exactly one validation sample")
    if len(bad_test) > 0:
        raise ValueError("[ERROR] Some users do not have exactly one test sample")
    print("[OK] Every user has exactly one validation sample")
    print("[OK] Every user has exactly one test sample")


def check_item_ids(train_df, val_df, test_df, item_map):
    print("\n========== Item ID Check ==========")
    max_item = len(item_map)
    for tag, df in [("train", train_df), ("valid", val_df), ("test", test_df)]:
        invalid = 0
        for _, row in df.iterrows():
            hist = parse_history(row["history"])
            tgt = int(row["target"])
            all_ids = hist + [tgt]
            for idx in all_ids:
                if idx <= 0 or idx > max_item:
                    invalid += 1
        print(f"{tag:5s}: invalid item IDs = {invalid}")
        if invalid > 0:
            raise ValueError(f"[ERROR] {tag} contains invalid item IDs")
        print(f"[OK] {tag} item IDs valid")


def check_duplicate_targets(train_df, val_df, test_df):
    print("\n========== Target Check ==========")
    for tag, df in [("train", train_df), ("valid", val_df), ("test", test_df)]:
        dup_cnt = 0
        for _, row in df.iterrows():
            hist = parse_history(row["history"])
            tgt = int(row["target"])
            if tgt in hist:
                dup_cnt += 1
        print(f"{tag:5s}: target already in history = {dup_cnt:,}")
        if dup_cnt > 0:
            print("[WARNING] Some targets already appeared in their history.")
            print("This is not necessarily an error because a user can interact with the same item multiple times.")
        else:
            print(f"[OK] No target-history duplicates in {tag}")


def inspect_examples(train_df, val_df, test_df):
    print("\n========== Examples ==========")
    print("\n--- Train ---")
    print(train_df.head(3).to_string(index=False))
    print("\n--- Valid ---")
    print(val_df.head(3).to_string(index=False))
    print("\n--- Test ---")
    print(test_df.head(3).to_string(index=False))


def main():
    print("==============================================")
    print(" Amazon Sequential Dataset Verification")
    print("==============================================")
    train_df, val_df, test_df, item_map = load_data()
    check_basic_structure(train_df, val_df, test_df, item_map)
    check_history_length(train_df, val_df, test_df)
    check_user_counts(train_df, val_df, test_df)
    check_valid_test_one_sample(val_df, test_df)
    check_item_ids(train_df, val_df, test_df, item_map)
    check_duplicate_targets(train_df, val_df, test_df)
    inspect_examples(train_df, val_df, test_df)
    print("\n==============================================")
    print("[SUCCESS] Dataset structural verification passed")
    print("==============================================")


if __name__ == "__main__":
    main()
