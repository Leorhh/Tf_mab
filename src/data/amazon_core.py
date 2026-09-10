import json
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "amazon" / "Electronics.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "amazon"
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5


def count_json_interactions(input_file):
    user_counts = Counter()
    item_counts = Counter()
    total = 0
    valid = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
                uid = rec["reviewerID"]
                iid = rec["asin"]
                user_counts[uid] += 1
                item_counts[iid] += 1
                valid += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return user_counts, item_counts, total, valid


def count_csv_interactions(input_file):
    user_counts = Counter()
    item_counts = Counter()
    total = 0
    valid = 0
    with open(input_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            try:
                uid = row["user_id"]
                iid = row["item_id"]
                user_counts[uid] += 1
                item_counts[iid] += 1
                valid += 1
            except (KeyError, TypeError):
                continue
    return user_counts, item_counts, total, valid


def count_interactions(input_file):
    ext = input_file.suffix.lower()
    if ext == ".json":
        return count_json_interactions(input_file)
    elif ext == ".csv":
        return count_csv_interactions(input_file)
    else:
        raise ValueError(f"unsupported file type: {input_file}")


def filter_interactions(input_file, output_file, min_user_interactions, min_item_interactions):
    print(f"[INFO] Counting interactions:")
    print(f"       {input_file}")
    user_counts, item_counts, total, valid = count_interactions(input_file)

    print(f"[INFO] Total lines: {total:,}")
    print(f"[INFO] Valid interactions: {valid:,}")
    print(f"[INFO] Unique users: {len(user_counts):,}")
    print(f"[INFO] Unique items: {len(item_counts):,}")

    valid_users = set()
    for user_id, cnt in user_counts.items():
        if cnt >= min_user_interactions:
            valid_users.add(user_id)
    valid_items = set()
    for item_id, cnt in item_counts.items():
        if cnt >= min_item_interactions:
            valid_items.add(item_id)

    print(f"[INFO] Valid users: {len(valid_users):,} (>= {min_user_interactions})")
    print(f"[INFO] Valid items: {len(valid_items):,} (>= {min_item_interactions})")

    kept = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["user_id", "item_id", "timestamp", "rating"])

        if input_file.suffix.lower() == ".json":
            with open(input_file, "r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        rec = json.loads(line)
                        uid = rec["reviewerID"]
                        iid = rec["asin"]
                        if uid in valid_users and iid in valid_items:
                            writer.writerow([uid, iid, rec["unixReviewTime"], rec["overall"]])
                            kept += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        else:
            with open(input_file, "r", encoding="utf-8", newline="") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    uid = row["user_id"]
                    iid = row["item_id"]
                    if uid in valid_users and iid in valid_items:
                        writer.writerow([uid, iid, row["timestamp"], row["rating"]])
                        kept += 1
    print(f"[INFO] Kept interactions: {kept:,}")
    return kept


def get_dataset_stats(input_file):
    user_set = set()
    item_set = set()
    inter_cnt = 0
    with open(input_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_set.add(row["user_id"])
            item_set.add(row["item_id"])
            inter_cnt += 1
    return {
        "interactions": inter_cnt,
        "users": len(user_set),
        "items": len(item_set)
    }


def main():
    print("=" * 70)
    print("Amazon Electronics - Iterative 5-Core Preprocessing")
    print("=" * 70)

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found:\n{RAW_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # clean old temp files
    for f in OUTPUT_DIR.glob("interactions_5core_iter*.csv"):
        f.unlink()
    final_file = OUTPUT_DIR / "interactions_5core.csv"
    if final_file.exists():
        final_file.unlink()

    current_file = RAW_FILE
    previous_interactions = None
    iteration = 0

    while True:
        iteration += 1
        print()
        print("=" * 70)
        print(f"[INFO] Filtering iteration {iteration}")
        print("=" * 70)

        output_file = OUTPUT_DIR / f"interactions_5core_iter{iteration}.csv"
        kept = filter_interactions(current_file, output_file, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)
        stats = get_dataset_stats(output_file)

        print()
        print("[INFO] Current dataset:")
        print(f"       Users:        {stats['users']:,}")
        print(f"       Items:        {stats['items']:,}")
        print(f"       Interactions: {stats['interactions']:,}")

        if previous_interactions is not None and kept == previous_interactions:
            print()
            print("[INFO] 5-core filtering converged.")
            break
        previous_interactions = kept
        current_file = output_file

    if final_file.exists():
        final_file.unlink()
    output_file.rename(final_file)

    # delete intermediate iteration files
    for f in OUTPUT_DIR.glob("interactions_5core_iter*.csv"):
        if f.exists():
            f.unlink()

    print()
    print("=" * 70)
    print("PREPROCESSING FINISHED")
    print("=" * 70)
    final_stats = get_dataset_stats(final_file)
    print(f"Users:        {final_stats['users']:,}")
    print(f"Items:        {final_stats['items']:,}")
    print(f"Interactions: {final_stats['interactions']:,}")
    print()
    print(f"[OUTPUT] {final_file}")


if __name__ == "__main__":
    main()
