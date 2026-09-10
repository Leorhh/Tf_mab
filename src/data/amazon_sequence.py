import csv
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "amazon" / "interactions_5core.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits" / "amazon"

SEQUENCE_LENGTH = 50
MIN_SEQUENCE_LENGTH = 3


def load_interactions():
    user_records = defaultdict(list)
    print("=" * 70)
    print("[INFO] Loading interactions")
    print("=" * 70)
    print(f"[INPUT] {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["user_id"]
            iid = row["item_id"]
            ts = int(row["timestamp"])
            score = float(row["rating"])
            user_records[uid].append((iid, ts, score))
    print(f"[INFO] Users loaded: {len(user_records):,}")
    return user_records


def sort_user_interactions(user_records):
    print()
    print("[INFO] Sorting interactions by timestamp...")
    for uid in user_records:
        user_records[uid].sort(key=lambda x: x[1])
    print("[INFO] Sorting finished.")
    return user_records


def build_item_mapping(user_records):
    item_set = set()
    for seq in user_records.values():
        for iid, _, _ in seq:
            item_set.add(iid)
    sorted_items = sorted(item_set)
    item2id = {}
    for idx, item in enumerate(sorted_items):
        item2id[item] = idx + 1
    id2item = {}
    for k, v in item2id.items():
        id2item[str(v)] = k
    print()
    print(f"[INFO] Unique items: {len(item2id):,}")
    return item2id, id2item


def create_samples(user_records, item2id):
    train = []
    val = []
    test = []
    valid_user_cnt = 0

    for uid, seq in user_records.items():
        if len(seq) < MIN_SEQUENCE_LENGTH:
            continue
        valid_user_cnt += 1
        item_seq = []
        for item, _, _ in seq:
            item_seq.append(item2id[item])

        # test sample, last item as target
        test_hist = item_seq[:-1]
        test_tgt = item_seq[-1]
        if len(test_hist) > SEQUENCE_LENGTH:
            test_hist = test_hist[-SEQUENCE_LENGTH:]
        test.append((uid, test_hist, test_tgt))

        # val sample, second last item as target
        val_hist = item_seq[:-2]
        val_tgt = item_seq[-2]
        if len(val_hist) > SEQUENCE_LENGTH:
            val_hist = val_hist[-SEQUENCE_LENGTH:]
        val.append((uid, val_hist, val_tgt))

        # train samples
        train_end = len(item_seq) - 2
        for tgt_idx in range(1, train_end):
            start = max(0, tgt_idx - SEQUENCE_LENGTH)
            hist = item_seq[start:tgt_idx]
            tgt = item_seq[tgt_idx]
            train.append((uid, hist, tgt))

    print()
    print("=" * 70)
    print("[INFO] Sample generation finished")
    print("=" * 70)
    print(f"[INFO] Users used: {valid_user_cnt:,}")
    print(f"[INFO] Train samples: {len(train):,}")
    print(f"[INFO] Valid samples: {len(val):,}")
    print(f"[INFO] Test samples:  {len(test):,}")
    return train, val, test


def save_samples(sample_list, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "history", "target"])
        for uid, hist, tgt in sample_list:
            hist_str = " ".join(map(str, hist))
            writer.writerow([uid, hist_str, tgt])


def calculate_statistics(user_records, train, val, test):
    seq_lens = []
    for seq in user_records.values():
        if len(seq) >= MIN_SEQUENCE_LENGTH:
            seq_lens.append(len(seq))
    if seq_lens:
        avg_len = sum(seq_lens) / len(seq_lens)
        max_len = max(seq_lens)
        min_len = min(seq_lens)
    else:
        avg_len = 0
        max_len = 0
        min_len = 0
    stats = {
        "num_users": len(user_records),
        "num_users_used": len(seq_lens),
        "num_train_samples": len(train),
        "num_valid_samples": len(val),
        "num_test_samples": len(test),
        "sequence_length": SEQUENCE_LENGTH,
        "min_sequence_length": MIN_SEQUENCE_LENGTH,
        "average_user_sequence_length": avg_len,
        "min_user_sequence_length": min_len,
        "max_user_sequence_length": max_len
    }
    return stats


def main():
    print("=" * 70)
    print("Amazon Electronics - Sequential Dataset Construction")
    print("=" * 70)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Processed dataset not found:\n{INPUT_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    user_records = load_interactions()
    user_records = sort_user_interactions(user_records)
    item2id, id2item = build_item_mapping(user_records)
    train_samples, valid_samples, test_samples = create_samples(user_records, item2id)

    train_file = OUTPUT_DIR / "train.csv"
    valid_file = OUTPUT_DIR / "valid.csv"
    test_file = OUTPUT_DIR / "test.csv"
    save_samples(train_samples, train_file)
    save_samples(valid_samples, valid_file)
    save_samples(test_samples, test_file)

    with open(OUTPUT_DIR / "item2id.json", "w", encoding="utf-8") as f:
        json.dump(item2id, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "id2item.json", "w", encoding="utf-8") as f:
        json.dump(id2item, f, ensure_ascii=False, indent=2)

    stats = calculate_statistics(user_records, train_samples, valid_samples, test_samples)
    with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("SEQUENTIAL DATASET CONSTRUCTION FINISHED")
    print("=" * 70)
    print(f"Users:             {stats['num_users']:,}")
    print(f"Users used:        {stats['num_users_used']:,}")
    print(f"Train samples:     {stats['num_train_samples']:,}")
    print(f"Validation samples:{stats['num_valid_samples']:,}")
    print(f"Test samples:      {stats['num_test_samples']:,}")
    print(f"Average sequence:  {stats['average_user_sequence_length']:.2f}")
    print(f"Min sequence:      {stats['min_user_sequence_length']}")
    print(f"Max sequence:      {stats['max_user_sequence_length']}")
    print()
    print(f"[OUTPUT] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
