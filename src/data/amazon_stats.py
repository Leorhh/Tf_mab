import json
from collections import Counter
from pathlib import Path


DATA_PATH = Path("data/raw/amazon/Electronics.json")


def analyze():
    user_count = Counter()
    item_count = Counter()

    total = 0

    print("[INFO] Start scanning Amazon Electronics...")
    print(f"[INFO] File: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            user_id = data.get("reviewerID")
            item_id = data.get("asin")

            if user_id is None or item_id is None:
                continue

            user_count[user_id] += 1
            item_count[item_id] += 1

            total += 1

            if total % 1_000_000 == 0:
                print(f"[INFO] Processed {total:,} reviews")

    print("\n========== Amazon Statistics ==========")
    print(f"Total valid reviews: {total:,}")
    print(f"Unique users: {len(user_count):,}")
    print(f"Unique items: {len(item_count):,}")

    print("\nUser interaction distribution:")
    print(f"  >= 5 : {sum(v >= 5 for v in user_count.values()):,}")
    print(f"  >=10 : {sum(v >= 10 for v in user_count.values()):,}")
    print(f"  >=20 : {sum(v >= 20 for v in user_count.values()):,}")

    print("\nItem interaction distribution:")
    print(f"  >= 5 : {sum(v >= 5 for v in item_count.values()):,}")
    print(f"  >=10 : {sum(v >= 10 for v in item_count.values()):,}")
    print(f"  >=20 : {sum(v >= 20 for v in item_count.values()):,}")


if __name__ == "__main__":
    analyze()
