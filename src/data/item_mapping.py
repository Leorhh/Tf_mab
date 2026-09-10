import json
import os
import pandas as pd

INPUT_PATH = "data/processed/amazon/interactions_5core.csv"
OUTPUT_PATH = "data/processed/amazon/item_mapping.json"


def build_item_mapping():
    print("=" * 60)
    print("Building Amazon Item ID Mapping")
    print("=" * 60)
    print("\n[1/3] Loading item IDs...")
    df = pd.read_csv(
        INPUT_PATH,
        usecols=["item_id"],
    )
    item_ids = df["item_id"].unique().tolist()
    print(f"[INFO] Unique items: {len(item_ids):,}")

    # Reserve 0 for padding
    item2idx = {str(it): idx + 1 for idx, it in enumerate(item_ids)}
    idx2item = {str(idx): str(it) for it, idx in item2idx.items()}

    map_data = {
        "num_items": len(item_ids),
        "padding_idx": 0,
        "item_to_idx": item2idx,
        "idx_to_item": idx2item,
    }

    print("\n[2/3] Saving mapping...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(map_data, f, ensure_ascii=False)
    print(f"[INFO] Saved: {OUTPUT_PATH}")

    print("\n[3/3] Checking mapping...")
    print("First 5 mappings:")
    for it in item_ids[:5]:
        print(f"  {it} -> {item2idx[str(it)]}")

    print("\n" + "=" * 60)
    print("Mapping completed.")
    print("=" * 60)


if __name__ == "__main__":
    build_item_mapping()
