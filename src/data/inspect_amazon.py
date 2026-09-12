import json
from pathlib import Path


def inspect_amazon(path, n=5):
    path = Path(path)

    print(f"[INFO] File: {path}")
    print(f"[INFO] Size: {path.stat().st_size / 1024**3:.2f} GB")
    print()

    with path.open("r", encoding="utf-8") as f:
        for i in range(n):
            line = f.readline()

            if not line:
                break

            try:
                data = json.loads(line)
                print(f"========== Sample {i + 1} ==========")
                print(data)
                print()
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode error at line {i + 1}: {e}")


if __name__ == "__main__":
    inspect_amazon(
        "data/raw/amazon/Electronics.json",
        n=5
    )
