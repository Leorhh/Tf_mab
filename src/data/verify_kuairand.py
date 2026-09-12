from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "kuairand" / "KuaiRand-27K" / "data"
FILE_LIST = [
    "log_random_4_22_to_5_08_27k.csv",
    "log_standard_4_08_to_4_21_27k_part1.csv",
    "log_standard_4_08_to_4_21_27k_part2.csv",
    "log_standard_4_22_to_5_08_27k_part1.csv",
    "log_standard_4_22_to_5_08_27k_part2.csv",
    "user_features_27k.csv",
    "video_features_basic_27k.csv",
    "video_features_statistic_27k_part1.csv",
    "video_features_statistic_27k_part2.csv",
    "video_features_statistic_27k_part3.csv",
]


def inspect_csv(file_path):
    print("\n" + "=" * 80)
    print(file_path.name)
    print("=" * 80)
    file_size = file_path.stat().st_size / (1024 ** 3)
    print(f"Size: {file_size:.2f} GB")
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"Columns count: {len(header)}")
        print("Header fields:")
        print(header)
        sample_rows = []
        for idx, line in enumerate(reader):
            if idx < 3:
                sample_rows.append(line)
        print("\nFirst three rows:")
        for r in sample_rows:
            print(r)
    print("\n[OK] File read test passed")


def main():
    print("KuaiRand-27K Dataset Verification")
    print(f"Data folder: {DATA_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Directory not found:\n{DATA_DIR}")

    for fname in FILE_LIST:
        fp = DATA_DIR / fname
        if not fp.exists():
            print(f"\n[ERROR] Missing file: {fname}")
            continue
        inspect_csv(fp)

    print("\n" + "=" * 80)
    print("[SUCCESS] KuaiRand basic file check done")
    print("=" * 80)


if __name__ == "__main__":
    main()
