from pathlib import Path
import csv
import json
import time

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "kuairand" / "KuaiRand-27K" / "data"
OUT_DIR = ROOT / "data" / "processed" / "kuairand"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REWARD_WEIGHTS = {
    "is_click": 0.40,
    "is_like": 0.20,
    "is_follow": 0.15,
    "is_comment": 0.10,
    "is_forward": 0.10,
    "long_view": 0.05,
}

STANDARD_FILES = [
    "log_standard_4_08_to_4_21_27k_part1.csv",
    "log_standard_4_08_to_4_21_27k_part2.csv",
    "log_standard_4_22_to_5_08_27k_part1.csv",
    "log_standard_4_22_to_5_08_27k_part2.csv",
]
RANDOM_FILE = "log_random_4_22_to_5_08_27k.csv"

OUTPUT_COLUMNS = [
    "user_id",
    "video_id",
    "time_ms",
    "date",
    "hourmin",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
    "reward",
]


def safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def calculate_reward(row_dict):
    rew = 0.0
    for k, w in REWARD_WEIGHTS.items():
        rew += w * safe_int(row_dict.get(k, 0))
    return rew


def normalize_row(raw_row):
    uid = safe_int(raw_row.get("user_id"))
    vid = safe_int(raw_row.get("video_id"))
    tms = safe_int(raw_row.get("time_ms"))

    if uid < 0 or vid < 0 or tms <= 0:
        return None

    res = {}
    res["user_id"] = uid
    res["video_id"] = vid
    res["time_ms"] = tms
    res["date"] = raw_row.get("date", "")
    res["hourmin"] = raw_row.get("hourmin", "")
    res["is_click"] = safe_int(raw_row.get("is_click"))
    res["is_like"] = safe_int(raw_row.get("is_like"))
    res["is_follow"] = safe_int(raw_row.get("is_follow"))
    res["is_comment"] = safe_int(raw_row.get("is_comment"))
    res["is_forward"] = safe_int(raw_row.get("is_forward"))
    res["is_hate"] = safe_int(raw_row.get("is_hate"))
    res["long_view"] = safe_int(raw_row.get("long_view"))
    res["play_time_ms"] = safe_int(raw_row.get("play_time_ms"))
    res["duration_ms"] = safe_int(raw_row.get("duration_ms"))
    res["profile_stay_time"] = safe_int(raw_row.get("profile_stay_time"))
    res["comment_stay_time"] = safe_int(raw_row.get("comment_stay_time"))
    res["is_profile_enter"] = safe_int(raw_row.get("is_profile_enter"))
    res["is_rand"] = safe_int(raw_row.get("is_rand"))
    res["tab"] = safe_int(raw_row.get("tab"))
    res["reward"] = calculate_reward(res)
    return res




def process_files(file_list, out_name):
    out_path = OUT_DIR / out_name
    stats = {
        "input_rows": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "positive_reward_rows": 0,
        "click_rows": 0,
        "like_rows": 0,
        "follow_rows": 0,
        "comment_rows": 0,
        "forward_rows": 0,
        "long_view_rows": 0,
    }
    print("\n" + "=" * 70)
    print(f"Processing -> {out_name}")
    print("=" * 70)

    with open(out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for idx, fname in enumerate(file_list, start=1):
            in_path = RAW_DIR / fname
            if not in_path.exists():
                raise FileNotFoundError(f"Missing file:\n{in_path}")
            print(f"\n[{idx}/{len(file_list)}] {fname}")

            with open(in_path, "r", encoding="utf-8", newline="") as f_in:
                reader = csv.DictReader(f_in)
                for line_row in reader:
                    stats["input_rows"] += 1
                    norm_data = normalize_row(line_row)
                    if norm_data is None:
                        stats["invalid_rows"] += 1
                        continue

                    writer.writerow(norm_data)
                    stats["valid_rows"] += 1

                    if norm_data["reward"] > 0:
                        stats["positive_reward_rows"] += 1
                    stats["click_rows"] += norm_data["is_click"]
                    stats["like_rows"] += norm_data["is_like"]
                    stats["follow_rows"] += norm_data["is_follow"]
                    stats["comment_rows"] += norm_data["is_comment"]
                    stats["forward_rows"] += norm_data["is_forward"]
                    stats["long_view_rows"] += norm_data["long_view"]

                    if stats["input_rows"] % 5_000_000 == 0:
                        print(f"  processed: {stats['input_rows']:,}")

    stat_file = OUT_DIR / out_name.replace(".csv", "_stats.json")
    with open(stat_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nFinished.")
    for k, v in stats.items():
        print(f"{k:25s}: {v:,}")
    print(f"\nOutput: {out_path}")
    return stats


def main():
    print("=" * 70)
    print("KuaiRand-27K Streaming Cleaning")
    print("=" * 70)
    print("\nReward configuration:")
    for k, w in REWARD_WEIGHTS.items():
        print(f"  {k:15s}: {w}")
    print("\nRaw directory:")
    print(RAW_DIR)
    print("\nOutput directory:")
    print(OUT_DIR)

    start = time.time()
    stat_std = process_files(STANDARD_FILES, "standard_interactions.csv")
    stat_rand = process_files([RANDOM_FILE], "random_interactions.csv")
    total_sec = time.time() - start

    summary_info = {
        "reward_weights": REWARD_WEIGHTS,
        "standard": stat_std,
        "random": stat_rand,
        "processing_seconds": total_sec,
    }
    summary_path = OUT_DIR / "cleaning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("CLEANING FINISHED")
    print("=" * 70)
    print(f"Processing time: {total_sec / 60:.2f} minutes")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
