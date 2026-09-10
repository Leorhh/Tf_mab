import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "kuairand"
STANDARD_FILE = DATA_DIR / "standard_interactions.csv"
RANDOM_FILE = DATA_DIR / "random_interactions.csv"
OUTPUT_FILE = DATA_DIR / "global_stats.json"

PROGRESS_INTERVAL = 5_000_000


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def percentile_from_counter(cnt, pct):
    if not cnt:
        return None
    total = sum(cnt.values())
    if total == 0:
        return None
    target = total * pct
    accum = 0
    for v in sorted(cnt):
        accum += cnt[v]
        if accum >= target:
            return v
    return max(cnt)


def process_file(file_path, ds_name):
    print()
    print("=" * 70)
    print(f"Processing statistics -> {ds_name}")
    print("=" * 70)
    print(f"File: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    start = time.time()
    total_rows = 0
    user_set = set()
    video_set = set()

    click_cnt = 0
    like_cnt = 0
    follow_cnt = 0
    comment_cnt = 0
    forward_cnt = 0
    longview_cnt = 0
    pos_reward_cnt = 0

    reward_counter = Counter()
    user_interact_cnt = Counter()
    video_interact_cnt = Counter()

    min_tms = None
    max_tms = None
    min_d = None
    max_d = None

    reward_sum = 0.0
    reward_sq_sum = 0.0

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1

            uid = row.get("user_id")
            vid = row.get("video_id")
            if uid:
                user_set.add(uid)
                user_interact_cnt[uid] += 1
            if vid:
                video_set.add(vid)
                video_interact_cnt[vid] += 1

            click = safe_int(row.get("is_click")) or 0
            like = safe_int(row.get("is_like")) or 0
            follow = safe_int(row.get("is_follow")) or 0
            comment = safe_int(row.get("is_comment")) or 0
            forward = safe_int(row.get("is_forward")) or 0
            longview = safe_int(row.get("long_view")) or 0

            if click:
                click_cnt += 1
            if like:
                like_cnt += 1
            if follow:
                follow_cnt += 1
            if comment:
                comment_cnt += 1
            if forward:
                forward_cnt += 1
            if longview:
                longview_cnt += 1

            rew = safe_float(row.get("reward"))
            if rew is not None:
                reward_sum += rew
                reward_sq_sum += rew * rew
                reward_counter[round(rew, 6)] += 1
                if rew > 0:
                    pos_reward_cnt += 1

            tms = safe_int(row.get("time_ms"))
            if tms is not None:
                if min_tms is None or tms < min_tms:
                    min_tms = tms
                if max_tms is None or tms > max_tms:
                    max_tms = tms
            date_str = row.get("date")
            if date_str:
                if min_d is None or date_str < min_d:
                    min_d = date_str
                if max_d is None or date_str > max_d:
                    max_d = date_str

            if total_rows % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start
                speed = total_rows / elapsed if elapsed > 0 else 0
                print(f"[{ds_name}] {total_rows:,} rows | {speed:,.0f} rows/s")

    elapsed = time.time() - start
    avg_user_inter = total_rows / len(user_set) if user_set else 0
    avg_video_inter = total_rows / len(video_set) if video_set else 0

    rew_mean = reward_sum / total_rows if total_rows > 0 else 0
    rew_var = reward_sq_sum / total_rows - rew_mean ** 2 if total_rows > 0 else 0
    rew_var = max(rew_var, 0)
    rew_std = math.sqrt(rew_var)

    stats = {
        "dataset": ds_name,
        "basic": {
            "interactions": total_rows,
            "unique_users": len(user_set),
            "unique_videos": len(video_set),
            "interactions_per_user": avg_user_inter,
            "interactions_per_video": avg_video_inter,
        },
        "behavior": {
            "click_rows": click_cnt,
            "like_rows": like_cnt,
            "follow_rows": follow_cnt,
            "comment_rows": comment_cnt,
            "forward_rows": forward_cnt,
            "long_view_rows": longview_cnt,
            "positive_reward_rows": pos_reward_cnt,
        },
        "behavior_rate": {
            "click_rate": click_cnt / total_rows if total_rows else 0,
            "like_rate": like_cnt / total_rows if total_rows else 0,
            "follow_rate": follow_cnt / total_rows if total_rows else 0,
            "comment_rate": comment_cnt / total_rows if total_rows else 0,
            "forward_rate": forward_cnt / total_rows if total_rows else 0,
            "long_view_rate": longview_cnt / total_rows if total_rows else 0,
            "positive_reward_rate": pos_reward_cnt / total_rows if total_rows else 0,
        },
        "reward": {
            "mean": rew_mean,
            "std": rew_std,
            "min": min(reward_counter) if reward_counter else None,
            "max": max(reward_counter) if reward_counter else None,
            "distribution": dict(sorted(reward_counter.items())),
        },
        "user_interactions": {
            "min": min(user_interact_cnt.values()) if user_interact_cnt else None,
            "max": max(user_interact_cnt.values()) if user_interact_cnt else None,
            "mean": avg_user_inter,
            "p50": percentile_from_counter(Counter(user_interact_cnt.values()), 0.50),
            "p90": percentile_from_counter(Counter(user_interact_cnt.values()), 0.90),
            "p95": percentile_from_counter(Counter(user_interact_cnt.values()), 0.95),
            "p99": percentile_from_counter(Counter(user_interact_cnt.values()), 0.99),
        },
        "video_interactions": {
            "min": min(video_interact_cnt.values()) if video_interact_cnt else None,
            "max": max(video_interact_cnt.values()) if video_interact_cnt else None,
            "mean": avg_video_inter,
            "p50": percentile_from_counter(Counter(video_interact_cnt.values()), 0.50),
            "p90": percentile_from_counter(Counter(video_interact_cnt.values()), 0.90),
            "p95": percentile_from_counter(Counter(video_interact_cnt.values()), 0.95),
            "p99": percentile_from_counter(Counter(video_interact_cnt.values()), 0.99),
        },
        "time": {
            "min_time_ms": min_tms,
            "max_time_ms": max_tms,
            "min_date": min_d,
            "max_date": max_d,
        },
        "processing": {
            "seconds": elapsed,
            "minutes": elapsed / 60,
        },
    }

    print()
    print("-" * 70)
    print(f"{ds_name} statistics")
    print("-" * 70)
    print(f"Interactions       : {total_rows:,}")
    print(f"Unique users       : {len(user_set):,}")
    print(f"Unique videos      : {len(video_set):,}")
    print()
    print("Behavior:")
    print(f"  Click            : {click_cnt:,} ({click_cnt / total_rows:.4%})")
    print(f"  Like             : {like_cnt:,} ({like_cnt / total_rows:.4%})")
    print(f"  Follow           : {follow_cnt:,} ({follow_cnt / total_rows:.4%})")
    print(f"  Comment          : {comment_cnt:,} ({comment_cnt / total_rows:.4%})")
    print(f"  Forward          : {forward_cnt:,} ({forward_cnt / total_rows:.4%})")
    print(f"  Long View        : {longview_cnt:,} ({longview_cnt / total_rows:.4%})")
    print(f"  Positive Reward  : {pos_reward_cnt:,} ({pos_reward_cnt / total_rows:.4%})")
    print()
    print("User interactions:")
    print(f"  Min              : {stats['user_interactions']['min']}")
    print(f"  Median           : {stats['user_interactions']['p50']}")
    print(f"  P90              : {stats['user_interactions']['p90']}")
    print(f"  P95              : {stats['user_interactions']['p95']}")
    print(f"  P99              : {stats['user_interactions']['p99']}")
    print(f"  Max              : {stats['user_interactions']['max']}")
    print()
    print("Video interactions:")
    print(f"  Min              : {stats['video_interactions']['min']}")
    print(f"  Median           : {stats['video_interactions']['p50']}")
    print(f"  P90              : {stats['video_interactions']['p90']}")
    print(f"  P95              : {stats['video_interactions']['p95']}")
    print(f"  P99              : {stats['video_interactions']['p99']}")
    print(f"  Max              : {stats['video_interactions']['max']}")
    print()
    print("Time:")
    print(f"  Date range       : {min_d} -> {max_d}")
    print()
    print(f"Processing time    : {elapsed / 60:.2f} minutes")
    return stats


def main():
    print("=" * 70)
    print("KuaiRand-27K Global Statistics")
    print("=" * 70)
    total_start = time.time()
    stat_std = process_file(STANDARD_FILE, "STANDARD")
    stat_rand = process_file(RANDOM_FILE, "RANDOM")

    out_data = {
        "standard": stat_std,
        "random": stat_rand,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - total_start
    print()
    print("=" * 70)
    print("GLOBAL STATISTICS FINISHED")
    print("=" * 70)
    print(f"Total processing time: {total_elapsed / 60:.2f} minutes")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
