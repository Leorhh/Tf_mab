"""Compare a fresh run's result tables against the frozen reference tables.

Float backends differ across machines (CUDA vs MPS vs CPU), so the
check is tolerance-based, not bitwise. The frozen references are read
from outputs/paper_tables_frozen/; the new tables default to
outputs/paper_tables/.

Usage:
    python compare_results.py [path/to/new_tables/]
"""
import os
import sys

import numpy as np
import pandas as pd

FROZEN_DIR = "outputs/paper_tables_frozen"

# relative tolerance per comparison; rankings/conclusions matter, last
# digits don't
RTOL = 0.02
ATOL = 1e-4


def _numeric(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", ""), errors="coerce")


def compare_table(name, frozen_path, new_path):
    old = pd.read_csv(frozen_path)
    new = pd.read_csv(new_path)
    if list(old.columns) != list(new.columns):
        return [(name, "<columns>", "MISMATCH", "columns differ")]

    key_cols = [c for c in old.columns if old[c].dtype == object]
    if key_cols:
        try:
            merged = old.merge(new, on=key_cols, suffixes=("_old", "_new"),
                               how="outer", indicator=True,
                               validate="one_to_one")
        except pd.errors.MergeError:
            # keys are not unique; fall back to sorted row alignment
            merged = None
    else:
        merged = None

    issues = []
    if merged is None:
        sort_cols = list(old.columns)
        old_s = old.sort_values(sort_cols).reset_index(drop=True)
        new_s = new.sort_values(sort_cols).reset_index(drop=True)
        if len(old_s) != len(new_s):
            issues.append((name, "<rows>", "MISMATCH",
                           f"row count {len(old_s)} vs {len(new_s)}"))
        n = min(len(old_s), len(new_s))
        merged = old_s.iloc[:n].join(
            new_s.iloc[:n], lsuffix="_old", rsuffix="_new")
        key_cols = []
    else:
        missing = int((merged["_merge"] != "both").sum())
        if missing:
            issues.append((name, "<rows>", "MISMATCH",
                           f"{missing} rows only in one table"))

    for col in old.columns:
        if col in key_cols:
            continue
        a = merged.get(f"{col}_old")
        b = merged.get(f"{col}_new")
        if a is None or b is None:
            continue
        fa, fb = _numeric(a), _numeric(b)
        mask = fa.notna() & fb.notna()
        if not mask.any():
            continue
        diff = (fa[mask] - fb[mask]).abs()
        tol = ATOL + RTOL * fa[mask].abs()
        bad = diff > tol
        if bad.any():
            issues.append((name, col, "DRIFT",
                           f"{int(bad.sum())} cells, worst {diff[bad].max():.4g}"))
    return issues or [(name, "<all>", "OK", "")]


def main(new_dir):
    all_issues = []
    for fname in sorted(os.listdir(FROZEN_DIR)):
        if not fname.endswith(".csv"):
            continue
        new_path = os.path.join(new_dir, fname)
        if not os.path.exists(new_path):
            print(f"{fname:45s} MISSING in new run")
            all_issues.append((fname, "<file>", "MISSING", ""))
            continue
        for table, col, status, note in compare_table(
            fname, os.path.join(FROZEN_DIR, fname), new_path
        ):
            print(f"{table:45s} {col:25s} {status:8s} {note}")
            if status not in ("OK",):
                all_issues.append((table, col, status, note))

    print("\n" + "=" * 70)
    if all_issues:
        print(f"{len(all_issues)} difference(s) found - review before trusting the run.")
    else:
        print("All tables match within tolerance. Backend drift is within noise.")
    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "outputs/paper_tables"))
