"""
Check the 8-block windows that the Milestone 3 models use, and report how many rows they cover.

INPUT:  processed_data/features/pm25_features_with_splits.parquet   (train and dev rows only)
        processed_data/three_hour_aggregated.parquet                (the gap-free block table, blocks before 2025)
OUTPUT: reports/windows/window_checks_by_station.csv   and the same tables printed. Nothing is changed.

The features file holds blocks t, t-1, t-2, t-3 and the rolling statistics only, and it has gaps (rows that were not usable
are absent), so a window cannot be cut from row order. m3_data.add_window_columns() rebuilds blocks t-4 ... t-7 from the
gap-free block table. This script runs m3_data.verify_windows(), which stops the run if the rebuilt window disagrees with
the stored features, and then prints the evidence:

  - the window of every usable row is complete (all 8 blocks valid), so the number of windows equals the number of usable
    rows (123,364 train, 31,192 development: the Milestone 2 figures);
  - the last three window values equal pm25_lag_1..3, the mean and standard deviation of the 8 values equal the stored
    rolling statistics, and the target is the block two blocks after the origin;
  - no window crosses a station and no block is missing (the block table has no absent rows);
  - how many consecutive rows of the features file are NOT 3 hours apart (why row order cannot be used);
  - no training target falls inside a validation period (the leakage guard of every fold).
Only train and dev rows are read; the 2025 test rows and the held-out stations are never loaded.

Requires: pip install pandas pyarrow numpy
"""
import pandas as pd

from config import REPORTS_DIR
from m3_data import (BLOCKS_PATH, MAX_LAG, build_features, expanding_folds, load_rows, loco_folds, read_block_grid, rows_hash,
                     verify_windows)


def main():
    """Run every window check on the train and dev rows, print the evidence and save the per-station table."""
    rows = load_rows()                      # load_rows() already runs verify_windows() and the row-count check
    counts = verify_windows(rows)
    print("--- Window checks passed ---")
    print(f"rows by split: {counts['rows_by_split']}; windows with all {MAX_LAG + 1} blocks valid: {counts['complete_windows']:,} "
          f"of {counts['rows']:,}")

    by_station = rows.groupby(["split", "station_id", "city"]).size().rename("usable_rows").reset_index()
    by_station["complete_windows"] = by_station["usable_rows"]
    by_station["windows_lost"] = 0
    print("\nUsable rows and complete windows per station:")
    print(by_station.to_string(index=False))

    grid = read_block_grid(rows["station_id"].unique())
    gaps = grid.groupby("station_id")["block_start_utc"].diff().dropna()
    print(f"\nBlock table ({BLOCKS_PATH.name}): {len(grid):,} blocks of {grid['station_id'].nunique()} stations; "
          f"every step within a station is 3 h: {bool((gaps == pd.Timedelta(hours=3)).all())}; "
          f"NaN block means (kept as NaN, never filled): {int(grid['pm25_3h_mean'].isna().sum()):,}")

    ordered = rows.sort_values(["station_id", "block_start_utc"])
    same_station = ordered["station_id"].eq(ordered["station_id"].shift(1))
    step = ordered["block_start_utc"].diff()
    print(f"Features file: {int((same_station & (step != pd.Timedelta(hours=3))).sum()):,} of {int(same_station.sum()):,} consecutive "
          f"same-station rows are NOT 3 h apart, so windows are built from the block table, never from row order.")

    print("\nLeakage guard of every fold (a training target must end before the validation period):")
    for fold in expanding_folds(rows) + loco_folds(rows):       # each constructor asserts the guard and stops on a failure
        print(f"  {fold.name:10s} fit {int(fold.fit.sum()):7,}  validation {int(fold.val.sum()):6,}  seen-station {int(fold.seen.sum()):6,}")
    dev = rows[rows["split"] == "dev"]
    print(f"\nEvaluation row set: {len(dev):,} development rows, key hash {rows_hash(dev)}")
    print(f"Feature set B has {build_features(rows, 'B').shape[1]} columns (8 window values + 4 time features).")

    out = REPORTS_DIR / "windows"
    out.mkdir(parents=True, exist_ok=True)
    by_station.to_csv(out / "window_checks_by_station.csv", index=False)
    print(f"\nTable saved in {out}")


if __name__ == "__main__":
    main()
