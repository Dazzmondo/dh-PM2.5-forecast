"""
Check what the 3-hour block rule does: a block's mean needs at least MIN_VALID_HOURS (2) valid hours out of 3.

INPUT:  processed_data/three_hour_aggregated.parquet   (aggregate_to_3hour.py)
        processed_data/canonical_hourly.parquet        (for the what-if runs with a stricter or looser rule)
OUTPUT: reports/block_rule/*.csv   and the same tables printed to the terminal. Nothing is changed.

Counts, from 2018 on, how many blocks are
  - empty (0 valid hours: nothing to aggregate),
  - dropped by the rule (exactly 1 valid hour: some data, but not enough, so the block is missing),
  - kept with a partial mean (exactly 2 valid hours),
overall, per station and per station-year. It then repeats the feature building with a rule of 1, 2 and 3 valid hours and
shows the effect on the number of usable rows and on a persistence forecast for the development year (decision pool only).

Requires: pip install pandas pyarrow
"""
import pandas as pd

from aggregate_to_3hour import MIN_VALID_HOURS
from analysis_helpers import (POOL_START, features_from_hourly, load_canonical, report_dir, verify_against_pipeline)
from config import PROCESSED_DIR
from create_splits import HELD_OUT_STATIONS

RULES_TO_COMPARE = (1, 2, 3)   # valid hours required out of 3


def count_blocks(out) -> None:
    """Count empty, dropped and partial blocks from 2018 on: overall, per station and per station-year."""
    path = PROCESSED_DIR / "three_hour_aggregated.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run aggregate_to_3hour.py first.")
    blocks = pd.read_parquet(path)
    blocks = blocks[blocks["block_start_utc"] >= POOL_START].copy()
    blocks["year"] = blocks["block_start_utc"].dt.year
    blocks["empty"] = blocks["valid_hours"] == 0
    blocks["dropped_by_rule"] = blocks["valid_hours"] < MIN_VALID_HOURS
    blocks["dropped_by_rule"] &= ~blocks["empty"]
    blocks["partial_kept"] = blocks["valid_hours"] == MIN_VALID_HOURS

    print(f"--- Blocks from 2018 on (rule: at least {MIN_VALID_HOURS} of 3 hours valid) ---")
    print(blocks["valid_hours"].value_counts().sort_index().to_frame("blocks").assign(
        percent=lambda t: (100 * t["blocks"] / t["blocks"].sum()).round(1)))
    by_station = blocks.groupby("station_id").agg(blocks=("valid_hours", "size"), empty=("empty", "sum"),
                                                  dropped_by_rule=("dropped_by_rule", "sum"), partial_kept=("partial_kept", "sum"))
    with_data = by_station["blocks"] - by_station["empty"]
    by_station["dropped_percent_of_blocks_with_data"] = (100 * by_station["dropped_by_rule"] / with_data).round(2)
    by_station["partial_percent_of_kept_blocks"] = (100 * by_station["partial_kept"] / (with_data - by_station["dropped_by_rule"])).round(2)
    print("\nBy station:")
    print(by_station)
    print(f"\nTotals: dropped by the rule {int(by_station['dropped_by_rule'].sum()):,}, "
          f"kept with a partial mean {int(by_station['partial_kept'].sum()):,}, empty {int(by_station['empty'].sum()):,}")
    by_year = blocks[blocks["dropped_by_rule"]].groupby(["station_id", "year"]).size().unstack(fill_value=0)
    print("\nBlocks dropped by the rule, by station and year:")
    print(by_year)
    by_station.to_csv(out / "blocks_by_station.csv")
    by_year.to_csv(out / "dropped_blocks_by_station_year.csv")


def compare_rules(out) -> None:
    """Rebuild the feature rows under a rule of 1, 2 and 3 valid hours and compare rows and a persistence forecast."""
    canonical = load_canonical()
    verify_against_pipeline(canonical)
    hourly = canonical[~canonical["station_id"].isin(HELD_OUT_STATIONS) & (canonical["timestamp_utc"] >= POOL_START)]
    table = {}
    for rule in RULES_TO_COMPARE:
        rows = features_from_hourly(hourly, min_valid_hours=rule)
        dev = rows[rows["split"] == "dev"]
        table[f"{rule} of 3 valid hours"] = {"train_rows": int((rows["split"] == "train").sum()), "dev_rows": len(dev),
                                            "dev_MAE_own_target": round((dev["pm25_current"] - dev["target_pm25"]).abs().mean(), 3)}
    table = pd.DataFrame(table).T
    print("\n--- What-if: the rule changed, everything else the same (non-held-out stations) ---")
    print(table)
    table.to_csv(out / "rule_comparison.csv")


def main():
    """Print and save the block counts and the rule comparison."""
    out = report_dir("block_rule")
    count_blocks(out)
    compare_rules(out)
    print(f"\nTables saved in {out}")


if __name__ == "__main__":
    main()
