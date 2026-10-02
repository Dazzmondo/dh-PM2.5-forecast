"""
Aggregate canonical hourly PM2.5 into 3-hour mean blocks.

INPUT:  processed_data/canonical_hourly.parquet   (canonicalize_raw_data.py)
OUTPUT: processed_data/three_hour_aggregated.parquet
        one row per station per block: station_id, block_start_utc,
        pm25_3h_mean, valid_hours (0-3), city, country, timezone,
        latitude, longitude

Rule: a 3-hour block needs at least MIN_VALID_HOURS of its 3 hourly values;
otherwise pm25_3h_mean is NaN (never filled). valid_hours records how many
of the 3 hours were valid.

Design notes
- The block grid is UTC and fixed-width, so "+2 blocks = +6 hours" is
  exactly true all year. Local time is applied later, only to LABEL
  time-of-day (see feature_extraction.py).
- Station metadata (city, country, timezone, latitude, longitude) is
  carried straight through from the canonical file; this script holds no
  station lists of its own.
- The canonical file is unique per (station_id, hour) by construction. If
  it isn't, this stops rather than silently keeping the first row.
- Blocks are produced from each station's first to last observation; the
  gap-free grid is completed in feature_extraction.py (ensure_full_grid).

Requires: pip install pandas pyarrow
"""
import pandas as pd

from config import PROCESSED_DIR

INPUT_PATH = PROCESSED_DIR / "canonical_hourly.parquet"
OUTPUT_PATH = PROCESSED_DIR / "three_hour_aggregated.parquet"

MIN_VALID_HOURS = 2  # out of 3, per block
STATIC_COLUMNS = ["city", "country", "timezone", "latitude", "longitude"]


def aggregate_station(group: pd.DataFrame) -> pd.DataFrame:
    """Turn ONE station's hourly series into 3-hour mean blocks.

    Blocks are fixed UTC windows (00:00-03:00, 03:00-06:00, ...). A block's mean
    is kept only if at least MIN_VALID_HOURS of its 3 hours have a value.
    """
    group = group.set_index("timestamp_utc").sort_index()   # time-indexed, so resample() can cut it into windows
    if group.index.has_duplicates:   # the same hour twice means canonicalize_raw_data.py left a duplicate
        raise SystemExit("Duplicate hours found in the canonical file - re-run canonicalize_raw_data.py, "
                         "which is responsible for resolving them.")
    resampled = group["pm25"].resample("3h")   # fixed 3-hour windows, each labelled by its START time
    # mean() ignores missing hours; count() is how many hours in the window have a value (0-3)
    block_mean, valid_hours = resampled.mean(), resampled.count()
    out = pd.DataFrame({"block_start_utc": block_mean.index,
                        "pm25_3h_mean": block_mean.values,
                        "valid_hours": valid_hours.values})
    # too few valid hours: the block gets no mean (missing values are never filled in)
    out.loc[out["valid_hours"] < MIN_VALID_HOURS, "pm25_3h_mean"] = float("nan")
    return out


def main():
    """Aggregate every station's hourly data to 3-hour blocks and save one table."""
    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH} not found. Run canonicalize_raw_data.py first.")
    df = pd.read_parquet(INPUT_PATH)

    parts = []   # one aggregated table per station, joined at the end
    for station_id, group in df.groupby("station_id"):
        agg = aggregate_station(group)
        agg["station_id"] = station_id
        for col in STATIC_COLUMNS:
            # these never change within a station (they come from station_reference.csv), so take the first
            agg[col] = group[col].iloc[0]
        parts.append(agg)
    result = pd.concat(parts, ignore_index=True)

    print(f"{len(result)} 3-hour blocks, {int(result['pm25_3h_mean'].isna().sum())} "
          f"failed the {MIN_VALID_HOURS}/3-hour rule and are NaN.")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
