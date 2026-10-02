"""
Build the model-ready feature table from 3-hour aggregated PM2.5.

INPUT:  processed_data/three_hour_aggregated.parquet   (aggregate_to_3hour.py)
        one row per station per UTC 3-hour block: station_id, city, country,
        timezone, latitude, longitude, block_start_utc, pm25_3h_mean, valid_hours.
OUTPUT: processed_data/features/pm25_features.parquet
        the input columns plus local_timestamp, time_block, day_of_week, month,
        season, pm25_current, pm25_lag_1..3, pm25_rolling_mean, pm25_rolling_std,
        target_pm25 and target_block_start_utc. Rows without a full history or a
        valid target are dropped; nothing is imputed.

Modelling assumption: a forecast is issued AFTER the current 3-hour block has
completed, so the complete current-block mean (pm25_current) is a legitimate
input, and the target is the 3-hour mean TARGET_HORIZON_BLOCKS blocks (6 hours)
after the current block. Nothing from the target block or later enters any
feature.

Design notes
- The grid is UTC; time_block / day_of_week / month / season are derived
  from each station's LOCAL time (its `timezone`), because diurnal
  pollution patterns follow local clocks. A UTC grid keeps "+2 blocks = +6h"
  exact on DST-change days; converting only the label gets local meaning
  without that cost.
- Rolling mean/std include the current block (it is known at forecast time).
- target_block_start_utc is stored so the split script can see exactly
  which period each label comes from.

Requires: pip install pandas pyarrow
"""
import pandas as pd

from config import PROCESSED_DIR

INPUT_PATH = PROCESSED_DIR / "three_hour_aggregated.parquet"
OUTPUT_PATH = PROCESSED_DIR / "features" / "pm25_features.parquet"

ROLLING_WINDOW_BLOCKS = 8   # 8 x 3h = 24h of history, including the current block
TARGET_HORIZON_BLOCKS = 2   # 2 x 3h = 6h ahead
STATIC_COLUMNS = ["city", "country", "timezone", "latitude", "longitude"]   # station attributes that never change


def ensure_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each station onto a gap-free 3-hour UTC grid. Without this,
    shift()/rolling() would treat blocks days apart as neighbours."""
    filled = []
    for station_id, group in df.groupby("station_id"):
        for col in STATIC_COLUMNS:   # sanity check: each static column should hold exactly one value per station
            n = group[col].dropna().nunique()
            if n > 1:
                print(f"WARNING: station {station_id} has {n} distinct values for '{col}' - check the reference file.")
        group = group.set_index("block_start_utc").sort_index()
        # every 3-hour block start between this station's first and last block
        full_range = pd.date_range(group.index.min(), group.index.max(), freq="3h", tz="UTC")
        group = group.reindex(full_range)   # blocks that had no row are added as empty (NaN) rows
        group["station_id"] = station_id
        for col in STATIC_COLUMNS:
            group[col] = group[col].ffill().bfill()   # copy the station's static values into the added rows
        group.index.name = "block_start_utc"
        filled.append(group.reset_index())
    return pd.concat(filled, ignore_index=True)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """time_block / day_of_week / month / season from each station's LOCAL time."""
    parts = []
    for tz_name, group in df.groupby("timezone"):   # one timezone at a time (each station has its own local time)
        g = group.copy()
        # drop tz info after converting: mixing several tz-aware zones in one
        # column would collapse it to dtype 'object' and break .dt below
        g["local_timestamp"] = g["block_start_utc"].dt.tz_convert(tz_name).dt.tz_localize(None)
        parts.append(g)
    df = pd.concat(parts).sort_index()   # join the per-timezone pieces back together

    df["time_block"] = df["local_timestamp"].dt.hour // 3   # 0 = 00:00-02:59 local ... 7 = 21:00-23:59
    df["day_of_week"] = df["local_timestamp"].dt.dayofweek  # 0 = Monday ... 6 = Sunday
    df["month"] = df["local_timestamp"].dt.month            # 1-12
    # meteorological seasons: Dec-Feb winter, Mar-May spring, Jun-Aug summer, Sep-Nov autumn
    season_map = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    df["season"] = df["month"].map(season_map)
    return df


def add_lag_rolling_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add, per station: the current block, 3 lags, a 24-hour rolling mean and std, and the target.

    Relies on the gap-free grid from ensure_full_grid(), so a shift of N rows is exactly N blocks.
    """
    df = df.sort_values(["station_id", "block_start_utc"])   # time order within each station
    df["pm25_current"] = df["pm25_3h_mean"]   # the block that has just finished: known when the forecast is made
    # each station's own series, so a lag never reaches from one station into another
    grouped = df.groupby("station_id")["pm25_3h_mean"]

    for k in (1, 2, 3):
        df[f"pm25_lag_{k}"] = grouped.shift(k)   # the value k blocks (3k hours) earlier
    # mean / std of the last 8 blocks including this one; min_periods=8 gives NaN unless all 8 blocks have a value
    df["pm25_rolling_mean"] = grouped.transform(
        lambda s: s.rolling(ROLLING_WINDOW_BLOCKS, min_periods=ROLLING_WINDOW_BLOCKS).mean())
    df["pm25_rolling_std"] = grouped.transform(
        lambda s: s.rolling(ROLLING_WINDOW_BLOCKS, min_periods=ROLLING_WINDOW_BLOCKS).std())

    # the grid is gap-free, so shifting by N rows is exactly N blocks
    df["target_pm25"] = grouped.shift(-TARGET_HORIZON_BLOCKS)   # the value 2 blocks (6 h) LATER: what the model predicts
    df["target_block_start_utc"] = df["block_start_utc"] + pd.Timedelta(hours=3 * TARGET_HORIZON_BLOCKS)
    return df


def main():
    """Build the feature table: complete grid, then time features, then lags/rolling/target."""
    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH} not found. Run aggregate_to_3hour.py first.")
    df = pd.read_parquet(INPUT_PATH)
    df = ensure_full_grid(df)              # 1. every station gets every 3-hour block
    df = add_temporal_features(df)         # 2. time of day, weekday, month, season (local time)
    df = add_lag_rolling_and_target(df)    # 3. history features and the 6-hour-ahead target

    before = len(df)
    required = ["pm25_current", "pm25_lag_1", "pm25_lag_2", "pm25_lag_3",
                "pm25_rolling_mean", "pm25_rolling_std", "target_pm25"]
    df = df.dropna(subset=required)   # a row needs ALL inputs and the target; nothing is imputed
    print(f"Dropped {before - len(df)} rows lacking full history or a valid target ({len(df)} remain).")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH} ({len(df)} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
