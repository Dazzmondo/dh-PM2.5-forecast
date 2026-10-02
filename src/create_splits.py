"""
Label every feature row: train / dev / test / held_out_test / future_update / excluded.

INPUT:  processed_data/features/pm25_features.parquet              (feature_extraction.py)
OUTPUT: processed_data/features/pm25_features_with_splits.parquet  (same rows plus a `split` column)

Two dimensions:
  Time:   train = from the station's availability_start (never before 2018)
          up to end-2023; dev = 2024; test = 2025; 2026 = future_update
          (reserved for the Milestone 4 retraining demo).
  Place:  stations in HELD_OUT_STATIONS never enter train/dev/test. Their
          2025 rows become held_out_test (unseen-city evaluation); their
          other rows are kept but marked excluded.

A row is only labelled with a period if its INPUT block and its TARGET block
(6h later) fall in the same PERIOD (train / dev / test / future). Only rows
whose input and target timestamps straddle a split boundary are excluded;
New Year inside the training period is not a boundary.

Safety checks (the run stops, writing nothing, if one fails):
  - every code in HELD_OUT_STATIONS exists in station_reference.csv (catches typos);
  - every held-out station is in the feature table and has 2025 rows, so the
    unseen-city evaluation cannot silently shrink;
  - no held-out row ends up in train, dev or test (leakage check).

Requires: pip install pandas pyarrow numpy
"""
import numpy as np
import pandas as pd

from config import PROCESSED_DIR
from stations import load_station_reference

INPUT_PATH = PROCESSED_DIR / "features" / "pm25_features.parquet"
OUTPUT_PATH = PROCESSED_DIR / "features" / "pm25_features_with_splits.parquet"

# Stations kept out of train/dev/test entirely: unseen cities used to test how well the model generalises
HELD_OUT_STATIONS = {
    "GB1060A",  # Derry Rosemount
    "GB0566A",  # London Bloomsbury
    "GB0620A",  # London N. Kensington
    "FR04002",  # Paris Gennevilliers
    "FR04058",  # Paris Saint-Denis
}

STUDY_START_YEAR = 2018           # no training data before this year, even if a station has older files
DEV_YEAR = 2024                   # development set (model selection)
TEST_YEAR = 2025                  # final test set
FUTURE_UPDATE_START_YEAR = 2026   # new data held back for the Milestone 4 update demo


def period_of(year: pd.Series, train_start_year: pd.Series) -> pd.Series:
    """Which period a calendar year belongs to, for a given station."""
    # np.select takes the FIRST condition that is true: 2026+ is "future", then 2025 "test", then 2024 "dev";
    # any earlier year from the station's own start year onwards is "train"; older years are "before_study"
    return pd.Series(np.select(
        [year >= FUTURE_UPDATE_START_YEAR, year == TEST_YEAR, year == DEV_YEAR, year >= train_start_year],
        ["future", "test", "dev", "train"], default="before_study"), index=year.index)


def assign_splits(df: pd.DataFrame, train_start_year: pd.Series) -> pd.Series:
    """Return the split label of every row (the rules are in the module docstring)."""
    in_period = period_of(df["block_start_utc"].dt.year, train_start_year)           # period of the input block
    target_period = period_of(df["target_block_start_utc"].dt.year, train_start_year)  # period of the block predicted
    same_period = in_period == target_period  # False only for rows straddling a split boundary
    is_held_out = df["station_id"].isin(HELD_OUT_STATIONS)

    # np.select takes the FIRST condition that is true, so the ORDER below matters.
    conditions = [
        same_period & (in_period == "future"),                  # 2026 rows: kept for the update demo
        same_period & is_held_out & (in_period == "test"),      # an unseen city's 2025 rows: the unseen-city test
        is_held_out,                                            # every other row of an unseen city: unused
        same_period & (in_period == "test"),
        same_period & (in_period == "dev"),
        same_period & (in_period == "train"),
    ]
    choices = ["future_update", "held_out_test", "excluded", "test", "dev", "train"]
    # a row matching nothing (before the study, or input and target in different periods) is "excluded"
    return pd.Series(np.select(conditions, choices, default="excluded"), index=df.index)


def main():
    """Label every row with its split, run the safety checks, and save the result."""
    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH} not found. Run feature_extraction.py first.")
    stations = load_station_reference()

    # a mistyped held-out code would silently exclude nobody, so check they all exist
    typo = HELD_OUT_STATIONS - set(stations["station_id"])
    if typo:
        raise SystemExit(f"HELD_OUT_STATIONS contains codes that are not in station_reference.csv: {sorted(typo)}")

    df = pd.read_parquet(INPUT_PATH)
    unknown = set(df["station_id"]) - set(stations["station_id"])   # every station in the data must be in the reference
    if unknown:
        raise SystemExit(f"Feature table has stations missing from station_reference.csv: {sorted(unknown)}")

    # training starts in each station's first available year (never before 2018): this is how the Irish
    # stations start in 2022 and the others in 2018, with no special rule
    avail_year = stations.set_index("station_id")["availability_start"].dt.year
    train_start = df["station_id"].map(avail_year).clip(lower=STUDY_START_YEAR)

    df["split"] = assign_splits(df, train_start)
    print(df["split"].value_counts().to_string())

    # report how many rows were excluded because their input and target block are in different periods
    in_p = period_of(df["block_start_utc"].dt.year, train_start)
    tg_p = period_of(df["target_block_start_utc"].dt.year, train_start)
    print(f"({int((in_p != tg_p).sum())} rows excluded because their input and target fall in different periods)")

    absent = sorted(HELD_OUT_STATIONS - set(df["station_id"]))
    if absent:
        raise SystemExit(f"STOPPED: held-out stations missing from the feature table: {absent}. "
                         f"The unseen-city evaluation needs all of them - check their raw data and earlier steps.")

    # a held-out row in train/dev/test would defeat the unseen-city experiment, so stop if one slipped in
    leaked = df[df["station_id"].isin(HELD_OUT_STATIONS) & df["split"].isin(["train", "dev", "test"])]
    if not leaked.empty:
        raise SystemExit(f"LEAKAGE CHECK FAILED: {len(leaked)} held-out rows landed in train/dev/test.")
    print("Leakage check passed: no held-out station rows in train/dev/test.")

    # every unseen city needs 2025 rows, otherwise its part of the evaluation would silently be empty
    held_rows = df[df["split"] == "held_out_test"]["station_id"].value_counts()
    no_test_rows = sorted(s for s in HELD_OUT_STATIONS if held_rows.get(s, 0) == 0)
    if no_test_rows:
        raise SystemExit(f"STOPPED: held-out stations with no {TEST_YEAR} test rows: {no_test_rows}.")
    print("Held-out test rows per station: " + ", ".join(f"{s}={int(held_rows[s])}" for s in sorted(HELD_OUT_STATIONS)))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
