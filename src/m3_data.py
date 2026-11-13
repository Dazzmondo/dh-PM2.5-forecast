"""
Shared data code for the Milestone 3 experiments. Not a pipeline step: it is only imported (by train.py, check_windows.py, ...).

INPUT:  processed_data/features/pm25_features_with_splits.parquet   (create_splits.py)
        processed_data/three_hour_aggregated.parquet                (aggregate_to_3hour.py; the gap-free block table)
OUTPUT: nothing is written; the functions return DataFrames, boolean masks and small dictionaries.

What lives here, and why:

1. load_rows(): reads ONLY the splits you name. The test, held-out and future splits are refused unless allow_final=True, so an
   experiment script cannot touch them by accident (the 2025 test year and the held-out cities are used once, in the final
   evaluation, after every choice is frozen).
2. add_window_columns(): the features file holds blocks t, t-1, t-2, t-3 and the 8-block rolling mean/std only. Blocks t-4 ... t-7
   (the rest of the 24-hour window, and the t-6 block that is the same clock time as the target one day earlier) are rebuilt from
   the gap-free block table, NOT from row order in the features file (which has gaps). verify_windows() then proves the rebuilt
   window agrees with the stored lags, rolling statistics and target, and stops the run if it does not.
3. build_features(): the feature sets A, A+S and B and the time-feature variants (cyclical, one-hot UTC block, none).
4. expanding_folds() / loco_folds(): the validation folds, with the leakage guard (a training row's TARGET must end before the
   validation period starts).
5. Small helpers: a hash of the evaluation row keys, the metric function, the inner early-stopping window.

Naming: pm25_lag_k is the block k blocks (3k hours) before the current block t; pm25_lag_0 does not exist (pm25_current is t).
The target is block t+2, so the same clock time as the target one day earlier is t+2-8 = t-6 = pm25_lag_6.

Requires: pip install pandas pyarrow numpy
"""
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import PROCESSED_DIR
from feature_extraction import ROLLING_WINDOW_BLOCKS, TARGET_HORIZON_BLOCKS, ensure_full_grid
from fit_scalers import cyclical_features

FEATURES_PATH = PROCESSED_DIR / "features" / "pm25_features_with_splits.parquet"
BLOCKS_PATH = PROCESSED_DIR / "three_hour_aggregated.parquet"

DEV_PERIOD_START = pd.Timestamp("2024-01-01", tz="UTC")     # the development year starts here
TEST_PERIOD_START = pd.Timestamp("2025-01-01", tz="UTC")    # no block at or after this is read unless allow_final=True
FINAL_SPLITS = {"test", "held_out_test", "future_update"}   # used once, after the freeze
SELECTION_YEARS = (2021, 2022, 2023)                        # validation years of the expanding folds
LOCO_FIT_END_YEAR = 2022                                    # leave-one-city-out: train on targets up to the end of 2022 ...
LOCO_VAL_YEAR = 2023                                        # ... and validate the left-out city in 2023
CORK_UCC = "IE004BP"                                        # the Cork UCC station (many negative readings)
MAX_LAG = ROLLING_WINDOW_BLOCKS - 1                         # the window is t-7 ... t
EXPECTED_ROWS = {"train": 123_364, "dev": 31_192}           # the Milestone 2 row counts, checked on every load
EXPECTED_PERSISTENCE_DEV_MAE = 3.869                        # the Milestone 2 benchmark on the 31,192 development rows

BASE_COLUMNS = ["pm25_current", "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_rolling_mean", "pm25_rolling_std"]
FEATURE_SETS = {
    "A": BASE_COLUMNS,                                                       # the Milestone 2 baseline
    "A+S": BASE_COLUMNS + ["pm25_lag_6"],                                    # A plus the same clock time as the target, one day earlier
    "B": ["pm25_current"] + [f"pm25_lag_{k}" for k in range(1, MAX_LAG + 1)],  # the raw 8-block window
}


def load_rows(splits=("train", "dev"), allow_final: bool = False) -> pd.DataFrame:
    """Read the feature rows of the named splits and add the window columns (t-4 ... t-7), then verify them.

    The test, held-out and future splits need allow_final=True; only the final-evaluation script may pass it.
    """
    wanted = set(splits)
    if wanted & FINAL_SPLITS and not allow_final:
        raise SystemExit(f"STOPPED: splits {sorted(wanted & FINAL_SPLITS)} are for the final evaluation only "
                         f"(after every choice is frozen). Experiment scripts may read only train and dev.")
    if not FEATURES_PATH.exists() or not BLOCKS_PATH.exists():
        raise SystemExit("Processed data not found. Run the pipeline (create_splits.py) first.")
    rows = pq.read_table(FEATURES_PATH, filters=[("split", "in", sorted(wanted))]).to_pandas()
    rows = add_window_columns(rows, allow_final)
    counts = verify_windows(rows)
    for name, expected in EXPECTED_ROWS.items():
        if name in wanted and counts["rows_by_split"].get(name) != expected:
            raise SystemExit(f"STOPPED: split {name!r} has {counts['rows_by_split'].get(name)} rows, expected {expected}.")
    return rows.reset_index(drop=True)


def read_block_grid(station_ids, allow_final: bool = False) -> pd.DataFrame:
    """The gap-free per-station block table (every 3-hour block present, missing means as NaN), sorted by station and time."""
    filters = [("station_id", "in", sorted(station_ids))]
    if not allow_final:
        filters.append(("block_start_utc", "<", TEST_PERIOD_START))    # 2025 blocks are never even read
    blocks = pq.read_table(BLOCKS_PATH, filters=filters).to_pandas()
    return ensure_full_grid(blocks).sort_values(["station_id", "block_start_utc"]).reset_index(drop=True)


def add_window_columns(rows: pd.DataFrame, allow_final: bool = False) -> pd.DataFrame:
    """Add pm25_lag_4 ... pm25_lag_7 (and check columns) from the gap-free block table, joined on station and block start."""
    grid = read_block_grid(rows["station_id"].unique(), allow_final)
    by_station = grid.groupby("station_id")["pm25_3h_mean"]
    for k in range(4, MAX_LAG + 1):
        grid[f"pm25_lag_{k}"] = by_station.shift(k)          # one row = one block on the gap-free grid, so shift(k) = k blocks
    # the same quantities as the stored features, recomputed from the grid, only to verify the window
    for k in (1, 2, 3):
        grid[f"check_lag_{k}"] = by_station.shift(k)
    grid["check_rolling_mean"] = by_station.transform(lambda s: s.rolling(ROLLING_WINDOW_BLOCKS, min_periods=ROLLING_WINDOW_BLOCKS).mean())
    grid["check_rolling_std"] = by_station.transform(lambda s: s.rolling(ROLLING_WINDOW_BLOCKS, min_periods=ROLLING_WINDOW_BLOCKS).std())
    grid["check_target"] = by_station.shift(-TARGET_HORIZON_BLOCKS)
    keep = ["station_id", "block_start_utc"] + [f"pm25_lag_{k}" for k in range(4, MAX_LAG + 1)] + \
           [c for c in grid.columns if c.startswith("check_")]
    return rows.merge(grid[keep], on=["station_id", "block_start_utc"], how="left", validate="1:1")


def verify_windows(rows: pd.DataFrame) -> dict:
    """Stop the run if the rebuilt 8-block window disagrees with the stored features; return the counts for check_windows.py."""
    if rows[["pm25_lag_4", "pm25_lag_5", "pm25_lag_6", "pm25_lag_7"]].isna().any().any():
        raise SystemExit("WINDOW CHECK FAILED: a usable row has a missing block inside its 8-block window.")
    for k in (1, 2, 3):
        if not np.array_equal(rows[f"pm25_lag_{k}"].to_numpy(), rows[f"check_lag_{k}"].to_numpy()):
            raise SystemExit(f"WINDOW CHECK FAILED: pm25_lag_{k} differs from block t-{k} rebuilt from the block table.")
    if not np.array_equal(rows["target_pm25"].to_numpy(), rows["check_target"].to_numpy()):
        raise SystemExit("WINDOW CHECK FAILED: target_pm25 is not the block two blocks after the origin.")
    window = rows[["pm25_current"] + [f"pm25_lag_{k}" for k in range(1, MAX_LAG + 1)]].to_numpy()
    if not np.allclose(window.mean(axis=1), rows["pm25_rolling_mean"].to_numpy(), atol=1e-9, rtol=0):
        raise SystemExit("WINDOW CHECK FAILED: the mean of the 8 window values differs from pm25_rolling_mean.")
    # pandas' rolling std is an online calculation: for a window of 8 identical values it leaves about 1e-6 instead of 0
    # (24 of 154,556 rows), so the std is compared with an absolute tolerance of 1e-5 ug/m3, the mean with 1e-9
    if not np.allclose(window.std(axis=1, ddof=1), rows["pm25_rolling_std"].to_numpy(), atol=1e-5, rtol=0):
        raise SystemExit("WINDOW CHECK FAILED: the std of the 8 window values differs from pm25_rolling_std.")
    expected_gap = pd.Timedelta(hours=3 * TARGET_HORIZON_BLOCKS)
    if not ((rows["target_block_start_utc"] - rows["block_start_utc"]) == expected_gap).all():
        raise SystemExit("WINDOW CHECK FAILED: a target block does not start 6 hours after its origin block.")
    if rows.duplicated(["station_id", "block_start_utc"]).any():
        raise SystemExit("WINDOW CHECK FAILED: duplicate (station, block) rows.")
    return {"rows": len(rows), "rows_by_split": rows["split"].value_counts().to_dict(),
            "complete_windows": len(rows)}   # every row passed the checks above, so every window is complete


def time_features(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Time-of-day / season inputs: 'cyclical' (sin/cos of local hour and month), 'onehot_utc' (the UTC block 0-7) or 'none'."""
    if kind == "cyclical":
        return cyclical_features(rows)                       # the pipeline's own definition (fit_scalers.py); nothing is fitted
    if kind == "onehot_utc":
        utc_block = rows["block_start_utc"].dt.hour // 3     # the fixed UTC window, unlike the local `time_block` label
        return pd.DataFrame({f"utc_block_{b}": (utc_block == b).astype(float) for b in range(8)}, index=rows.index)
    if kind == "none":
        return pd.DataFrame(index=rows.index)
    raise SystemExit(f"Unknown time_features {kind!r} (use cyclical, onehot_utc or none).")


def build_features(rows: pd.DataFrame, feature_set: str, time_kind: str = "cyclical") -> pd.DataFrame:
    """The model inputs for a feature set (A, A+S or B) plus the chosen time features. Nothing here is fitted."""
    if feature_set not in FEATURE_SETS:
        raise SystemExit(f"Unknown feature_set {feature_set!r} (use {', '.join(FEATURE_SETS)}).")
    return pd.concat([rows[FEATURE_SETS[feature_set]], time_features(rows, time_kind)], axis=1)


def place_features(rows: pd.DataFrame, fit_mask: np.ndarray, variant: str) -> pd.DataFrame:
    """Place inputs for the ablation: 'none', 'latlon', 'country' (one-hot) or 'all'.

    The country categories come from the TRAINING rows only; a country that is absent from training gets all zeros.
    """
    parts = []
    if variant in ("latlon", "all"):
        parts.append(rows[["latitude", "longitude"]])
    if variant in ("country", "all"):
        known = sorted(rows.loc[fit_mask, "country"].unique())
        parts.append(pd.DataFrame({f"country_{c}": (rows["country"] == c).astype(float) for c in known}, index=rows.index))
    if variant not in ("none", "latlon", "country", "all"):
        raise SystemExit(f"Unknown place variant {variant!r} (use none, latlon, country or all).")
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=rows.index)


@dataclass
class Fold:
    """One train/validation split as boolean masks over the rows: fit rows, validation rows, and the 'seen-station' validation rows."""
    name: str
    fit: np.ndarray
    val: np.ndarray
    seen: np.ndarray
    val_start: pd.Timestamp


def _years(rows: pd.DataFrame):
    """The calendar year (UTC) of every row's origin block and target block, as arrays."""
    return rows["block_start_utc"].dt.year.to_numpy(), rows["target_block_start_utc"].dt.year.to_numpy()


def assert_no_leak(rows: pd.DataFrame, fold: Fold) -> None:
    """Stop if any fit row's TARGET block starts on or after the validation period, or if fit and validation rows overlap."""
    if (fold.fit & fold.val).any():
        raise SystemExit(f"LEAKAGE CHECK FAILED in fold {fold.name}: a row is in both the fit and validation sets.")
    if rows.loc[fold.fit, "target_block_start_utc"].max() >= fold.val_start:
        raise SystemExit(f"LEAKAGE CHECK FAILED in fold {fold.name}: a training target lies inside the validation period.")


def expanding_folds(rows: pd.DataFrame) -> list:
    """The three selection folds (validation years 2021-2023, train split only) and the development fold 'D' (train -> 2024).

    A fold's training rows are those whose TARGET ends before the validation year (the 6-hour purge); validation rows have both
    origin and target in the validation year. 'seen' marks validation rows of stations that have training rows in the fold.
    """
    split = rows["split"].to_numpy()
    origin_year, target_year = _years(rows)
    is_train = split == "train"
    folds = []
    for year in SELECTION_YEARS:
        fit = is_train & (target_year < year)
        val = is_train & (origin_year == year) & (target_year == year)
        seen_stations = set(rows.loc[fit, "station_id"].unique())
        seen = val & rows["station_id"].isin(seen_stations).to_numpy()
        folds.append(Fold(f"F{year}", fit, val, seen, pd.Timestamp(f"{year}-01-01", tz="UTC")))
    dev = split == "dev"
    folds.append(Fold("D", is_train, dev, dev.copy(), DEV_PERIOD_START))    # all 12 stations have training rows in 2018-2023
    for fold in folds:
        assert_no_leak(rows, fold)
    if set(rows.loc[folds[-1].val, "station_id"]) - set(rows.loc[folds[-1].fit, "station_id"]):
        raise SystemExit("The development fold contains a station with no training rows.")
    return folds


def loco_folds(rows: pd.DataFrame) -> list:
    """Leave-one-city-out folds (train split only): train on the other cities with targets up to the end of 2022, validate the left-out city in 2023."""
    is_train = rows["split"].to_numpy() == "train"
    origin_year, target_year = _years(rows)
    folds = []
    for city in sorted(rows.loc[is_train, "city"].unique()):
        is_city = (rows["city"] == city).to_numpy()
        fit = is_train & ~is_city & (target_year <= LOCO_FIT_END_YEAR)
        val = is_train & is_city & (origin_year == LOCO_VAL_YEAR) & (target_year == LOCO_VAL_YEAR)
        # the left-out city is unseen by design, so every validation row counts as 'unseen'; the decision metric is the whole fold
        folds.append(Fold(city, fit, val, val.copy(), pd.Timestamp(f"{LOCO_VAL_YEAR}-01-01", tz="UTC")))
        assert_no_leak(rows, folds[-1])
    return folds


def inner_early_stopping_split(rows: pd.DataFrame, fit_mask: np.ndarray, months: int = 12):
    """Split a fold's fit rows into a core part and an inner validation window (the last `months` months by TARGET time).

    Core rows' targets end before the inner window starts, so early stopping never sees a target that the core fit used.
    """
    targets = rows["target_block_start_utc"]
    inner_start = targets[fit_mask].max() - pd.DateOffset(months=months)
    inner = fit_mask & (targets >= inner_start).to_numpy()
    core = fit_mask & (rows["target_block_start_utc"] < inner_start).to_numpy()
    if core.sum() == 0 or inner.sum() == 0:
        raise SystemExit("The early-stopping window is empty: the fold's training period is too short.")
    return core, inner


def rows_hash(rows: pd.DataFrame) -> str:
    """SHA-256 of the evaluation row keys (station and block start), so a run records exactly which rows it was scored on."""
    keys = (rows["station_id"] + "|" + rows["block_start_utc"].dt.strftime("%Y-%m-%dT%H:%M")).sort_values()
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def mae_rmse(actual, predicted) -> tuple:
    """(MAE, RMSE) in ug/m3 between two arrays."""
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2)))
