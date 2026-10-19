"""
Fit the scalers on the TRAINING split only and save them as a small, versioned file (the "scaler artefact").

INPUT:  processed_data/features/pm25_features_with_splits.parquet   (create_splits.py)
OUTPUT: processed_data/scalers/scalers.json   (store_preprocessed_data.py uploads it to processed/scalers/ next to the dataset)

Why: linear and neural models need inputs on a similar scale; XGBoost does not. Anything that learns from data (a mean, a
standard deviation) must learn it from the training rows alone, otherwise the development and test years leak into the
inputs. The processed Parquet files stay UNSCALED. A model applies the saved numbers, unchanged, to the development, test and
held-out rows; apply_scaler() below does exactly that and never refits.

Two candidate variants are fitted so that M3 can compare them (neither is "the rule"):
  standard         (x - mean) / std on the raw values
  log1p_standard   log(1 + max(x, 0)) first, then (x - mean) / std. Negative readings are kept in the stored data; clipping at
                   0 happens only here, because log1p cannot take a value of -1 or less.
The concentration columns (current, lags, rolling mean) and the spread column (rolling_std) are fitted as separate groups, each
with its own mean and std, because rolling_std is a different kind of quantity (always 0 or more). The target is fitted the
same way, so a model's predictions can be turned back into ug/m3 (inverse_target).

The cyclical time features (sin/cos of the local hour and of the month) need no fitting: cyclical_features() builds them from
local_timestamp, so they follow the local clock including daylight saving. The `season` column is not used (it repeats the month).

The file records what it was fitted on (rows, stations, period, a hash of the feature table, the code version), so any model can
say exactly which dataset its scaling came from.

Requires: pip install pandas pyarrow numpy
"""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import PIPELINE_VERSION, PROCESSED_DIR, git_commit
from create_splits import HELD_OUT_STATIONS

FEATURES_PATH = PROCESSED_DIR / "features" / "pm25_features_with_splits.parquet"
SCALER_PATH = PROCESSED_DIR / "scalers" / "scalers.json"

ARTEFACT_VERSION = "v1"   # change when the layout of the file changes
FIT_SPLIT = "train"       # the only split the numbers are learned from
CONCENTRATION_COLUMNS = ["pm25_current", "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_rolling_mean"]
SPREAD_COLUMNS = ["pm25_rolling_std"]
TARGET_COLUMN = "target_pm25"
VARIANTS = ("standard", "log1p_standard")
BLOCK_HOURS = 3           # a block's midpoint is 1.5 hours after its start


def sha256_of_file(path) -> str:
    """Fingerprint of a file's contents, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(values: pd.Series, variant: str) -> pd.Series:
    """The values before standardising: unchanged, or log(1 + max(x, 0)) for the log1p variant."""
    if variant == "standard":
        return values
    if variant == "log1p_standard":
        return np.log1p(values.clip(lower=0))
    raise ValueError(f"unknown variant {variant!r}")


def fit_columns(train: pd.DataFrame, columns: list, variant: str) -> dict:
    """Mean and standard deviation (population, ddof=0, as scikit-learn's StandardScaler) of each column, from the train rows."""
    fitted = {}
    for column in columns:
        values = prepare(train[column], variant)
        fitted[column] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return fitted


def fit(features: pd.DataFrame) -> dict:
    """Fit every variant on the training rows only and return the artefact as a dictionary (stops if the rows are wrong)."""
    train = features[features["split"] == FIT_SPLIT]
    if train.empty:
        raise SystemExit("No training rows to fit the scalers on.")
    if set(train["station_id"]) & HELD_OUT_STATIONS:
        raise SystemExit("LEAKAGE CHECK FAILED: held-out stations are among the training rows.")
    variants = {}
    for variant in VARIANTS:
        variants[variant] = {"concentration": fit_columns(train, CONCENTRATION_COLUMNS, variant),
                             "spread": fit_columns(train, SPREAD_COLUMNS, variant),
                             "target": fit_columns(train, [TARGET_COLUMN], variant)}
    return {
        "artefact_version": ARTEFACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": git_commit(),
        "fitted_on": {"split": FIT_SPLIT, "rows": len(train), "stations": sorted(train["station_id"].unique()),
                      "first_block_utc": train["block_start_utc"].min().isoformat(),
                      "last_block_utc": train["block_start_utc"].max().isoformat(),
                      "features_file_sha256": sha256_of_file(FEATURES_PATH)},
        "variants": variants,
        "cyclical": {"hour": {"period": 24, "from": "local_timestamp", "offset_hours": BLOCK_HOURS / 2},
                     "month": {"period": 12, "from": "local_timestamp"}},
        "notes": "Fitted on the train split only. Apply unchanged to dev, test and held-out rows. The Parquet files are unscaled.",
    }


def apply_scaler(df: pd.DataFrame, artefact: dict, variant: str) -> pd.DataFrame:
    """Scale the feature columns of `df` with the SAVED numbers (never refitting) and return them as a new table."""
    groups = artefact["variants"][variant]
    scaled = pd.DataFrame(index=df.index)
    for group in ("concentration", "spread"):
        for column, p in groups[group].items():
            scaled[column] = (prepare(df[column], variant) - p["mean"]) / p["std"]
    return scaled


def scale_target(values: pd.Series, artefact: dict, variant: str) -> pd.Series:
    """Scale the target with the saved numbers."""
    p = artefact["variants"][variant]["target"][TARGET_COLUMN]
    return (prepare(values, variant) - p["mean"]) / p["std"]


def inverse_target(scaled: pd.Series, artefact: dict, variant: str) -> pd.Series:
    """Turn scaled predictions back into ug/m3 (for the log1p variant: exp(x) - 1)."""
    p = artefact["variants"][variant]["target"][TARGET_COLUMN]
    values = scaled * p["std"] + p["mean"]
    return np.expm1(values) if variant == "log1p_standard" else values


def cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """sin/cos of the local hour at the block midpoint and of the month, from local_timestamp. Nothing is fitted."""
    midpoint = df["local_timestamp"] + pd.Timedelta(hours=BLOCK_HOURS / 2)
    hour = midpoint.dt.hour + midpoint.dt.minute / 60
    month = df["local_timestamp"].dt.month
    return pd.DataFrame({"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                         "month_sin": np.sin(2 * np.pi * (month - 1) / 12), "month_cos": np.cos(2 * np.pi * (month - 1) / 12)},
                        index=df.index)


def main():
    """Fit on the train split, save the artefact, and show that it scales train to about 0 / 1 and dev unchanged."""
    if not FEATURES_PATH.exists():
        raise SystemExit(f"{FEATURES_PATH} not found. Run create_splits.py first.")
    features = pd.read_parquet(FEATURES_PATH)
    artefact = fit(features)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    info = artefact["fitted_on"]
    print(f"Fitted on {info['rows']:,} {FIT_SPLIT} rows of {len(info['stations'])} stations ({info['first_block_utc'][:10]} to "
          f"{info['last_block_utc'][:10]}). Saved {SCALER_PATH}")

    saved = json.loads(SCALER_PATH.read_text(encoding="utf-8"))   # re-read, so the check uses exactly what was saved
    print("\nCheck: scaled values with the SAVED numbers (train should be about mean 0 / std 1; dev must NOT be forced to it):")
    for variant in VARIANTS:
        for split in ("train", "dev"):   # the test and held-out rows are not looked at here
            rows = features[features["split"] == split]
            if rows.empty:
                continue
            scaled = apply_scaler(rows, saved, variant)
            print(f"  {variant:15s} {split:14s} rows {len(rows):6,}  pm25_current mean {scaled['pm25_current'].mean():+.3f} "
                  f"std {scaled['pm25_current'].std(ddof=0):.3f} | rolling_std mean {scaled['pm25_rolling_std'].mean():+.3f}")
    train = features[features["split"] == FIT_SPLIT]
    roundtrip = inverse_target(scale_target(train[TARGET_COLUMN].clip(lower=0), saved, "log1p_standard"), saved, "log1p_standard")
    print(f"Check: log1p target scaled and turned back gives the original values (max difference "
          f"{float((roundtrip - train[TARGET_COLUMN].clip(lower=0)).abs().max()):.2e} ug/m3).")
    cyclical = cyclical_features(features.head(5))
    print(f"Check: cyclical features are in [-1, 1]: {bool(cyclical.abs().max().max() <= 1)} (nothing is fitted for them).")


if __name__ == "__main__":
    main()
