"""
Persistence baseline on the development year (2024): the forecast for the block 6 hours ahead is simply the current block.

INPUT:  processed_data/features/pm25_features_with_splits.parquet   (create_splits.py)
        processed_data/three_hour_aggregated.parquet                (only for the looser comparison set below)
OUTPUT: reports/baseline/persistence_dev_by_station.csv   and the same tables printed to the terminal. Nothing is changed.

The target is the exact one the models will use: the 3-hour mean PM2.5 two blocks (6 hours) after the current block. The
forecast is `pm25_current`, the mean of the block that has just finished. Nothing is fitted, so nothing can leak; the
development rows are the ones any later model is also scored on, which is why this baseline is computed on them.

Missing-data policy (stated, not hidden): a row exists only if the current block, the three lags, the 24-hour rolling mean
and standard deviation, and the target are ALL valid. Persistence itself needs only the current block and the target, so the
script also reports a looser set that keeps those rows, to show what the strict policy costs.

Requires: pip install pandas pyarrow numpy
"""
import numpy as np
import pandas as pd

from analysis_helpers import FEATURES_PATH, label_splits, report_dir
from config import PROCESSED_DIR
from feature_extraction import add_lag_rolling_and_target, ensure_full_grid


def scores(rows: pd.DataFrame) -> dict:
    """Number of rows, MAE, RMSE and mean target of the persistence forecast on these rows."""
    error = rows["pm25_current"] - rows["target_pm25"]
    return {"rows": len(rows), "MAE": error.abs().mean(), "RMSE": float(np.sqrt((error ** 2).mean())),
            "mean_target": rows["target_pm25"].mean()}


def main():
    """Print and save the persistence baseline on the development split, overall and per station."""
    if not FEATURES_PATH.exists():
        raise SystemExit(f"{FEATURES_PATH} not found. Run the pipeline (create_splits.py) first.")
    dev = pd.read_parquet(FEATURES_PATH)
    dev = dev[dev["split"] == "dev"]
    overall = scores(dev)
    print("--- Persistence baseline, development year (2024), strict policy: all inputs and the target valid ---")
    print(f"rows {overall['rows']:,}; MAE {overall['MAE']:.3f} ug/m3; RMSE {overall['RMSE']:.3f}; mean target {overall['mean_target']:.2f}; "
          f"MAE is {100 * overall['MAE'] / overall['mean_target']:.0f}% of the mean target")

    by_station = pd.DataFrame({sid: scores(group) for sid, group in dev.groupby("station_id")}).T
    by_station["MAE_percent_of_mean"] = (100 * by_station["MAE"] / by_station["mean_target"]).round(1)
    print("\nBy station:")
    print(by_station.astype({"rows": int}).round(3))

    # the looser set: only the current block and the target are needed for persistence
    blocks = pd.read_parquet(PROCESSED_DIR / "three_hour_aggregated.parquet")
    loose = add_lag_rolling_and_target(ensure_full_grid(blocks)).dropna(subset=["pm25_current", "target_pm25"])
    loose = label_splits(loose)
    loose = loose[loose["split"] == "dev"]
    looser = scores(loose)
    print(f"\nLooser set (only the current block and the target needed): rows {looser['rows']:,}; MAE {looser['MAE']:.3f}; "
          f"the strict policy keeps {100 * overall['rows'] / looser['rows']:.1f}% of those rows")

    out = report_dir("baseline")
    by_station.to_csv(out / "persistence_dev_by_station.csv")
    pd.DataFrame({"strict": overall, "looser": looser}).T.to_csv(out / "persistence_dev_overall.csv")
    print(f"\nTables saved in {out}")


if __name__ == "__main__":
    main()
