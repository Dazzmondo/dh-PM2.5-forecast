"""
EDA 2: what the forecast target looks like: its spread, its skew, and whether a log transform would help.

INPUT:  processed_data/features/pm25_features_with_splits.parquet   (create_splits.py; the `train` rows of non-held-out stations)
OUTPUT: reports/figures/02_target_distribution.png
        reports/eda/distributions/target_summary_by_source.csv, target_summary_by_station.csv   and the same tables printed

Question: is the 3-hour mean PM2.5 (the target) roughly symmetric or strongly right-skewed, and does log1p make it
symmetric? Only the training rows are used, so nothing here is chosen with the development, test or held-out data.

log1p needs values of 0 or more. A few block means are slightly negative (instrument noise; they are kept in the stored
data), so the log view clips at 0 first. That clipping is part of this view only, not of the stored data.

Requires: pip install pandas pyarrow numpy matplotlib
"""
import numpy as np
import pandas as pd

from analysis_helpers import (INK_SECONDARY, SOURCE_COLORS, SOURCE_LABELS, get_pyplot, load_features, print_conclusion,
                              report_dir, save_figure)
from stations import load_station_reference

PERCENTILES = (0.05, 0.5, 0.95, 0.99)
LINEAR_AXIS_QUANTILE = 0.995   # the linear panel stops at this quantile so the long tail does not squash the picture


def summarise(values: pd.Series) -> dict:
    """Count, mean, percentiles, maximum and skewness of a target series, on the raw and the log1p scale."""
    logged = np.log1p(values.clip(lower=0))
    out = {"n": len(values), "mean": values.mean()}
    out.update({f"p{int(100 * q)}": values.quantile(q) for q in PERCENTILES})
    out.update({"max": values.max(), "skew_raw": values.skew(), "skew_log1p": logged.skew()})
    return out


def draw(train: pd.DataFrame) -> None:
    """Two panels: the target on its own scale and on the log1p scale, one outline histogram per source."""
    plt = get_pyplot()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    upper = train["target_pm25"].quantile(LINEAR_AXIS_QUANTILE)
    for source, group in train.groupby("source"):
        label = f"{SOURCE_LABELS[source]} (n={len(group):,})"
        left.hist(group["target_pm25"].clip(upper=upper), bins=np.linspace(0, upper, 50), density=True, histtype="step",
                  color=SOURCE_COLORS[source], label=label)
        right.hist(np.log1p(group["target_pm25"].clip(lower=0)), bins=np.linspace(0, np.log1p(upper), 40), density=True,
                   histtype="step", color=SOURCE_COLORS[source], label=label)
    left.set_title("Target on its own scale", loc="left")
    left.set_xlabel(f"3-hour mean PM2.5, 6 h ahead (ug/m3; values above {upper:.0f} shown at {upper:.0f})")
    left.set_ylabel("share of rows per ug/m3")
    right.set_title("Target after log1p (negatives clipped to 0)", loc="left")
    right.set_xlabel("log(1 + PM2.5), PM2.5 in ug/m3 (the spike at 0 is blocks clipped from negative values)")
    right.set_ylabel("share of rows per unit")
    right.set_yticks(right.get_yticks())
    left.legend(loc="upper right")
    for axis in (left, right):
        axis.tick_params(colors=INK_SECONDARY)
    fig.suptitle("Training-split target, non-held-out stations (2018-2023; Irish stations 2022-2023)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "02_target_distribution.png")


def main():
    """Summarise the target by source and by station, draw the figure and print the conclusion."""
    features = load_features()
    stations = load_station_reference().set_index("station_id")
    train = features[features["split"] == "train"].copy()
    train["source"] = train["station_id"].map(stations["source"])
    out = report_dir("eda/distributions")

    by_source = pd.DataFrame({source: summarise(group["target_pm25"]) for source, group in train.groupby("source")}).T
    by_station = pd.DataFrame({sid: summarise(group["target_pm25"]) for sid, group in train.groupby("station_id")}).T
    overall = summarise(train["target_pm25"])
    print("--- Training-split target by source (ug/m3) ---")
    print(by_source.round(2))
    print("\n--- By station ---")
    print(by_station.round(2))
    by_source.to_csv(out / "target_summary_by_source.csv")
    by_station.to_csv(out / "target_summary_by_station.csv")
    draw(train)

    print_conclusion(
        finding=(f"The target is strongly right-skewed (skewness {overall['skew_raw']:.1f} on its own scale, {overall['skew_log1p']:.1f} "
                 f"after log1p); median {overall['p50']:.1f}, 99th percentile {overall['p99']:.1f}, maximum {overall['max']:.0f} ug/m3 "
                 f"over {overall['n']:,} training rows."),
        decision="Store the target unchanged. Tree models (XGBoost) need no transform. For linear and neural models, compare "
                 "log1p (negatives clipped to 0 inside the model's own step) with the raw scale in M3, using the same MAE on the "
                 "raw scale. Report MAE and RMSE together, because the tail makes them differ.",
        limitation="One distribution pools stations with different levels (Athens Aristotelous is far higher than the Irish stations). "
                   "Skewness is sensitive to a few extreme blocks, and the training years differ by station (Irish stations have two).")


if __name__ == "__main__":
    main()
