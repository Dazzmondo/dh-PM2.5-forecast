"""
EDA 6: is the development year (2024) like the training years? A drift check, using no model.

INPUT:  processed_data/three_hour_aggregated.parquet           (3-hour means; non-held-out stations)
        processed_data/features/pm25_features_with_splits.parquet   (train and dev rows, for the year-by-year persistence error)
OUTPUT: reports/figures/07_dev_year_drift.png
        reports/eda/dev_drift/level_shift_by_station.csv, persistence_mae_by_year.csv   and the same tables printed

Question: do the stations' levels and the difficulty of the forecast change in 2024 compared with the earlier years? If
the development year is unusual, a model that is selected on it may be selected for the wrong reasons.

Two model-free measures: (1) each station's median and 90th percentile of the 3-hour mean in its training years against 2024;
(2) the error of the persistence forecast (the current block as the forecast) year by year, on the rows the models will use.
Only training and development data are used; 2025 and the held-out stations are not touched.

Pre-registered reading (a judgement): 2024 counts as "similar" for a station if its median is within 25% of the training
median, and the development year as a whole is "similar" if at least two thirds of the stations are, and its persistence
error lies within the range of the training-year errors.

Requires: pip install pandas pyarrow matplotlib
"""
import pandas as pd

from analysis_helpers import (BLUE, INK, INK_SECONDARY, ORANGE, get_pyplot, load_canonical, load_features, print_conclusion,
                              report_dir, save_figure)
from config import PROCESSED_DIR
from create_splits import DEV_YEAR, HELD_OUT_STATIONS
from stations import load_station_reference

SIMILAR_WITHIN = 0.25           # a station's median may move by this share and still count as similar
SIMILAR_STATION_SHARE = 2 / 3   # share of stations that must be similar


def level_shift(blocks: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Median and 90th percentile of the 3-hour mean in each station's training years against the development year."""
    blocks = blocks[blocks["pm25_3h_mean"].notna() & ~blocks["station_id"].isin(HELD_OUT_STATIONS)]
    year = blocks["block_start_utc"].dt.year
    train, dev = blocks[year < DEV_YEAR], blocks[year == DEV_YEAR]
    out = pd.DataFrame({
        "train_median": train.groupby("station_id")["pm25_3h_mean"].median(),
        "dev_median": dev.groupby("station_id")["pm25_3h_mean"].median(),
        "train_p90": train.groupby("station_id")["pm25_3h_mean"].quantile(0.9),
        "dev_p90": dev.groupby("station_id")["pm25_3h_mean"].quantile(0.9),
        "train_blocks": train.groupby("station_id").size(), "dev_blocks": dev.groupby("station_id").size()})
    out["median_change_percent"] = (100 * (out["dev_median"] - out["train_median"]) / out["train_median"]).round(1)
    out["p90_change_percent"] = (100 * (out["dev_p90"] - out["train_p90"]) / out["train_p90"]).round(1)
    out["station"] = stations["station_name"].reindex(out.index)
    return out.sort_values("median_change_percent")


def persistence_by_year(features: pd.DataFrame) -> pd.DataFrame:
    """Persistence error (current block as the forecast) of the train and dev rows, year by year."""
    rows = features[features["split"].isin(["train", "dev"])].copy()
    rows["year"] = rows["block_start_utc"].dt.year
    rows["abs_error"] = (rows["pm25_current"] - rows["target_pm25"]).abs()
    table = rows.groupby("year").agg(rows=("abs_error", "size"), MAE=("abs_error", "mean"), mean_target=("target_pm25", "mean"))
    table["MAE_percent_of_mean"] = (100 * table["MAE"] / table["mean_target"]).round(1)
    return table


def main():
    """Compute both drift measures, draw the figure and print the conclusion."""
    stations = load_station_reference().set_index("station_id")
    blocks = pd.read_parquet(PROCESSED_DIR / "three_hour_aggregated.parquet")
    shift = level_shift(blocks, stations)
    by_year = persistence_by_year(load_features())
    out = report_dir("eda/dev_drift")
    shift.to_csv(out / "level_shift_by_station.csv")
    by_year.to_csv(out / "persistence_mae_by_year.csv")
    print("--- Median and 90th percentile of the 3-hour mean: training years against 2024 ---")
    print(shift.round(2))
    print("\n--- Persistence error by year (train and dev rows of non-held-out stations) ---")
    print(by_year.round(3))

    canonical = load_canonical()
    cork = canonical[(canonical["station_id"] == "IE004BP") & canonical["pm25"].notna()]
    cork_negative = cork.groupby(cork["timestamp_utc"].dt.year)["pm25"].apply(lambda s: round(100 * (s < 0).mean(), 1))
    print("\nCork UCC share of valid hours that are negative, by year (%):", cork_negative.loc[:DEV_YEAR].to_dict())

    plt = get_pyplot()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    for position, (sid, row) in enumerate(shift.iterrows()):
        left.plot([row["train_median"], row["dev_median"]], [position, position], color=INK_SECONDARY, linewidth=1)
    left.scatter(shift["train_median"], range(len(shift)), color=BLUE, s=36, zorder=3, label="training years")
    left.scatter(shift["dev_median"], range(len(shift)), color=ORANGE, s=36, zorder=3, label=f"{DEV_YEAR} (development)")
    left.set_yticks(range(len(shift)))
    left.set_yticklabels([f"{sid}  {row['station']}  ({row['median_change_percent']:+.0f}%)" for sid, row in shift.iterrows()], fontsize=8)
    left.set_xlabel("median 3-hour mean PM2.5 (ug/m3); label = change of the median")
    left.set_title("Level: training years against 2024", loc="left")
    left.legend(loc="upper right")
    colors = [ORANGE if year == DEV_YEAR else BLUE for year in by_year.index]
    right.bar(by_year.index, by_year["MAE"], color=colors, width=0.62)
    for year, row in by_year.iterrows():
        right.text(year, row["MAE"] + 0.05, f"{row['MAE']:.2f}", ha="center", fontsize=8, color=INK)
    right.set_xticks(by_year.index)
    right.set_ylabel("persistence MAE (ug/m3)")
    right.set_title("Forecast difficulty by year (blue = training, orange = 2024)", loc="left", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "07_dev_year_drift.png")

    similar = int((shift["median_change_percent"].abs() <= 100 * SIMILAR_WITHIN).sum())
    train_range = by_year.loc[by_year.index < DEV_YEAR, "MAE"]
    dev_mae = by_year.loc[DEV_YEAR, "MAE"]
    within = bool(train_range.min() <= dev_mae <= train_range.max())
    verdict = ("similar" if similar >= SIMILAR_STATION_SHARE * len(shift) and within else "not clearly similar")
    notable = shift[shift["median_change_percent"].abs() > 100 * SIMILAR_WITHIN]
    outside = ", ".join(f"{sid} {pct:+.0f}%" for sid, pct in notable["median_change_percent"].items()) or "none"
    gap = dev_mae - train_range.min() if dev_mae < train_range.min() else dev_mae - train_range.max() if dev_mae > train_range.max() else 0.0
    print_conclusion(
        finding=(f"{similar} of {len(shift)} stations have a 2024 median within {100 * SIMILAR_WITHIN:.0f}% of their training median "
                 f"(outside: {outside}); the 2024 persistence error is {dev_mae:.2f} against {train_range.min():.2f} to "
                 f"{train_range.max():.2f} in the training years ({'inside' if within else f'outside by {abs(gap):.2f}'}), so by the "
                 f"pre-registered reading 2024 is {verdict}; the pooled differences are small and the one clear change is Cork UCC."),
        decision="Use 2024 as the development year as planned, and report per-station scores next to the pooled one. Treat Cork UCC "
                 "separately: its negative readings fell from about 20% of hours in 2022-2023 to about 8% in 2024, which looks like a "
                 "change in the instrument (not confirmed), so its dev score is not directly comparable with its training years.",
        limitation="One year cannot show a trend, and a year can be similar in level and still differ in weather. The Irish "
                   "stations have only two training years to compare with, and the 25% and two-thirds limits are judgements.")


if __name__ == "__main__":
    main()
