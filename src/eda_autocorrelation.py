"""
EDA 4: how much of the value six hours ahead is already in the current block? Autocorrelation of the 3-hour means.

INPUT:  processed_data/three_hour_aggregated.parquet   (aggregate_to_3hour.py; training years of non-held-out stations)
OUTPUT: reports/figures/05_autocorrelation.png
        reports/eda/autocorrelation/acf_by_station.csv   and the same tables printed

Question: how strongly is a block's mean related to the block k blocks later? Lag 2 (6 hours) is the forecast horizon, so the
correlation at lag 2 is the correlation between the current block and the target. Lag 8 (24 hours) shows whether yesterday
at the same time helps.

Method: on each station's gap-free 3-hour grid, correlate every block with the block k later, using only pairs where BOTH
blocks are valid and BOTH fall in the training years (2018-2023). Each station is correlated on its own (so differences in
level between stations do not inflate the result) and the figure shows the average over the stations of each source.
No future information is used for any prediction: both blocks of a pair are in the training years, and the pairs never reach
into the development or test year.

Requires: pip install pandas pyarrow matplotlib
"""
import pandas as pd

from analysis_helpers import SOURCE_COLORS, SOURCE_LABELS, get_pyplot, print_conclusion, report_dir, save_figure
from config import PROCESSED_DIR
from create_splits import DEV_YEAR, HELD_OUT_STATIONS
from feature_extraction import TARGET_HORIZON_BLOCKS, ensure_full_grid
from stations import load_station_reference

MAX_LAG_BLOCKS = 16     # 16 blocks = 48 hours
DAY_LAG_BLOCKS = 8      # 8 blocks = 24 hours


def acf_for_station(values: pd.Series) -> pd.Series:
    """Correlation of the block mean with the block k later (k = 1..MAX_LAG_BLOCKS); only pairs with both blocks valid count.

    `values` is one station's gap-free series, so shifting by k rows is exactly k blocks.
    """
    return pd.Series({k: values.corr(values.shift(-k)) for k in range(1, MAX_LAG_BLOCKS + 1)})


def main():
    """Compute the autocorrelation per station, draw the figure and print the conclusion."""
    path = PROCESSED_DIR / "three_hour_aggregated.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run aggregate_to_3hour.py first.")
    blocks = ensure_full_grid(pd.read_parquet(path)).sort_values(["station_id", "block_start_utc"])
    # training years only (the table starts in 2018) and non-held-out stations; the rows stay contiguous, so shifts stay exact
    blocks = blocks[(blocks["block_start_utc"].dt.year < DEV_YEAR) & ~blocks["station_id"].isin(HELD_OUT_STATIONS)]
    stations = load_station_reference().set_index("station_id")
    acf = pd.DataFrame({station_id: acf_for_station(group["pm25_3h_mean"].reset_index(drop=True))
                        for station_id, group in blocks.groupby("station_id")}).T
    acf.columns.name = "lag_blocks"
    out = report_dir("eda/autocorrelation")
    acf.to_csv(out / "acf_by_station.csv")
    source = stations["source"].reindex(acf.index)
    print("--- Autocorrelation of the 3-hour mean at lags of 1, 2 (= 6 h), 3, 8 (= 24 h) and 16 blocks, by station ---")
    print(acf[[1, TARGET_HORIZON_BLOCKS, 3, DAY_LAG_BLOCKS, MAX_LAG_BLOCKS]].round(2))

    plt = get_pyplot()
    fig, axis = plt.subplots(figsize=(7.4, 3.9))
    hours = [3 * k for k in acf.columns]
    for name, group in acf.groupby(source):
        axis.plot(hours, group.mean(), color=SOURCE_COLORS[name], marker="o", markersize=3,
                  label=f"{SOURCE_LABELS[name]} ({len(group)} station{'s' if len(group) > 1 else ''}, mean)")
        axis.fill_between(hours, group.min(), group.max(), color=SOURCE_COLORS[name], alpha=0.12, linewidth=0)
    axis.axvline(3 * TARGET_HORIZON_BLOCKS, color="#52514e", linewidth=0.9, linestyle=":")
    axis.text(3 * TARGET_HORIZON_BLOCKS + 0.6, 0.02, "6 h: forecast horizon", fontsize=8, color="#52514e")
    axis.set_xlabel("lag (hours between the two blocks)")
    axis.set_ylabel("correlation of the 3-hour means")
    axis.set_xticks(range(0, 49, 6))
    axis.set_title("Autocorrelation of the 3-hour mean PM2.5 within each station (training years); band = lowest to highest station",
                   loc="left", fontsize=9)
    axis.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "05_autocorrelation.png")

    at_horizon = acf[TARGET_HORIZON_BLOCKS]
    at_day = acf[DAY_LAG_BLOCKS]
    day_beats_nine_hours = int((at_day > acf[3]).sum())
    print_conclusion(
        finding=(f"The correlation between the current block and the block 6 hours ahead is {at_horizon.median():.2f} (median over stations, "
                 f"range {at_horizon.min():.2f} to {at_horizon.max():.2f}), so the current block alone accounts for roughly "
                 f"{100 * at_horizon.median() ** 2:.0f}% of the variance; at 24 hours it is {at_day.median():.2f}, and at {day_beats_nine_hours} "
                 f"of {len(acf)} stations the 24-hour correlation is higher than the 9-hour one (a daily cycle)."),
        decision="Keep the current block, three lags and the 24-hour rolling mean and standard deviation as inputs. Add 'same block "
                 "yesterday' (lag 8) as a candidate feature for M3 and keep it only if it helps on the development year. "
                 "Expect a model to beat the persistence baseline by using the cycle and the history, not the current block alone.",
        limitation="A linear correlation on a skewed variable is dominated by the high episodes, and says nothing about forecast "
                   "error at low levels. Stations are averaged within source, so the Irish group rests on two years.")


if __name__ == "__main__":
    main()
