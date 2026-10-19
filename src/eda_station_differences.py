"""
EDA 5: how different are the stations, and how much of the variation is simply "which station is it"?

INPUT:  processed_data/three_hour_aggregated.parquet   (training years 2018-2023 of non-held-out stations)
        metadata/station_reference.csv
OUTPUT: reports/figures/06_station_differences.png
        reports/eda/station_differences/station_summary.csv, variance_explained.csv   and the same tables printed

Question: do stations differ enough in level that a model could learn "this station is usually X" instead of learning
about the weather and the sources? The share of the variance of the 3-hour means that is explained by the group a block
belongs to (station, city, country, source) measures this directly (eta squared = between-group variation / total).

The figure shows each station's spread (5th, 25th, 50th, 75th and 95th percentile) with n = the number of valid blocks and the
percentage of expected blocks that are valid in the training years. Training years only; held-out stations are not used.

Requires: pip install pandas pyarrow matplotlib
"""
import pandas as pd

from analysis_helpers import (INK, SOURCE_COLORS, SOURCE_LABELS, get_pyplot, print_conclusion, report_dir, save_figure)
from config import PROCESSED_DIR
from create_splits import DEV_YEAR, HELD_OUT_STATIONS
from stations import load_station_reference

FIRST_BLOCK_OF_DEV = pd.Timestamp(f"{DEV_YEAR}-01-01", tz="UTC")


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Share of the variance of `values` explained by group membership (between-group sum of squares / total)."""
    means = values.groupby(groups).transform("mean")
    return float(((means - values.mean()) ** 2).sum() / ((values - values.mean()) ** 2).sum())


def main():
    """Summarise every training station, measure how much the group explains, draw the figure and print the conclusion."""
    path = PROCESSED_DIR / "three_hour_aggregated.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run aggregate_to_3hour.py first.")
    stations = load_station_reference().set_index("station_id")
    blocks = pd.read_parquet(path)
    blocks = blocks[(blocks["block_start_utc"] < FIRST_BLOCK_OF_DEV) & ~blocks["station_id"].isin(HELD_OUT_STATIONS)]
    for column in ("source", "city", "country"):
        blocks[column] = blocks["station_id"].map(stations[column])
    valid = blocks[blocks["pm25_3h_mean"].notna()]
    out = report_dir("eda/station_differences")

    rows = {}
    for sid, group in valid.groupby("station_id"):
        start = max(pd.Timestamp(stations.loc[sid, "availability_start"], tz="UTC"), pd.Timestamp("2018-01-01", tz="UTC"))
        expected = int((FIRST_BLOCK_OF_DEV - start) / pd.Timedelta(hours=3))
        values = group["pm25_3h_mean"]
        rows[sid] = {"station": stations.loc[sid, "station_name"], "source": stations.loc[sid, "source"], "n_blocks": len(values),
                     "percent_of_expected_blocks_valid": round(100 * len(values) / expected, 1), "mean": values.mean(),
                     "p5": values.quantile(0.05), "p25": values.quantile(0.25), "median": values.median(),
                     "p75": values.quantile(0.75), "p95": values.quantile(0.95)}
    summary = pd.DataFrame(rows).T.sort_values("median")
    numeric = ["n_blocks", "percent_of_expected_blocks_valid", "mean", "p5", "p25", "median", "p75", "p95"]
    summary[numeric] = summary[numeric].astype(float)
    print("--- Training-year 3-hour means by station (ug/m3) ---")
    print(summary.round(2))
    summary.to_csv(out / "station_summary.csv")

    explained = pd.Series({level: eta_squared(valid["pm25_3h_mean"], valid[level] if level != "station" else valid["station_id"])
                           for level in ("station", "city", "country", "source")}, name="variance_explained")
    print("\n--- Share of the variance of the 3-hour means explained by the group (eta squared) ---")
    print(explained.round(3))
    explained.to_csv(out / "variance_explained.csv")

    plt = get_pyplot()
    fig, (axis, bars) = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [3.2, 1]})
    for position, (sid, row) in enumerate(summary.iterrows()):
        color = SOURCE_COLORS[row["source"]]
        axis.plot([row["p5"], row["p95"]], [position, position], color=color, linewidth=1.2)
        axis.add_patch(plt.Rectangle((row["p25"], position - 0.28), row["p75"] - row["p25"], 0.56, color=color, alpha=0.55, linewidth=0))
        axis.plot([row["median"]] * 2, [position - 0.28, position + 0.28], color=INK, linewidth=1.6)
        axis.text(row["p95"] + 0.6, position, f"n={int(row['n_blocks']):,}  {row['percent_of_expected_blocks_valid']:.0f}% valid",
                  va="center", fontsize=7.5, color="#52514e")
    axis.set_yticks(range(len(summary)))
    axis.set_yticklabels([f"{sid}  {row['station']}" for sid, row in summary.iterrows()])
    axis.set_xlabel("3-hour mean PM2.5 (ug/m3). Line = 5th-95th percentile, box = quartiles, black bar = median")
    axis.set_title("Level and spread by station (training years)", loc="left")
    axis.set_xlim(left=min(0, summary["p5"].min() - 0.5))
    handles = [plt.Line2D([0], [0], color=SOURCE_COLORS[s], linewidth=4, label=SOURCE_LABELS[s]) for s in SOURCE_COLORS]
    axis.legend(handles=handles, loc="upper left")
    bars.barh(range(len(explained)), explained.values, color="#2a78d6", height=0.5)
    bars.set_yticks(range(len(explained)))
    bars.set_yticklabels(explained.index)
    bars.invert_yaxis()
    for i, value in enumerate(explained.values):
        bars.text(value + 0.01, i, f"{100 * value:.0f}%", va="center", fontsize=8)
    bars.set_xlim(0, max(0.5, float(explained.max()) + 0.12))
    bars.set_xlabel("share of variance explained")
    bars.set_title("Which group explains it", loc="left")
    fig.tight_layout()
    save_figure(fig, "06_station_differences.png")

    print_conclusion(
        finding=(f"Medians range from {summary['median'].min():.1f} to {summary['median'].max():.1f} ug/m3 across the {len(summary)} training "
                 f"stations; knowing the station explains {100 * explained['station']:.0f}% of the variance of a 3-hour mean "
                 f"(city {100 * explained['city']:.0f}%, country {100 * explained['country']:.0f}%, source {100 * explained['source']:.0f}%)."),
        decision="Keep station_id, city, latitude/longitude and country out of the baseline feature set, as planned: station identity "
                 "explains only a minority of the variance, so leaving it out should cost little, and the model must learn from the "
                 "history and the cycles to work in a new city. Measure the place information as a named ablation, validated by "
                 "leaving whole cities out (not stations), because the held-out units are cities.",
        limitation="This is evidence that place is not the main driver, not proof that adding it would hurt. Station level also "
                   "reflects real site differences (traffic versus background) that a model could use in a new city. Training years "
                   "differ by station, and the held-out cities cannot be profiled without using them.")


if __name__ == "__main__":
    main()
