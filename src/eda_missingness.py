"""
EDA 1: how much of each station's expected hourly PM2.5 is actually there, and what the gaps look like.

INPUT:  processed_data/canonical_hourly.parquet   (canonicalize_raw_data.py)
        metadata/station_reference.csv            (availability_start: where each station is expected to begin)
OUTPUT: reports/figures/01_missingness_heatmap.png
        reports/eda/missingness/completeness_by_station_year.csv, gaps_by_station.csv   and the same tables printed

Question: which station-years are well covered, which are not, and are the missing hours scattered single hours (which a
short-gap method could bridge) or long outages (which nothing should bridge)?

Counts cover all stations and years 2018-2025 (a count is not a fitted value). Expected hours come ONLY from
availability_start, never from the data, so a station that lost months cannot shrink its own expected period.
The 75% line is the project's predefined completeness target (validate_data_quality.py): a flag, not a filter.

Requires: pip install pandas pyarrow numpy matplotlib
"""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from analysis_helpers import (INK, INK_SECONDARY, SEQUENTIAL_BLUE, SURFACE, get_pyplot, load_canonical, print_conclusion,
                              report_dir, save_figure)
from create_splits import HELD_OUT_STATIONS
from validate_data_quality import COMPLETENESS_THRESHOLD, STUDY_END, STUDY_START
from stations import load_station_reference

SHORT_GAP_HOURS = 3     # a gap of up to this many hours could be bridged by a short-gap method
LONG_GAP_HOURS = 24     # a gap of at least this many hours is an outage


def completeness(canonical: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Valid hours / expected hours for every station and year (expected hours from availability_start only)."""
    valid = canonical[canonical["pm25"].notna()]
    valid_by_year = valid.groupby(["station_id", valid["timestamp_utc"].dt.year]).size()
    rows = []
    for _, st in stations.iterrows():
        start = max(pd.Timestamp(st["availability_start"], tz="UTC"), STUDY_START)
        for year in range(STUDY_START.year, STUDY_END.year + 1):
            first = max(start, pd.Timestamp(f"{year}-01-01", tz="UTC"))
            last = min(STUDY_END, pd.Timestamp(f"{year}-12-31 23:00", tz="UTC"))
            expected = max(0, int((last - first) / pd.Timedelta(hours=1)) + 1)
            have = int(valid_by_year.get((st["station_id"], year), 0))
            rows.append({"station_id": st["station_id"], "year": year, "expected_hours": expected, "valid_hours": have,
                         "percent_valid": round(100 * have / expected, 1) if expected else np.nan})
    return pd.DataFrame(rows)


def gap_summary(canonical: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """For each station: how its missing hours split into short gaps, medium gaps and outages, and the longest gap."""
    out = []
    for _, st in stations.iterrows():
        start = max(pd.Timestamp(st["availability_start"], tz="UTC"), STUDY_START)
        hours = pd.date_range(start, STUDY_END, freq="h")
        series = canonical[canonical["station_id"] == st["station_id"]].set_index("timestamp_utc")["pm25"]
        missing = series.reindex(hours).isna().to_numpy()
        # lengths of the runs of consecutive missing hours
        edges = np.flatnonzero(np.diff(np.concatenate(([0], missing.astype(int), [0]))))
        lengths = edges[1::2] - edges[0::2]
        total = int(missing.sum())
        short = int(lengths[lengths <= SHORT_GAP_HOURS].sum())
        long = int(lengths[lengths >= LONG_GAP_HOURS].sum())
        out.append({"station_id": st["station_id"], "expected_hours": len(hours), "missing_hours": total,
                    "percent_missing": round(100 * total / len(hours), 1), "gaps": len(lengths),
                    "percent_of_missing_in_short_gaps": round(100 * short / total, 1) if total else 0.0,
                    "percent_of_missing_in_outages": round(100 * long / total, 1) if total else 0.0,
                    "longest_gap_days": round(lengths.max() / 24, 1) if len(lengths) else 0.0})
    return pd.DataFrame(out).set_index("station_id")


def draw_heatmap(table: pd.DataFrame, stations: pd.DataFrame) -> None:
    """Heatmap of percent valid hours, stations down the side and years across; cells under the target are outlined."""
    plt = get_pyplot()
    order = stations.sort_values(["source", "city", "station_id"])
    grid = table.pivot(index="station_id", columns="year", values="percent_valid").loc[order["station_id"]]
    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.grid(False)
    image = ax.imshow(np.ma.masked_invalid(grid.to_numpy()), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    names = order.set_index("station_id")["station_name"]
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels([f"{sid}  {names[sid]}" + ("  (held out)" if sid in HELD_OUT_STATIONS else "") for sid in grid.index])
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.iloc[i, j]
            if np.isnan(value):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", edgecolor=INK_SECONDARY, linewidth=0))
                continue
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7.5, color="white" if value > 55 else INK)
            if value < 100 * COMPLETENESS_THRESHOLD:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))
    ax.set_title("Share of expected hourly PM2.5 that is valid, by station and year (%)", loc="left")
    ax.set_xlabel("Year (UTC). Hatched = before the station's first expected date. Outlined = below 75%.")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("% of expected hours valid")
    colorbar.outline.set_visible(False)
    fig.patch.set_facecolor(SURFACE)
    save_figure(fig, "01_missingness_heatmap.png")


def main():
    """Build the completeness table, the gap summary and the heatmap, then print the conclusion."""
    canonical = load_canonical()
    stations = load_station_reference()
    out = report_dir("eda/missingness")
    table = completeness(canonical, stations)
    gaps = gap_summary(canonical, stations)
    table.to_csv(out / "completeness_by_station_year.csv", index=False)
    gaps.to_csv(out / "gaps_by_station.csv")
    draw_heatmap(table, stations)

    low = table[table["percent_valid"] < 100 * COMPLETENESS_THRESHOLD]
    print("\n--- Station-years below the 75% target ---")
    print(low.to_string(index=False))
    print("\n--- Gaps by station (all missing hours from the station's first expected hour to the end of 2025) ---")
    print(gaps)
    total_missing = int(gaps["missing_hours"].sum())
    in_short = int((gaps["missing_hours"] * gaps["percent_of_missing_in_short_gaps"] / 100).sum())
    in_outages = int((gaps["missing_hours"] * gaps["percent_of_missing_in_outages"] / 100).sum())
    print_conclusion(
        finding=(f"{len(low)} of {int(table['percent_valid'].notna().sum())} station-years are below 75% valid; of {total_missing:,} "
                 f"missing hours, {100 * in_short / total_missing:.0f}% sit in gaps of {SHORT_GAP_HOURS} hours or less and "
                 f"{100 * in_outages / total_missing:.0f}% in outages of {LONG_GAP_HOURS} hours or more."),
        decision="No imputation or interpolation: keep missing hours missing, keep the 2-of-3 block rule, and keep the 75% target as a "
                 "flag. Report the flagged station-years (and that Copenhagen 2025 and Kilmainham 2025 give little or no test data) "
                 "next to every score.",
        limitation="Percentages count hours, not information: a station can be 80% complete and still miss every winter episode. "
                   "Expected hours come from availability_start, which is itself a reference-table judgement.")


if __name__ == "__main__":
    main()
