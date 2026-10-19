"""
EDA 3: do PM2.5 levels follow the calendar and the clock? Seasonal (by month) and diurnal (by local hour) patterns.

INPUT:  processed_data/canonical_hourly.parquet   (hourly values; training years only, non-held-out stations)
OUTPUT: reports/figures/03_seasonal_pattern.png, reports/figures/04_diurnal_pattern.png
        reports/eda/seasonal_diurnal/*.csv   and the same tables printed

Question: how large are the seasonal and daily cycles compared with the typical level, and do the stations of one country
share them? If they do, month and time of day are worth giving to a model (as cyclical features).

Local time is used for both (each station's own time zone), because heating and traffic follow the local clock. Hourly values
are used rather than the 3-hour time_block label, which is not a fixed local window (see check_time_labels.py).
Training years only: 2018-2023 (Irish stations 2022-2023, so their monthly curves rest on two years).

Requires: pip install pandas pyarrow numpy matplotlib
"""
import pandas as pd

from analysis_helpers import (BLUE, INK_SECONDARY, ORANGE, SOURCE_COLORS, get_pyplot, in_decision_pool, load_canonical,
                              print_conclusion, report_dir, save_figure)
from create_splits import DEV_YEAR

WINTER_MONTHS = (12, 1, 2)
SUMMER_MONTHS = (6, 7, 8)
COUNTRY_NAMES = {"DK": "Denmark", "ES": "Spain", "GB": "Great Britain (Belfast)", "GR": "Greece", "IE": "Ireland"}


def add_local_time(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add local_month and local_hour, using each station's own time zone (so daylight saving is handled)."""
    parts = []
    for zone, group in hourly.groupby("timezone"):
        group = group.copy()
        local = group["timestamp_utc"].dt.tz_convert(zone)
        group["local_month"], group["local_hour"] = local.dt.month, local.dt.hour
        parts.append(group)
    return pd.concat(parts)


def amplitude(profile: pd.Series) -> float:
    """Spread of a profile (highest minus lowest value) as a share of its average: 0.3 means a 30% swing."""
    return float((profile.max() - profile.min()) / profile.mean())


def main():
    """Build the monthly and hourly profiles, draw both figures and print the conclusion."""
    canonical = load_canonical()
    pool = canonical[in_decision_pool(canonical, "timestamp_utc") & (canonical["timestamp_utc"].dt.year < DEV_YEAR)
                     & canonical["pm25"].notna()]
    hourly = add_local_time(pool)
    out = report_dir("eda/seasonal_diurnal")

    monthly = hourly.groupby(["country", "station_id", "local_month"])["pm25"].median().unstack("local_month")
    winter = hourly[hourly["local_month"].isin(WINTER_MONTHS)].groupby(["country", "station_id", "local_hour"])["pm25"].median().unstack("local_hour")
    summer = hourly[hourly["local_month"].isin(SUMMER_MONTHS)].groupby(["country", "station_id", "local_hour"])["pm25"].median().unstack("local_hour")
    summary = pd.DataFrame({
        "seasonal_amplitude": monthly.apply(amplitude, axis=1),
        "diurnal_amplitude_winter": winter.apply(amplitude, axis=1),
        "diurnal_amplitude_summer": summer.apply(amplitude, axis=1),
        "highest_month": monthly.idxmax(axis=1),
        "lowest_month": monthly.idxmin(axis=1),
        "evening_peak_hour_winter": winter.loc[:, 12:23].idxmax(axis=1),
    }).round(2)
    print("--- Cycle sizes per station (training years; amplitude = (max - min) / mean of the median profile) ---")
    print(summary)
    monthly.to_csv(out / "monthly_median_by_station.csv")
    winter.to_csv(out / "hourly_median_winter_by_station.csv")
    summer.to_csv(out / "hourly_median_summer_by_station.csv")
    summary.to_csv(out / "cycle_summary_by_station.csv")

    plt = get_pyplot()
    countries = list(COUNTRY_NAMES)
    source_of = hourly.groupby("country")["source"].first()
    fig, axes = plt.subplots(1, len(countries), figsize=(11, 3.3), sharex=True)
    for axis, country in zip(axes, countries):
        for _, row in monthly.loc[country].iterrows():
            axis.plot(row.index, row.values, color=SOURCE_COLORS[source_of[country]], alpha=0.75)
        axis.set_title(COUNTRY_NAMES[country], loc="left", fontsize=9)
        axis.set_xticks([1, 4, 7, 10])
        axis.set_xlabel("local month")
    axes[0].set_ylabel("median hourly PM2.5 (ug/m3)")
    fig.suptitle("Seasonal pattern: median hourly PM2.5 by local month, one line per station (training years)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "03_seasonal_pattern.png")

    fig, axes = plt.subplots(1, len(countries), figsize=(11, 3.3), sharex=True)
    for axis, country in zip(axes, countries):
        for label, table, color in (("winter (Dec-Feb)", winter, BLUE), ("summer (Jun-Aug)", summer, ORANGE)):
            curve = table.loc[country].median()   # the median of the stations' own median profiles
            axis.plot(curve.index, curve.values, color=color, label=label)
        axis.set_title(COUNTRY_NAMES[country], loc="left", fontsize=9)
        axis.set_xticks([0, 6, 12, 18])
        axis.set_xlabel("local hour")
        axis.tick_params(colors=INK_SECONDARY)
    axes[0].set_ylabel("median hourly PM2.5 (ug/m3)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Diurnal pattern: median hourly PM2.5 by local hour (median across each country's stations, training years)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "04_diurnal_pattern.png")

    seasonal_ok = int((summary["seasonal_amplitude"] >= 0.25).sum())
    diurnal = summary[["diurnal_amplitude_winter", "diurnal_amplitude_summer"]].max(axis=1)
    print_conclusion(
        finding=(f"{seasonal_ok} of {len(summary)} stations have a seasonal swing of at least 25% of their level (median {summary['seasonal_amplitude'].median():.0%}); "
                 f"the daily swing is {diurnal.median():.0%} (median over stations) and its shape differs between winter and summer."),
        decision="Give models month and local hour as cyclical (sin/cos) features built from local_timestamp, with the season "
                 "label dropped (it repeats the month). Keep the cycle features in the baseline set; test in M3 that they help.",
        limitation="Monthly curves for the Irish stations rest on only two years, so a single unusual winter can shape them, and a swing "
                   "measured against a small level (Cork, median 3 ug/m3) looks very large. Medians hide the high episodes that matter most, and country panels pool stations of different types.")


if __name__ == "__main__":
    main()
