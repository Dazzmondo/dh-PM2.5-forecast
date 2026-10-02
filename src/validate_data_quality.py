"""
Data-quality checks on the canonical hourly PM2.5 measurements, run BEFORE
3-hour aggregation. This is a reporting step: it prints its findings and
writes reports, and nothing is removed or changed.

INPUT:  processed_data/canonical_hourly.parquet (canonicalize_raw_data.py):
    station_id, source, city, country, timezone, latitude, longitude,
    timestamp_utc (UTC, hourly, interval-start), pm25
OUTPUT: processed_data/quality_reports/
    duplicates_report.csv    repeated (station, hour) rows (should be empty)
    completeness_report.csv  valid hours per station and year
    value_sanity_report.csv  negative values and values above OUTLIER_MAX_PM25

Checks
- Schema: UTC timestamps, numeric PM2.5, timestamps on the hour, every
  station_id known to station_reference.csv.
- Completeness: valid hours / expected hours for every station and year in
  the study window (2018-01-01 to 2025-12-31; 2026 is reserved for the
  Milestone 4 update demo). Expected hours come ONLY from availability_start
  in station_reference.csv, never from the data, so a station whose download
  lost months cannot shrink its own expected period, and a station or year
  with no rows shows up as 0%. Years below COMPLETENESS_THRESHOLD are
  flagged, not removed.
- Values: negative readings and readings above OUTLIER_MAX_PM25 are counted
  and reported only.

Requires: pip install pandas pyarrow
"""
import pandas as pd

from config import PROCESSED_DIR
from stations import load_station_reference

INPUT_PATH = PROCESSED_DIR / "canonical_hourly.parquet"
REPORT_DIR = PROCESSED_DIR / "quality_reports"

COMPLETENESS_THRESHOLD = 0.75   # a station-year with a smaller share of valid hours is flagged (not removed)
OUTLIER_MAX_PM25 = 500.0  # soft flag, never a deletion

# The study window: hours outside it are not counted
STUDY_START = pd.Timestamp("2018-01-01 00:00", tz="UTC")
STUDY_END = pd.Timestamp("2025-12-31 23:00", tz="UTC")  # last hourly interval of the study period


def check_schema(df: pd.DataFrame, stations: pd.DataFrame) -> None:
    """Print whether the table has the structure the later steps rely on
    (UTC timestamps, numeric PM2.5, hourly timestamps, known stations). Warnings only."""
    tz = getattr(df["timestamp_utc"].dt, "tz", None)   # None when the column has no timezone at all
    print("timestamp_utc is timezone-aware UTC - OK." if tz is not None and str(tz) == "UTC"
          else f"WARNING: timestamp_utc timezone is {tz}, expected UTC.")
    print("pm25 is numeric - OK." if pd.api.types.is_numeric_dtype(df["pm25"]) else "WARNING: pm25 is not numeric.")
    off_hour = int((df["timestamp_utc"].dt.minute != 0).sum())
    print("All timestamps fall on the hour - OK." if off_hour == 0 else f"WARNING: {off_hour} timestamps are not on the hour.")
    unknown = sorted(set(df["station_id"]) - set(stations["station_id"]))
    print("All station_ids exist in station_reference.csv - OK." if not unknown else f"WARNING: unknown station_ids: {unknown}")


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return the rows where the same station-hour appears more than once (there should be none)."""
    # keep=False marks EVERY copy of a repeated hour, not just the later ones
    dupes = df[df.duplicated(["station_id", "timestamp_utc"], keep=False)]
    print("No duplicate (station, hour) rows." if dupes.empty else
          f"WARNING: {len(dupes)} duplicate (station, hour) rows - canonicalize_raw_data.py should have removed these.")
    return dupes.sort_values(["station_id", "timestamp_utc"])


def check_completeness(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Share of hourly values that are valid, for every station and year it should have data.

    How many hours to EXPECT comes from availability_start in station_reference.csv, never
    from the data itself. Years below COMPLETENESS_THRESHOLD are flagged, not removed.
    """
    df = df.drop_duplicates(["station_id", "timestamp_utc"], keep="first")   # a repeated hour must not count twice
    rows = []
    for _, st in stations.iterrows():
        avail_start = max(STUDY_START, pd.Timestamp(st["availability_start"], tz="UTC"))   # first hour we expect data
        s_df = df[df["station_id"] == st["station_id"]]   # this station's rows only
        for year in range(avail_start.year, STUDY_END.year + 1):
            # the part of this year inside the expected period (shorter for the first and last year)
            win_start = max(avail_start, pd.Timestamp(f"{year}-01-01 00:00", tz="UTC"))
            win_end = min(STUDY_END, pd.Timestamp(f"{year}-12-31 23:00", tz="UTC"))
            expected = int((win_end - win_start).total_seconds() // 3600) + 1   # hourly values that window should hold
            in_win = s_df[(s_df["timestamp_utc"] >= win_start) & (s_df["timestamp_utc"] <= win_end)]
            valid = int(in_win["pm25"].notna().sum())   # hours that actually have a PM2.5 value
            rows.append({"station_id": st["station_id"], "station_name": st["station_name"], "year": year,
                         "expected_from": win_start.date(), "valid_hours": valid, "expected_hours": expected,
                         "completeness_pct": round(100 * valid / expected, 1),
                         "passes_threshold": valid / expected >= COMPLETENESS_THRESHOLD})
    report = pd.DataFrame(rows)   # one row per station-year
    failing = report[~report["passes_threshold"]]
    if failing.empty:
        print(f"All expected station-years reach {COMPLETENESS_THRESHOLD:.0%} completeness.")
    else:
        print(f"{len(failing)} expected station-year(s) below {COMPLETENESS_THRESHOLD:.0%} "
              f"(0% usually means not downloaded yet):")
        print(failing[["station_id", "station_name", "year", "valid_hours", "expected_hours", "completeness_pct"]]
              .to_string(index=False))
    print("Note: this is a diagnostic - the station set is fixed and nothing is removed automatically.")
    return report


def check_value_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """Count negative readings and very high readings. Reported only: nothing is removed."""
    negative, extreme = df[df["pm25"] < 0], df[df["pm25"] > OUTLIER_MAX_PM25]
    print(f"Negative PM2.5 values: {len(negative)}")
    print(f"Values above {OUTLIER_MAX_PM25:g} ug/m3 (flagged only): {len(extreme)}")
    return pd.concat([negative.assign(issue="negative"), extreme.assign(issue="extreme_high")])


def main():
    """Load the canonical table, run every check and write the three CSV reports."""
    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH} not found. Run canonicalize_raw_data.py first.")
    stations = load_station_reference()
    df = pd.read_parquet(INPUT_PATH)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    check_schema(df, stations)
    check_duplicates(df).to_csv(REPORT_DIR / "duplicates_report.csv", index=False)
    check_completeness(df, stations).to_csv(REPORT_DIR / "completeness_report.csv", index=False)
    check_value_sanity(df).to_csv(REPORT_DIR / "value_sanity_report.csv", index=False)
    print(f"\nReports written to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
