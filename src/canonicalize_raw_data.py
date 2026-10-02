"""
Parse the three raw sources (EPA CSV, UK-AIR CSV, EEA Parquet) into ONE
common hourly schema, keyed on station_id (the EU/EIONET code).

INPUT:  raw_data/epa/, raw_data/uk_air/, raw_data/eea/   (original downloads)
OUTPUT: processed_data/canonical_hourly.parquet
    station_id, source, city, country, timezone, latitude, longitude,
    timestamp_utc (hourly, UTC, INTERVAL-START labelled), pm25 (ug/m3)

What it does
1. Matches every file to a station (FOLDER MAPPING below).
2. Converts timestamps to UTC and labels each hour by its START. The
   convention of each source is stated next to the constants further down.
3. Applies the source quality rules:
   - UK-AIR: uses the single "PM2.5 particulate matter" column (the files
     also hold Volatile and Non-volatile fractions) and checks the site name
     in the file header against the station the folder belongs to.
   - EEA: keeps rows whose Validity flag is 1, 2 or 3; other rows become
     missing. Files that are not hourly (AggType other than "hour") are
     skipped with a message.
   - EPA: drops the empty placeholder rows stamped HH:01:00.
4. Drops identical duplicate station-hours (e.g. overlapping download
   windows). DIFFERENT values for the same station-hour stop the run: some
   EEA stations have more than one PM2.5 sampling point over time, so
   overlap is possible and choosing one by row order would be arbitrary.
5. Stops, writing nothing, if a file cannot be parsed or matched to a
   station, or (REQUIRE_ALL_STATIONS = True) if a selected station has no
   data, so a partial dataset can never be mistaken for a complete one.

FOLDER MAPPING. EPA and UK-AIR files don't contain the EU code, so each
station's folder is declared in station_reference.csv (column raw_folder),
relative to raw_data/<source>/. Folder names stay human-readable, e.g.
    raw_data/epa/dublin/dublinRathmines/…csv      -> IE0028A
    raw_data/uk_air/london/londonBloomsbury/…csv  -> GB0566A
    raw_data/uk_air/belfast/…csv                  -> GB0567A  (one station in the city)
A CSV that sits in no mapped folder makes the run fail rather than being
skipped quietly. EEA files name their station inside the file, so that is
what identifies them; if an EEA file sits in a mapped folder but contains a
DIFFERENT station, the run fails too (a misfiled download).

Requires: pip install pandas pyarrow
"""
import re
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR
from stations import load_station_reference

LOCAL_RAW_DIR = RAW_DIR   # original downloads: epa/, uk_air/ and eea/ sub-folders
OUTPUT_PATH = PROCESSED_DIR / "canonical_hourly.parquet"

# Timestamp conventions. A row stamped 09:00 under "hour ending" covers
# 08:00-09:00; we shift to interval-START so all sources match.
UKAIR_TIMESTAMPS_ARE_HOUR_ENDING = True  # stated in every UK-AIR file header: "GMT hour ending"
EPA_TIMESTAMPS_ARE_HOUR_ENDING = True    # INFERRED, not documented: files start each day at 01:00.
#   Could be confirmed by comparing with validated EEA data for the same station.
# EEA: Start/End bound each hour (already interval-start) but the service
# labels them UTC+1 for every country, so we subtract 1h to reach true UTC.
EEA_LABEL_UTC_OFFSET_HOURS = 1
# EEA Validity codes we accept. 1 = valid. 2 = valid but below the detection
# limit (the measured value is given). 3 = valid, below the detection limit,
# value replaced by half the limit. Everything else (-1, -99, ...) is invalid
# and carries a sentinel value (0, -999, -9999) that must never be used.
EEA_VALID_CODES = (1, 2, 3)

REQUIRE_ALL_STATIONS = True  # True: a selected station with no data stops the run (use for the real build).
#                              False: warn and continue (partial test runs only).


def strip_html_tags(s: str) -> str:
    """Remove HTML tags from a column name: EPA headers look like 'PM<sub>2.5</sub>'."""
    return re.sub(r"<[^>]+>", "", s)


def matches_station_id(sid: str, text: str) -> bool:
    """True if sid appears in text as a whole segment (bounded by a
    non-alphanumeric character or string start/end), not inside a longer token."""
    # pattern: (start of text OR a non-alphanumeric char) + the code + (a non-alphanumeric char OR end of text)
    return bool(re.search(rf"(?:^|[^A-Za-z0-9]){re.escape(sid)}(?:[^A-Za-z0-9]|$)", text))


def station_for_folder(file_path: Path, source_dir: Path, folders: "pd.Series"):
    """Which station's raw_folder contains this file? (None if none does.)
    `folders` maps station_id -> raw_folder (relative to source_dir, '/' separated)."""
    rel = file_path.relative_to(source_dir).parent.as_posix()   # the file's folder, relative to raw_data/<source>/
    rel = "" if rel == "." else rel                             # a file directly in the source folder has no sub-folder
    # a file belongs to a station if its folder IS that station's raw_folder or sits inside it
    hits = [sid for sid, folder in folders.items() if rel == folder or rel.startswith(folder + "/")]
    return hits[0] if len(hits) == 1 else None   # exactly one match, otherwise the caller reports the file


def _exactly_one(candidates: list, what: str, path: Path, columns) -> str:
    """Return the single matching column name, or raise a clear error. This stops a changed file
    layout from silently picking the wrong column."""
    if len(candidates) != 1:
        raise ValueError(f"{path}: expected exactly one {what} column, found {len(candidates)} "
                         f"({candidates}); columns present: {list(columns)}")
    return candidates[0]


def parse_epa_file(path: Path) -> pd.DataFrame:
    """Parse one airquality.ie CSV into rows of (timestamp_utc, pm25, sampling_point).

    Steps: find the date and PM2.5 columns, drop the empty off-the-hour placeholder rows,
    move the hour-ending labels back to hour starts, and label the times as UTC.
    """
    # utf-8-sig drops the invisible byte-order mark these exports start with
    df = pd.read_csv(path, encoding="utf-8-sig")
    # {original column name: tidy lower-case name}; the tidy names are only used to FIND the right columns
    clean = {c: strip_html_tags(c).strip().lower() for c in df.columns}
    date_col = _exactly_one([c for c, cl in clean.items() if "date" in cl and "time" in cl],
                            "'Date and Time'", path, df.columns)
    pm_col = _exactly_one([c for c, cl in clean.items() if cl == "pm2.5"], "'PM2.5'", path, df.columns)

    # unreadable dates / non-numeric values become NaT / NaN instead of raising an error
    timestamp = pd.to_datetime(df[date_col], errors="coerce")
    pm25 = pd.to_numeric(df[pm_col], errors="coerce")

    # airquality.ie pads missing hours with EMPTY rows stamped HH:01:00 (real
    # readings are stamped HH:00:00). They carry no data, so drop them - but
    # stop if an off-hour row ever carries a value, since that would be new
    # behaviour we don't understand.
    off_hour = (timestamp.dt.minute != 0) | (timestamp.dt.second != 0)
    if (off_hour & pm25.notna()).any():
        raise ValueError(f"{path}: {int((off_hour & pm25.notna()).sum())} off-the-hour rows contain PM2.5 values; "
                         f"expected only empty placeholder rows.")
    if off_hour.any():
        print(f"  {path.name}: dropped {int(off_hour.sum())} empty placeholder rows stamped off the hour.")
    timestamp, pm25 = timestamp[~off_hour], pm25[~off_hour]   # keep only the real on-the-hour rows

    if EPA_TIMESTAMPS_ARE_HOUR_ENDING:
        timestamp = timestamp - pd.Timedelta(hours=1)
    # EPA times are labelled UTC (assumed, see the README timestamp table); "sampling_point" records which
    # file each row came from, so a conflict between two files can be traced
    out = pd.DataFrame({"timestamp_utc": timestamp.dt.tz_localize("UTC"),
                        "pm25": pm25,
                        "sampling_point": path.name})
    return out.dropna(subset=["timestamp_utc"])   # drop rows whose date could not be read


def parse_ukair_file(path: Path, expected_name_keyword: str) -> pd.DataFrame:
    """Parse one UK-AIR "All Hourly Pollutant Data" CSV into rows of (timestamp_utc, pm25, sampling_point).

    The file starts with a few lines of site information and then the data table: find where the
    table starts, check the site name matches the station's folder, then read the table.
    """
    raw_lines = path.read_text(errors="ignore").splitlines()   # plain text first, just to locate the table header
    # line number of the table header ("Date,time,..."); everything above it is site information
    header_idx = next((i for i, l in enumerate(raw_lines) if re.match(r"^Date,time", l, re.IGNORECASE)), None)
    if header_idx is None:
        raise ValueError(f"{path}: could not find a 'Date,time,...' header row.")

    # the site name sits on the line directly above the header row
    site_line = raw_lines[header_idx - 1] if header_idx > 0 else ""
    site_name = next((c.strip().strip('"') for c in site_line.split(",") if c.strip()), "")
    if expected_name_keyword.lower() not in site_name.lower():
        raise ValueError(f"{path}: file header says the site is {site_name!r} but this folder is for a "
                         f"station whose name should contain {expected_name_keyword!r}. Wrong folder or wrong code?")

    # skip the site-information lines, then read the table (odd characters are replaced rather than crashing)
    df = pd.read_csv(path, skiprows=header_idx, encoding="utf-8", encoding_errors="replace")
    clean = {c: strip_html_tags(c).strip().lower() for c in df.columns}   # tidy names, used only to find columns
    date_col = _exactly_one([c for c, cl in clean.items() if cl == "date"], "'Date'", path, df.columns)
    time_col = _exactly_one([c for c, cl in clean.items() if cl == "time"], "'time'", path, df.columns)
    # NOT just "contains pm2.5": that also matches the Non-volatile / Volatile fractions.
    pm_col = _exactly_one([c for c, cl in clean.items() if cl.startswith("pm2.5 particulate matter")],
                          "'PM2.5 particulate matter'", path, df.columns)

    # UK-AIR labels each day's last hour "24:00", which datetime parsing rejects,
    # so build the timestamp as (date at midnight) + (HH:MM as a duration);
    # "24:00" then correctly becomes the next day's 00:00.
    day = pd.to_datetime(df[date_col].astype(str), format="%d-%m-%Y", errors="coerce")
    clock = pd.to_timedelta(df[time_col].astype(str).str.strip() + ":00", errors="coerce")
    timestamp = day + clock
    if UKAIR_TIMESTAMPS_ARE_HOUR_ENDING:
        timestamp = timestamp - pd.Timedelta(hours=1)
    # UK-AIR times are GMT (= UTC, no daylight saving) and "hour ending"; the shift above has already
    # moved them to hour starts, so only the UTC label is attached here
    out = pd.DataFrame({"timestamp_utc": timestamp.dt.tz_localize("UTC"),
                        "pm25": pd.to_numeric(df[pm_col], errors="coerce"),
                        "sampling_point": path.name})
    return out.dropna(subset=["timestamp_utc"])


def parse_eea_file(path: Path, key_to_station: dict):
    """key_to_station maps the code found inside Samplingpoint (source_download_id)
    to a station_id. Returns None for files that aren't hourly."""
    df = pd.read_parquet(path)
    col = {c.lower(): c for c in df.columns}   # case-insensitive lookup: lower-case name -> the real column name
    for needed in ("samplingpoint", "start", "value"):   # the three columns every EEA file must have
        if needed not in col:
            raise ValueError(f"{path}: expected a '{needed}' column, found {list(df.columns)}.")

    # AggType says what each row is: "hour" (hourly), "day" (daily), ... Only hourly files can be used here.
    if "aggtype" in col:
        agg = set(df[col["aggtype"]].astype(str).str.lower().unique())
        if agg != {"hour"}:
            if "hour" in agg:
                raise ValueError(f"{path}: mixes hourly and non-hourly rows ({sorted(agg)}).")
            print(f"  SKIPPED (not hourly data): {path.name} holds {sorted(agg)} values. Daily values are "
                  f"in a different timezone convention and can't stand in for hourly readings.")
            return None

    # one file should hold one pollutant (PM2.5)
    if "pollutant" in col:
        pollutants = df[col["pollutant"]].dropna().unique()
        if len(pollutants) > 1:
            raise ValueError(f"{path}: contains several pollutant codes {list(pollutants)}; expected PM2.5 only.")

    def station_of(samplingpoint: str):
        """Our station_id for a sampling-point text, or None if it belongs to a station we don't use."""
        return next((sid for key, sid in key_to_station.items() if matches_station_id(key, samplingpoint)), None)

    # label each row with its station; rows of stations the project doesn't use are dropped on the next line
    df = df.assign(_station=df[col["samplingpoint"]].astype(str).map(station_of))
    df = df[df["_station"].notna()]

    # EEA marks bad hours with a Validity flag; only the codes in EEA_VALID_CODES keep their value
    value = pd.to_numeric(df[col["value"]], errors="coerce")
    if "validity" in col:
        validity = pd.to_numeric(df[col["validity"]], errors="coerce")  # handles "1" vs 1
        invalid = ~validity.isin(EEA_VALID_CODES)
        below_limit = validity.isin((2, 3))
        if invalid.any():
            print(f"  {path.name}: {int(invalid.sum())} invalid rows (Validity not in {EEA_VALID_CODES}) set to NaN.")
        if below_limit.any():
            print(f"  {path.name}: kept {int(below_limit.sum())} readings flagged valid-but-below-detection-limit.")
        value = value.where(~invalid)

    # "Start" is a plain datetime in UTC+1. Attaching "UTC" here only gives it a timezone label; the real
    # conversion to UTC is the 1-hour subtraction in the next statement.
    start = pd.to_datetime(df[col["start"]], errors="coerce").dt.tz_localize("UTC")
    return pd.DataFrame({"station_id": df["_station"],
                         "timestamp_utc": start - pd.Timedelta(hours=EEA_LABEL_UTC_OFFSET_HOURS),
                         "pm25": value,
                         "sampling_point": df[col["samplingpoint"]].astype(str)}).dropna(subset=["timestamp_utc"])


def resolve_duplicates(canonical: pd.DataFrame) -> pd.DataFrame:
    """Identical duplicate hours are dropped. Different values for the same
    (station, hour) stop the run."""
    key = ["station_id", "timestamp_utc"]   # a station-hour is identified by these two columns
    dup = canonical[canonical.duplicated(key, keep=False)]   # every row that shares its station-hour with another
    if dup.empty:
        return canonical

    # per repeated station-hour: how many DIFFERENT PM2.5 values it has (missing values are not counted)
    distinct = dup.groupby(key)["pm25"].nunique(dropna=True)
    conflicts = distinct[distinct > 1]   # more than one different value = the sources disagree
    if len(conflicts):
        examples = conflicts.reset_index().head(8)   # a few examples are enough to trace the problem to its files
        lines = []
        for _, r in examples.iterrows():
            rows = dup[(dup.station_id == r.station_id) & (dup.timestamp_utc == r.timestamp_utc)]
            lines.append(f"    {r.station_id} {r.timestamp_utc}: " +
                         ", ".join(f"{v}={s}" for v, s in zip(rows.pm25, rows.sampling_point)))
        raise SystemExit(
            f"\nSTOPPED: {len(conflicts)} station-hour(s) have DIFFERENT PM2.5 values from different files / sampling points:\n"
            + "\n".join(lines) +
            "\n  Decide a rule (e.g. prefer the newest sampling point) before continuing - don't pick by row order."
            "\n  If conflicts cluster on the last Sunday of March/October, that source may be stamping DST local time.")

    n_before = len(canonical)
    # keep a real value over a NaN when the duplicates are otherwise identical
    canonical = (canonical.sort_values(key + ["pm25"], na_position="last")
                          .drop_duplicates(key, keep="first"))
    print(f"  Dropped {n_before - len(canonical)} identical duplicate station-hours (e.g. overlapping download windows).")
    return canonical


def main():
    """Parse every raw file, join them into one hourly table, and save it (see the module docstring)."""
    stations = load_station_reference()
    frames, failures = [], []   # frames: one parsed table per file; failures: every problem found, reported together

    # --- EPA and UK-AIR: CSV files, matched to stations by the folder they sit in ---
    for source in ("epa", "uk_air"):
        src_stations = stations[stations.source == source].set_index("station_id")
        src_folders = src_stations["raw_folder"]   # station_id -> its folder
        folder = LOCAL_RAW_DIR / source
        files = sorted(folder.rglob("*.csv")) if folder.exists() else []
        for f in files:
            if "combined" in f.name.lower():   # a merged/derived file saved here by mistake; raw_data holds originals only
                print(f"  Ignoring derived file in raw folder: {f}")
                continue
            sid = station_for_folder(f, folder, src_folders)   # which station owns this file?
            if sid is None:
                failures.append(f"{f}: not inside any raw_folder listed for {source} stations in station_reference.csv "
                                f"({sorted(src_folders)})")
                continue
            try:
                parsed = (parse_epa_file(f) if source == "epa"
                          else parse_ukair_file(f, src_stations.loc[sid, "source_name_check"]))
            except ValueError as e:
                failures.append(str(e))
                continue
            parsed["station_id"] = sid   # these two parsers don't know the station, so it is added here
            frames.append(parsed)

    # --- EEA: Parquet files, matched to stations by the station code inside each file ---
    all_eea = stations[stations.source == "eea"]
    key_to_station = dict(zip(all_eea["source_download_id"], all_eea["station_id"]))   # code in the file -> station_id
    eea_dir = LOCAL_RAW_DIR / "eea"
    eea_folders = all_eea.set_index("station_id")["raw_folder"]   # only used to catch a file saved in the wrong folder
    skipped_non_hourly = []   # names of daily files, listed in a message at the end
    for f in (sorted(eea_dir.rglob("*.parquet")) if eea_dir.exists() else []):
        try:
            parsed = parse_eea_file(f, key_to_station)
        except ValueError as e:
            failures.append(str(e))
            continue
        if parsed is None:
            skipped_non_hourly.append(f.name)
            continue
        found_here = set(parsed["station_id"])   # the stations whose rows are in this file
        expected_here = station_for_folder(f, eea_dir, eea_folders)   # the station of the folder it sits in (if any)
        if not found_here:
            print(f"  WARNING: {f.name} contains none of the project's EEA stations - ignored.")
            continue
        if expected_here is not None and found_here != {expected_here}:
            failures.append(f"{f}: sits in the folder for {expected_here} but contains {sorted(found_here)} - misfiled?")
            continue
        frames.append(parsed)

    # --- stop on any problem: nothing is written, so a partial dataset can't pass for a complete one ---
    if failures:
        raise SystemExit("\nSTOPPED - these files could not be used:\n  - " + "\n  - ".join(failures))
    if not frames:
        raise SystemExit("No usable raw files found under raw_data/.")

    if skipped_non_hourly:
        print(f"\nSkipped {len(skipped_non_hourly)} non-hourly EEA file(s) (kept in raw_data, not used): {skipped_non_hourly}")

    # --- join all files, resolve duplicates, attach the station details ---
    canonical = resolve_duplicates(pd.concat(frames, ignore_index=True))
    # station details from station_reference.csv, attached to every row so later steps need no station lists
    meta = stations[["station_id", "source", "city", "country", "timezone", "latitude", "longitude"]]
    canonical = canonical.merge(meta, on="station_id", how="left")[
        ["station_id", "source", "city", "country", "timezone", "latitude", "longitude", "timestamp_utc", "pm25"]
    ].sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)

    # --- coverage: which of the selected stations produced data? ---
    found, expected = set(canonical.station_id), set(stations.station_id)
    print(f"\n--- Coverage: {len(found)} / {len(expected)} stations ---")
    print(canonical.groupby(["source", "station_id"]).size().rename("hourly_rows").to_string())
    missing = sorted(expected - found)
    if missing:
        print(f"MISSING: {missing}")
        if REQUIRE_ALL_STATIONS:
            raise SystemExit(f"STOPPED: no usable data for selected station(s) {missing} - not writing output. "
                             f"Add their raw files, or set REQUIRE_ALL_STATIONS = False for a partial test run.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(canonical)} canonical hourly rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
