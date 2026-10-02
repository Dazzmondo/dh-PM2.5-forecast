"""
Loader + validator for station_reference.csv - the ONE place station
facts live (IDs, coordinates, timezone, which source supplies the data,
and the window we expect data for).

station_id is the EU/EIONET station code (e.g. "IE0028A"). It is the
primary key used by every script and can serve as the primary/foreign key in
any database built from this project. Every OTHER identifier (EPA's "EPA-22", UK-AIR's "BEL2"
download code, UK-AIR's "UKA00212") is only a lookup attribute against
that key - see the column notes below.

Columns
-------
station_id           EU/EIONET code. Primary key.
station_name         Human-readable name (display only).
city, country        country is ISO-2. city groups stations for the report.
timezone             IANA name; used to derive local time-of-day features.
latitude, longitude  Decimal degrees (WGS84), from EEA station metadata.
source               Which raw source supplies this station: epa|uk_air|eea.
raw_folder           Where this station's files sit under raw_data/<source>/,
                     with forward slashes (e.g. "london/londonBloomsbury",
                     or just "cork" when the city has one station). This is
                     how EPA and UK-AIR files are matched to a station, since
                     their files don't contain the EU code. Folders can keep
                     human-readable names; the mapping lives here, in one place.
source_download_id   The ID that source's download mechanism needs.
                     Its meaning depends on `source`: EPA-22 / BEL2 for EPA and
                     UK-AIR; for EEA stations it is the station code exactly as it
                     appears inside the files' Samplingpoint column - the EU code
                     (DK0034A) for most countries, but Spain's files only carry the
                     national code (28079038). Never compare it across sources.
uk_air_internal_id   UK-AIR "UKA..." ID. Reference only; blank for non-UK.
source_name_check    UK-AIR only: lowercase keyword that must appear in the
                     site name printed inside each downloaded file. Guards
                     against a file sitting under the wrong station.
availability_start   First date we expect hourly PM2.5 from our chosen
                     source. Explicit on purpose: expected coverage is never
                     inferred from what happens to be in the data.
coordinates_source   Where the coordinates came from (provenance).

Requires: pip install pandas
"""
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config import REFERENCE_PATH

# Columns the CSV must contain.
REQUIRED_COLUMNS = [
    "station_id", "station_name", "city", "country", "timezone", "latitude",
    "longitude", "source", "source_download_id", "raw_folder", "uk_air_internal_id",
    "source_name_check", "availability_start", "coordinates_source",
]
# Columns that must be filled for EVERY station. The two left out (uk_air_internal_id,
# source_name_check) only apply to UK-AIR stations.
NON_BLANK_COLUMNS = [
    "station_id", "station_name", "city", "country", "timezone", "latitude",
    "longitude", "source", "source_download_id", "raw_folder", "availability_start",
    "coordinates_source",
]
VALID_SOURCES = {"epa", "uk_air", "eea"}


def load_station_reference(path: Path = REFERENCE_PATH) -> pd.DataFrame:
    """Load station_reference.csv and fail loudly, listing EVERY problem at
    once, if it is malformed."""
    # Read everything as text (so codes keep exactly what is written) and keep blanks as "" rather than NaN.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    problems = []   # every problem found is collected here and reported together at the end

    # 1. The expected columns exist. Stop at once if not: nothing else can be checked without them.
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"{path}: missing columns {missing_cols}")

    # 2. No required cell is blank.
    for col in NON_BLANK_COLUMNS:
        blanks = raw.loc[raw[col].str.strip() == "", "station_id"].tolist()
        if blanks:
            problems.append(f"blank '{col}' for: {blanks}")

    # 3. Each station_id appears once (it is the primary key).
    if raw["station_id"].duplicated().any():
        problems.append(f"duplicate station_id: {raw.loc[raw['station_id'].duplicated(), 'station_id'].tolist()}")

    # 4. source is one of epa / uk_air / eea.
    bad_source = raw.loc[~raw["source"].isin(VALID_SOURCES), "station_id"].tolist()
    if bad_source:
        problems.append(f"source not in {sorted(VALID_SOURCES)} for: {bad_source}")

    # 5. Two stations of the same source never share a download ID.
    dup_dl = raw.duplicated(subset=["source", "source_download_id"], keep=False)
    if dup_dl.any():
        problems.append(f"duplicate (source, source_download_id): {raw.loc[dup_dl, 'station_id'].tolist()}")

    # 6. Every UK-AIR station has the keyword used to verify the site name inside its files.
    ukair_no_check = raw.loc[(raw["source"] == "uk_air") & (raw["source_name_check"].str.strip() == ""), "station_id"].tolist()
    if ukair_no_check:
        problems.append(f"uk_air stations need a source_name_check keyword: {ukair_no_check}")

    # 7. Folders: normalise to forward slashes, then make sure no station's folder equals, or sits
    #    inside, another station's folder (the folder is how EPA/UK-AIR files are matched to stations).
    folders = raw.assign(raw_folder=raw["raw_folder"].str.replace("\\", "/", regex=False).str.strip("/ "))
    for source, grp in folders.groupby("source"):
        pairs = list(zip(grp["station_id"], grp["raw_folder"]))
        for a_id, a in pairs:
            for b_id, b in pairs:
                if a_id != b_id and (a == b or b.startswith(a + "/")):
                    problems.append(f"{source}: raw_folder {a!r} ({a_id}) overlaps {b!r} ({b_id}) - "
                                    f"each station needs its own non-nested folder")

    # 8. Every timezone is a valid IANA name (e.g. Europe/Dublin).
    for tz in raw["timezone"].unique():
        try:
            ZoneInfo(tz)
        except Exception:
            problems.append(f"unknown IANA timezone: {tz!r}")

    # 9. Convert to proper types. Anything unparseable becomes NaN and is reported just below.
    df = raw.copy()
    df["raw_folder"] = folders["raw_folder"]   # keep the cleaned folder names
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["availability_start"] = pd.to_datetime(df["availability_start"], errors="coerce")
    for col in ["latitude", "longitude", "availability_start"]:
        bad = df.loc[df[col].isna(), "station_id"].tolist()
        if bad:
            problems.append(f"unparseable '{col}' for: {bad}")
    if (~df["latitude"].between(-90, 90)).any() or (~df["longitude"].between(-180, 180)).any():
        problems.append("latitude/longitude outside valid range")

    # Report ALL problems together, so the CSV can be fixed in one pass.
    if problems:
        raise ValueError(f"{path} failed validation:\n  - " + "\n  - ".join(problems))
    return df
