"""
download_eea.py - STEP 1 (data access): download the EEA hourly PM2.5 Parquet files.

WHAT IT DOES
    Automates the manual EEA download (Air Quality Download Service, PM2.5, verified data) for the
    seven EEA stations in metadata/station_reference.csv (rows with source == "eea").
      1. It asks the EEA download API for the list of verified PM2.5 Parquet files of the
         project's countries (one request to the "ParquetFile/urls" endpoint).
      2. It keeps only the files whose name contains one of the stations' source_download_id
         (for example DK0034A or 28079038), so the other hundreds of stations are not downloaded.
      3. It downloads each of them, keeps only files that hold HOURLY values (daily files are
         reported and not saved, because canonicalize_raw_data.py cannot use them), and saves
             raw_data/eea/<raw_folder>/<file name>.parquet
         for example raw_data/eea/copenhagen/SPO-DK0034A_06001_104.parquet. That is the same folder
         layout the manual guide uses. Files that already exist are skipped, so the manual
         downloads are never overwritten.
    The API is the one behind the EEA web app (https://eeadmz1-downloads-webapp.azurewebsites.net):
    https://eeadmz1-downloads-api-appservice.azurewebsites.net, described in the EEA document
    "How to use Air Quality Downloads". Its robots.txt allows all access. No `airbase` package is
    needed, only `requests`.

WHAT TO EXPECT IN THE FILES
    The "ParquetFile/urls" endpoint ignores date filters, so every file is the station's COMPLETE
    series held by the service, not only 2018-2025 (Copenhagen's SPO-DK0034A_06001_100.parquet
    starts in 2013). That is intended: trimming to 2018-2025 is left to a later step. Rows
    before 2018 are labelled "excluded" by create_splits.py. The values on hours the manual files
    share with these files are identical (tested).

    Only VERIFIED data (dataset 2, E1a) is downloaded, like the manual route. The "up-to-date"
    dataset (dataset 1, E2a) is deliberately not used here: it is unverified and, for these
    stations, only holds 2026. It is the likely source for the 2026 update demo (a later milestone).
    Verified and up-to-date files of one sampling point have the SAME file name, so never save them
    in the same folder.

Requires: pip install requests pandas pyarrow
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from config import RAW_DIR
from stations import load_station_reference

# Where the files go: raw_data/eea/. The station sub-folder (for example madrid/madridCuatroCaminos)
# comes from the raw_folder column of station_reference.csv and is added in main().
LOCAL_RAW_DIR = RAW_DIR / "eea"

API_URL = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/urls"
DATASET_VERIFIED = 2       # EEA dataset numbers: 1 = Up-To-Date (E2a), 2 = Verified (E1a), 3 = Historical
POLLUTANT = "PM2.5"        # the EEA's name for the pollutant

MAX_RETRIES = 3            # tries per request before giving up
RETRY_DELAY_SECONDS = 3    # pause between tries
REQUEST_PAUSE_SECONDS = 1  # pause between file downloads, so the public server is not hammered
TIMEOUT_SECONDS = 180      # the URL list for four countries is large, so allow a generous wait


def request_with_retries(method: str, url: str, **kwargs) -> requests.Response:
    """Make an HTTP request, retrying on network errors and HTTP 4xx/5xx; raises the last error if all fail."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(method, url, timeout=TIMEOUT_SECONDS, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"Attempt {attempt} failed for {url} ({e}), retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_error


def list_file_urls(countries: list) -> list:
    """Return the URLs of all verified PM2.5 Parquet files for these country codes.

    The response is text with a header line first, then one URL per line (EEA document p.16), so only
    lines starting with "http" are kept.
    """
    body = {"countries": countries, "cities": [], "pollutants": [POLLUTANT],
            "dataset": DATASET_VERIFIED, "source": "Custom script"}
    resp = request_with_retries("POST", API_URL, json=body)
    return [line.strip() for line in resp.text.splitlines() if line.strip().startswith("http")]


def is_hourly(content: bytes) -> bool:
    """True if the Parquet file holds only hourly values (its AggType column says "hour" on every row).

    canonicalize_raw_data.py skips daily files and stops on a file that mixes hourly and other rows, so
    files that are not purely hourly are not saved.
    """
    df = pd.read_parquet(io.BytesIO(content), columns=["AggType"])
    return set(df["AggType"].astype(str).str.lower().unique()) == {"hour"}


def save_file(content: bytes, dest: Path) -> None:
    """Write bytes to `dest` through a ".tmp" file so a crash can never leave a half-written Parquet file."""
    dest.parent.mkdir(parents=True, exist_ok=True)   # create raw_data/eea/<raw_folder>/ if needed
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(dest)


def main():
    """Download every missing hourly file of the project's EEA stations and print a summary."""
    stations = load_station_reference()   # validated station table (stops with a clear message if it is malformed)
    eea = stations[stations.source == "eea"]
    countries = sorted(eea.country.unique())
    urls = list_file_urls(countries)
    print(f"The EEA lists {len(urls)} verified {POLLUTANT} files for {countries}.")

    counts = {"saved": 0, "skipped": 0, "not hourly": 0, "failed": 0}
    hourly_found = set()   # station_ids that ended up with at least one hourly file on disk
    for _, st in eea.iterrows():
        # the file name carries the station code, e.g. SPO-DK0034A_06001_104.parquet or SP_28079038_9_47.parquet
        mine = [u for u in urls if st.source_download_id in u.split("/")[-1]]
        if not mine:
            print(f"NOT FOUND: no file for {st.station_id} ({st.source_download_id}) in the EEA list")
        for url in mine:
            name = url.split("/")[-1]
            dest = LOCAL_RAW_DIR / st.raw_folder / name
            if dest.exists():   # re-runs skip files already downloaded (including the manual ones)
                counts["skipped"] += 1
                hourly_found.add(st.station_id)
                continue
            try:
                content = request_with_retries("GET", url).content
                hourly = is_hourly(content)
            except (requests.RequestException, ValueError, KeyError, OSError) as e:
                print(f"FAILED: {name} - {e}")
                counts["failed"] += 1
                continue
            if not hourly:
                print(f"Not saved (not purely hourly values): {name}")
                counts["not hourly"] += 1
            else:
                save_file(content, dest)
                print(f"Saved {dest} ({len(content) / 1e6:.1f} MB)")
                counts["saved"] += 1
                hourly_found.add(st.station_id)
            time.sleep(REQUEST_PAUSE_SECONDS)

    missing = sorted(set(eea.station_id) - hourly_found)
    print(f"Done: {counts['saved']} saved, {counts['skipped']} skipped (already exist), "
          f"{counts['not hourly']} not hourly, {counts['failed']} failed.")
    if missing:
        print(f"STOPPED: no hourly file for {missing}")
    if counts["failed"] or missing:
        sys.exit(1)   # a non-zero exit code lets automation notice a failed download


if __name__ == "__main__":
    main()
