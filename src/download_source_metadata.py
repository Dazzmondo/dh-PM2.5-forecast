"""
download_source_metadata.py - STEP 1 (data access): download the EEA station metadata, one CSV per country.

WHAT IT DOES
    For each country that has stations in metadata/station_reference.csv (DK, ES, FR, GB, GR, IE) it
    downloads the EEA air-quality metadata table for that country and saves it, unchanged, as
        metadata/source_metadata/<country>.csv
    for example metadata/source_metadata/denmark.csv (the United Kingdom is uk.csv).
    These files are where the station IDs, coordinates and station types in station_reference.csv came
    from (its coordinates_source column names them). No pipeline script reads them: they document
    where the station table comes from, and are the starting point if a station has to be added.

HOW IT WORKS
    The EEA "Air Quality Viewer" (https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements)
    lets a person choose a Country filter and press "Download CSV". The page does that with one request:
    a POST to .../AQViewer/download?fqn=Airquality_Dissem.b2g.measurements&f=csv whose JSON body holds the
    filter (Country = Denmark, for example). The server answers with a zip file that holds one file,
    DataExtract.csv. This script sends the same request for each country and saves DataExtract.csv under
    the country's name. The EEA server has no robots.txt.

    Unlike the other download scripts this one writes to metadata/source_metadata/, not to raw_data/,
    because the files describe the stations rather than measure anything.

STATUS
    [TESTED] Run against the live server for all six countries into a scratch folder. denmark.csv and
        greece.csv are byte-for-byte identical to the files that had been saved from the viewer by hand. The
        other four are larger or updated: france.csv, spain.csv and ireland.csv hold more rows than the
        hand-saved files (those had been narrowed by further filters, for example Spain to PM2.5 only), and
        uk.csv has the same 11,742 rows with different text in 994 rows of two columns (Measurement Equipment,
        Sampling Method). For all 17 stations the PM2.5 latitude and longitude in the new files equal those in
        station_reference.csv.
    [RISK] The tables change when the EEA updates its reporting (the files say "report year 2025"). Existing
        files are never refreshed, so delete a file to download a newer version; station_reference.csv would
        then have to be checked against it. The country files are large (spain.csv is about 18 MB).

Requires: pip install requests
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import requests

from config import METADATA_DIR
from stations import load_station_reference

# Where the files go.
LOCAL_METADATA_DIR = METADATA_DIR / "source_metadata"

BASE_URL = "https://discomap.eea.europa.eu/App/AQViewer/"
TABLE = "Airquality_Dissem.b2g.measurements"   # the table the viewer shows (its "fqn")

# Country code in station_reference.csv -> (country name in the viewer's Country filter, file name without .csv).
# A station in another country needs a line here: main() stops and says so.
COUNTRIES = {
    "DK": ("Denmark", "denmark"),
    "ES": ("Spain", "spain"),
    "FR": ("France", "france"),
    "GB": ("United Kingdom", "uk"),
    "GR": ("Greece", "greece"),
    "IE": ("Ireland", "ireland"),
}

MAX_RETRIES = 3            # tries per country before giving up
RETRY_DELAY_SECONDS = 3    # pause between tries
REQUEST_PAUSE_SECONDS = 2  # pause between countries, so the public server is not hammered
TIMEOUT_SECONDS = 300      # building a country's table can take a while on the server
USER_AGENT = "SETU-course-project-PM25-forecast (student research; contact dazzmondo95@yahoo.ie)"


def fetch_country_csv(viewer_name: str) -> bytes:
    """Return the CSV bytes of the metadata table for one country, retrying on errors.

    Raises requests.RequestException or ValueError after MAX_RETRIES failed attempts. A response that is not a
    zip file holding a CSV that starts with the "Country" column is treated as an error.
    """
    body = {"Page": 0, "SortBy": None, "SortAscending": True,
            "RequestFilter": {"Country": {"FieldName": "Country", "Values": [viewer_name]}}}
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(BASE_URL + "download", params={"fqn": TABLE, "f": "csv"}, json=body,
                                 headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()   # HTTP 4xx/5xx raises requests.HTTPError, retried below
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:   # raises zipfile.BadZipFile if it is not a zip
                content = z.read("DataExtract.csv")
            if not content.startswith(b"Country,") or content.count(b"\n") < 2:
                raise ValueError("DataExtract.csv does not look like the metadata table")
            return content
        except (requests.RequestException, zipfile.BadZipFile, ValueError, KeyError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"Attempt {attempt} failed for {viewer_name} ({e}), retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
    raise ValueError(f"{viewer_name}: {last_error}")


def save_file(content: bytes, dest: Path) -> None:
    """Write bytes to `dest` through a ".tmp" file so a crash can never leave a half-written CSV."""
    dest.parent.mkdir(parents=True, exist_ok=True)   # create metadata/source_metadata/ if needed
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(dest)


def main():
    """Download the missing country files for the countries of the project's stations and print a summary."""
    stations = load_station_reference()   # validated station table (stops with a clear message if it is malformed)
    codes = sorted(stations["country"].unique())
    unknown = [c for c in codes if c not in COUNTRIES]
    if unknown:
        raise SystemExit(f"STOPPED: no viewer country name for {unknown}: add them to COUNTRIES in this script.")
    counts = {"saved": 0, "skipped": 0, "failed": 0}
    for code in codes:
        viewer_name, stem = COUNTRIES[code]
        dest = LOCAL_METADATA_DIR / f"{stem}.csv"
        if dest.exists():   # re-runs skip files already downloaded
            counts["skipped"] += 1
            continue
        try:
            content = fetch_country_csv(viewer_name)
        except ValueError as e:
            print(f"FAILED after {MAX_RETRIES} attempts: {e}")
            counts["failed"] += 1
            continue
        save_file(content, dest)
        print(f"Saved {dest} ({len(content) / 1e6:.1f} MB)")
        counts["saved"] += 1
        time.sleep(REQUEST_PAUSE_SECONDS)
    print(f"Done: {counts['saved']} saved, {counts['skipped']} skipped (already exist), {counts['failed']} failed.")
    if counts["failed"]:
        sys.exit(1)   # a non-zero exit code lets automation notice a failed download


if __name__ == "__main__":
    main()
