"""
download_ukair.py - STEP 1 (data access): download the UK-AIR raw files.

WHAT IT DOES
    For each UK-AIR station in metadata/station_reference.csv (rows with
    source == "uk_air": BEL2, DERR, CLL2, KC1) it downloads the station's yearly hourly
    data file for every year in YEARS and saves it, unchanged, as
        raw_data/uk_air/<raw_folder>/<code>_<year>.RData
    for example raw_data/uk_air/london/londonKensington/KC1_2024.RData.
    The files are R data files (see "WHY .RData" below). canonicalize_raw_data.py reads them
    (parse_ukair_rdata_file) next to the CSV files of the manual download guide.

WHERE IT FITS IN THE PIPELINE
    download_ukair.py  ->  raw_data/uk_air/  ->  upload_raw_data.py (copies to GCS and
    writes the manifest)  ->  canonicalize_raw_data.py  ->  ...
    This follows the GCS lab (requests.get with a timeout, check the response, write
    the bytes to disk), with retries and file checks added. The upload to GCS is
    deliberately NOT done here: upload_raw_data.py does it for the whole raw_data
    folder, so every file also gets a checksum and a manifest record.

WHY .RData AND NOT THE CSV FILES OF THE MANUAL GUIDE
    The CSV files live under https://uk-air.defra.gov.uk/datastore/, which UK-AIR's robots.txt
    (https://uk-air.defra.gov.uk/robots.txt) disallows for automated clients ("Disallow:
    /datastore/"). Running a script against that path would ignore the rule, so this script
    does not do it. UK-AIR also publishes the same hourly data as .RData files for users of the
    R package "openair", under https://uk-air.defra.gov.uk/openair/R_data/. robots.txt does
    not disallow that path (it only names the interactive page /data/openair). The data are
    published under the Open Government Licence v3.0.
    Checked against the manual CSV files (see STATUS): the same hours, the same values.

Requires: pip install requests   (reading the files needs: pip install rdata)
"""
import sys
import time
from pathlib import Path

import requests

from config import RAW_DIR
from stations import load_station_reference

# Where the files go: raw_data/uk_air/. The station sub-folder (for example
# london/londonBloomsbury) comes from the raw_folder column of station_reference.csv
# and is added in main().
LOCAL_RAW_DIR = RAW_DIR / "uk_air"

# The years to download. range() stops BEFORE its second number, so
# range(2018, 2026) is 2018, 2019, ..., 2025: eight files per station. 
# Tested: without the 2025 files create_splits.py stops with
# "held-out stations with no 2025 test rows" for Derry and both London stations,
# because 2025 is the held-out test year.
YEARS = range(2018, 2026)

MAX_RETRIES = 3            # tries per file before giving up
RETRY_DELAY_SECONDS = 3    # pause between tries (also gives a busy server time to recover)
REQUEST_PAUSE_SECONDS = 2  # pause between files, so the public server is not hammered

# {site} is the source_download_id from station_reference.csv (BEL2, DERR, CLL2, KC1).
URL_TEMPLATE = "https://uk-air.defra.gov.uk/openair/R_data/{site}_{year}.RData"
USER_AGENT = "SETU-course-project-PM25-forecast (student research; contact dazzmondo95@yahoo.ie)"


def looks_like_valid_rdata(content: bytes) -> bool:
    """Cheap sanity check on a downloaded file before it is saved.

    The files are gzip-compressed (they start with the bytes 1f 8b), and a real yearly file is
    hundreds of KB. Anything else, such as an HTML error page that came back with HTTP 200, is rejected.
    canonicalize_raw_data.py does the real parsing and checks later.
    """
    return len(content) >= 100_000 and content[:2] == b"\x1f\x8b"


def download_station_year(site_code: str, year: int, dest: Path) -> str:
    """Download one station-year file to `dest`. Returns "saved", "missing" or "failed".

    "missing" means UK-AIR says the file does not exist (404): not an error to retry.
    "failed" means every attempt failed (reported, nothing is saved).
    """
    url = URL_TEMPLATE.format(site=site_code, year=year)
    # The file is first written to "<name>.RData.tmp" and renamed only when complete, so a crash or a
    # lost connection can never leave a half-written .RData for the pipeline to read.
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)   # timeout so a stalled connection cannot hang the run
            if resp.status_code == 404:
                return "missing"
            resp.raise_for_status()   # any other 4xx/5xx raises requests.HTTPError, which is retried below
            if not looks_like_valid_rdata(resp.content):
                raise ValueError("response doesn't look like a real .RData file")
            dest.parent.mkdir(parents=True, exist_ok=True)   # create raw_data/uk_air/<raw_folder>/ if needed
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            print(f"Saved {dest}")
            return "saved"
        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"Attempt {attempt} failed for {site_code}_{year} ({e}), retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
    print(f"FAILED after {MAX_RETRIES} attempts: {site_code}_{year} - {last_error}")
    tmp.unlink(missing_ok=True)   # clean up any partial .tmp file
    return "failed"


def main():
    """Download every missing station-year file for the UK-AIR stations and print a summary."""
    stations = load_station_reference()   # validated station table (stops with a clear message if it is malformed)
    counts = {"saved": 0, "skipped": 0, "missing": 0, "failed": 0}
    for _, st in stations[stations.source == "uk_air"].iterrows():
        for year in YEARS:
            # st.raw_folder, e.g. "london/londonKensington"; st.source_download_id, e.g. "KC1"
            dest = LOCAL_RAW_DIR / st.raw_folder / f"{st.source_download_id}_{year}.RData"
            manual_csv = dest.with_suffix(".csv")
            # A year that already exists (as .RData, or as a CSV from the manual guide) is skipped. The CSV
            # case matters: the CSV and the .RData file of the same year differ in the last decimal for a
            # few values (see canonicalize_raw_data.py), and the pipeline stops if both are present.
            # RISK: an existing file is never refreshed (see the module docstring).
            if dest.exists() or manual_csv.exists():
                counts["skipped"] += 1
                continue
            result = download_station_year(st.source_download_id, year, dest)
            counts[result] += 1
            if result == "missing":
                print(f"No data file for {st.station_id} ({st.source_download_id}) in {year}")
            time.sleep(REQUEST_PAUSE_SECONDS)
    print(f"Done: {counts['saved']} saved, {counts['skipped']} skipped (already exist), "
          f"{counts['missing']} missing on the server, {counts['failed']} failed.")
    if counts["failed"]:
        sys.exit(1)   # a non-zero exit code lets automation notice a failed download


if __name__ == "__main__":
    main()
