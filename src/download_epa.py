"""
download_epa.py - STEP 1 (data access): download the EPA Ireland (airquality.ie) raw files.

WHAT IT DOES
    For each of the six Irish stations in metadata/station_reference.csv (rows with
    source == "epa") it requests the airquality.ie readings page for every window of
    WINDOW_DAYS days from the station's availability_start to END_DATE, reads the readings
    out of the page, and saves them as a CSV in
        raw_data/epa/<raw_folder>/<station><ddmmyy>-<ddmmyy>.csv
    for example raw_data/epa/dublin/dublinRathmines/rathmines010122-020722.csv.
    That is the same folder layout and file-name convention as the manual download guide, so
    canonicalize_raw_data.py reads the files without changes. Files that already exist are
    skipped, so the manual downloads are never overwritten.

HOW airquality.ie WORKS (found with the browser's Network tab and then checked)
    https://airquality.ie/readings?station=EPA-21&dateFrom=01+Jan+2022&dateTo=02+Jul+2022
    is an ordinary page request. The server writes the readings straight into the HTML, inside
    the Highcharts chart definition, as one block per pollutant:
        name: 'PM<sub>2.5</sub>', ... data: [ [ Date.UTC(2022,0,4,20,0,0), 27 ], ... ]
    (JavaScript months start at 0.) There is no separate data request, which is why nothing from
    airquality.ie appears under the Network tab's Fetch/XHR filter. The "Download CSV" item in
    the chart menu builds the manual CSV in the browser from this same data. This script does
    the same: it turns the series into the manual CSV layout (a "Date and Time" column, then one
    column per pollutant with the same HTML tags in the header).

WINDOWS
    The site accepts about six months per request. The manual files use windows of 183 days
    (start + 182 days, for example 01 Jan 2022 to 02 Jul 2022, then 03 Jul 2022 to 01 Jan 2023),
    so this script uses the same rule and the same file names.

HOW THE OUTPUT DIFFERS FROM THE MANUAL FILES
    The values are identical (tested, see STATUS). The layout is not byte-identical: the browser
    export pads missing hours with empty rows stamped HH:01:00, which this script does not write,
    and missing values are written as empty cells. canonicalize_raw_data.py drops the padding
    rows anyway. Neither the manual files nor these windows contain the 00:00 reading of a
    window's first day (the site starts each window at 01:00); canonicalize_raw_data.py would
    show that as one missing hour per window boundary.

Requires: pip install requests
"""
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from config import RAW_DIR
from stations import load_station_reference

# Where the files go: raw_data/epa/. The station sub-folder (for example dublin/dublinRathmines)
# comes from the raw_folder column of station_reference.csv and is added in main().
LOCAL_RAW_DIR = RAW_DIR / "epa"

# Last day to download. 2026 is deliberately not downloaded: it is reserved for the update demo.
END_DATE = date(2025, 12, 31)

WINDOW_DAYS = 183          # days per request; the manual files use exactly this (end = start + 182 days)
MAX_RETRIES = 3            # tries per window before giving up
RETRY_DELAY_SECONDS = 3    # pause between tries
REQUEST_PAUSE_SECONDS = 2  # pause between windows, so the public server is not hammered

# {station_id} is the airquality.ie id from station_reference.csv (for example EPA-22); the dates look
# like "01 Jan 2022" (the "+" is a space in a URL).
URL_TEMPLATE = "https://airquality.ie/readings?station={station_id}&dateFrom={start}&dateTo={end}"
USER_AGENT = "SETU-course-project-PM25-forecast (student research; contact dazzmondo95@yahoo.ie)"


def windows(start: date, end: date):
    """Yield (window_start, window_end) pairs of WINDOW_DAYS days, from `start` to `end`.

    Windows follow each other with no gap and no overlap; the last one stops at `end`.
    """
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=WINDOW_DAYS - 1), end)
        yield current, window_end
        current = window_end + timedelta(days=1)


def file_name(prefix: str, start: date, end: date) -> str:
    """Name a window file like the manual ones: prefix, start ddmmyy, dash, end ddmmyy."""
    return f"{prefix}{start:%d%m%y}-{end:%d%m%y}.csv"


def file_prefix(raw_folder: str, city: str) -> str:
    """The lower-case station name used in the file names, taken from the last part of raw_folder.

    "dublin/dublinRathmines" gives "rathmines" and "cork" gives "cork": the city name is removed
    from the front when something follows it.
    """
    name = raw_folder.split("/")[-1].lower()
    city = city.lower()
    return name[len(city):] if name.startswith(city) and len(name) > len(city) else name


def fetch_page(station_id: str, start: date, end: date) -> str:
    """Return the HTML of the readings page for one window, retrying on network or server errors.

    Raises requests.RequestException (or ValueError, for a page that is neither a chart nor the site's
    "No data found" message, for example an error page that came back with HTTP 200) after MAX_RETRIES
    failed attempts.
    """
    url = URL_TEMPLATE.format(station_id=station_id,
                              start=f"{start:%d %b %Y}".replace(" ", "+"),
                              end=f"{end:%d %b %Y}".replace(" ", "+"))
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
            resp.raise_for_status()   # HTTP 4xx/5xx raises requests.HTTPError, retried below
            # a window with no readings still comes back as a normal page, with this message instead of a chart
            if "var lineChart" not in resp.text and "No data found for this station" not in resp.text:
                raise ValueError("response contains neither the chart data nor the 'no data' message")
            return resp.text
        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"Attempt {attempt} failed for {station_id} {start} ({e}), retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_error


def parse_series(html: str) -> list:
    """Read the chart series out of the page: a list of (name, {timestamp: value}) in page order.

    The name keeps its HTML tags (for example "PM<sub>2.5</sub>") because the manual CSV header does.
    Points with no value ("null") are left out.
    """
    series = []
    # one block per pollutant: name: '...' ... data: [ ... ] }
    for block in re.finditer(r"name:\s*'([^']+)'.*?data:\s*\[(.*?)\]\s*\}", html, re.S):
        points = {}
        for y, mo, d, h, mi, s, value in re.findall(
                r"\[\s*Date\.UTC\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\)\s*,\s*([-\d.eE+]+|null)\s*\]", block.group(2)):
            if value != "null":
                # JavaScript months count from 0, so add 1
                points[datetime(int(y), int(mo) + 1, int(d), int(h), int(mi), int(s))] = float(value)
        series.append((block.group(1), points))
    return series


def write_csv(series: list, dest: Path) -> None:
    """Write the series as a CSV in the manual layout, through a ".tmp" file so a crash leaves no partial file.

    One row per timestamp that has at least one value, oldest first; a pollutant with no value at that time
    gets an empty cell.
    """
    names = [name for name, _ in series]
    times = sorted({t for _, points in series for t in points})
    dest.parent.mkdir(parents=True, exist_ok=True)   # create raw_data/epa/<raw_folder>/ if needed
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    # utf-8-sig adds the byte-order mark the manual files start with. Like the manual files, the header cells and
    # timestamps are quoted and the numbers are not. (Names and timestamps contain no quote characters.)
    lines = ['"Date and Time",' + ",".join(f'"{name}"' for name in names)]
    for t in times:
        values = [f"{points[t]:.10g}" if t in points else "" for _, points in series]
        lines.append(f'"{t:%Y-%m-%d %H:%M:%S}",' + ",".join(values))
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8-sig", newline="")
    tmp.replace(dest)


def download_window(station_id: str, start: date, end: date, dest: Path) -> str:
    """Download one station-window and save it to `dest`. Returns "saved", "no data" or "failed".

    "no data" means the page loaded but holds no PM2.5 readings in that window (nothing is saved).
    "failed" means every attempt raised an error (reported, nothing is saved).
    """
    try:
        series = parse_series(fetch_page(station_id, start, end))
    except (requests.RequestException, ValueError) as e:
        print(f"FAILED after {MAX_RETRIES} attempts: {station_id} {start} to {end} - {e}")
        return "failed"
    pm25 = dict(series).get("PM<sub>2.5</sub>")
    if not pm25:   # no PM2.5 series, or one with no values
        print(f"No PM2.5 data: {station_id} {start} to {end}")
        return "no data"
    write_csv(series, dest)
    print(f"Saved {dest} ({len(pm25)} PM2.5 values)")
    return "saved"


def main():
    """Download every missing station-window file for the Irish stations and print a summary."""
    stations = load_station_reference()   # validated station table (stops with a clear message if it is malformed)
    counts = {"saved": 0, "skipped": 0, "no data": 0, "failed": 0}
    for _, st in stations[stations.source == "epa"].iterrows():
        prefix = file_prefix(st.raw_folder, st.city)
        # availability_start is each station's first available date (Galway Briarhill: 2022-12-22)
        for w_start, w_end in windows(st.availability_start.date(), END_DATE):
            dest = LOCAL_RAW_DIR / st.raw_folder / file_name(prefix, w_start, w_end)
            if dest.exists():   # re-runs skip files already downloaded (including the manual ones)
                counts["skipped"] += 1
                continue
            counts[download_window(st.source_download_id, w_start, w_end, dest)] += 1
            time.sleep(REQUEST_PAUSE_SECONDS)
    print(f"Done: {counts['saved']} saved, {counts['skipped']} skipped (already exist), "
          f"{counts['no data']} with no data, {counts['failed']} failed.")
    if counts["failed"]:
        sys.exit(1)   # a non-zero exit code lets automation notice a failed download


if __name__ == "__main__":
    main()
