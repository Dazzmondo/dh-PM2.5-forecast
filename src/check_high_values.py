"""
Show the evidence behind the decision on each PM2.5 value above 500 ug/m3 (the rule is in canonicalize_raw_data.py).

INPUT:  processed_data/quality_reports/high_value_audit.csv   (written by canonicalize_raw_data.py: value and decision)
        processed_data/canonical_hourly.parquet                (the neighbouring hours; the spike itself is already missing)
        raw_data/eea/                                          (optional: the EEA Validity / Verification flags)
OUTPUT: reports/high_values/high_value_evidence.csv   and the same evidence printed per value. Nothing is changed.

For each value it prints: the station and time, which split the hour belongs to, the station's own hours -6..+6,
the other stations of the same city or country within +/-3 h, the source's quality flags when the source has any,
and how far ONE hour moves the 3-hour block mean (the target is a 3-hour mean). It then repeats a simple
persistence forecast on the development year with the spikes put back, to show the choice does not depend on them.

The rule (a judgement): an isolated value that no other station confirms is set to missing; a value that is sustained
AND confirmed is kept. Corroboration is evidence, not proof. No model result is used to decide whether a value is real.

Requires: pip install pandas pyarrow
"""
import pandas as pd

from analysis_helpers import features_from_hourly, load_canonical, report_dir, verify_against_pipeline
from canonicalize_raw_data import AUDIT_PATH, CORROBORATION_FACTOR, MEDIAN_UNTIL_UTC, SUSTAINED_LEVEL
from config import RAW_DIR
from create_splits import DEV_YEAR, HELD_OUT_STATIONS, TEST_YEAR
from stations import load_station_reference

CONTEXT_HOURS = 6      # hours before and after shown for the station itself
NEIGHBOUR_HOURS = 3    # window used for the other stations (the same window as the rule)


def split_of(station_id: str, timestamp: pd.Timestamp) -> str:
    """Which split an hour falls in (by station and calendar year)."""
    if station_id in HELD_OUT_STATIONS:
        return "held-out station"
    if timestamp.year == TEST_YEAR:
        return "test year (2025)"
    if timestamp.year == DEV_YEAR:
        return "development year"
    return "training years"


def eea_flags(station_code: str, timestamp: pd.Timestamp):
    """The EEA (Validity, Verification) pairs of one raw hour, or None when the raw EEA files are not available."""
    files = sorted((RAW_DIR / "eea").rglob("*.parquet"))
    if not files:
        return None
    label = (timestamp + pd.Timedelta(hours=1)).tz_localize(None)   # EEA hourly files are labelled in UTC+1
    found = []
    for path in files:
        raw = pd.read_parquet(path, columns=["Samplingpoint", "Start", "Validity", "Verification"])
        hit = raw[raw["Samplingpoint"].str.contains(station_code) & (raw["Start"] == label)]
        found += hit[["Validity", "Verification"]].drop_duplicates().values.tolist()
    return found


def show_value(row: pd.Series, canonical: pd.DataFrame, pivot: pd.DataFrame, stations: pd.DataFrame, medians: pd.Series) -> dict:
    """Print the evidence for one value above 500 and return it as one table row."""
    sid, when, value = row["station_id"], pd.Timestamp(row["timestamp_utc"]), row["value"]
    info = stations.loc[sid]
    print(f"\n{sid} {info['station_name']} ({info['source']}, {info['city']}, {info['country']})  {when}  value {value:.1f}")
    print(f"    split: {split_of(sid, when)}; station median before 2025: {medians.get(sid, float('nan')):.1f}")
    around = [pivot[sid].get(when + pd.Timedelta(hours=k), float("nan")) for k in range(-CONTEXT_HOURS, CONTEXT_HOURS + 1)]
    around[CONTEXT_HOURS] = value
    print("    same station, hours -6..+6:", " ".join("nan" if pd.isna(x) else f"{x:.0f}" for x in around))
    near_text = []
    for other in pivot.columns:
        if other == sid or not (info["city"] == stations.loc[other, "city"] or info["country"] == stations.loc[other, "country"]):
            continue
        window = [pivot[other].get(when + pd.Timedelta(hours=k), float("nan")) for k in range(-NEIGHBOUR_HOURS, NEIGHBOUR_HOURS + 1)]
        top = pd.Series(window).max()
        threshold = max(SUSTAINED_LEVEL, CORROBORATION_FACTOR * medians.get(other, 0))
        same = "same city" if info["city"] == stations.loc[other, "city"] else "same country"
        print(f"    {same}: {other} ({stations.loc[other, 'station_name']}) +/-3 h max {top:.0f} (needs more than {threshold:.0f})")
        near_text.append(f"{other}:{top:.0f}")
    if not near_text:
        print("    no other station of the same city or country in the project")
    flags = eea_flags(sid if not sid.startswith("ES") else str(info["source_download_id"]), when) if info["source"] == "eea" else None
    print("    source flags:", flags if flags is not None else "none (this source has no per-value flag in the files used)")
    block = when.floor("3h")
    hours = pivot[sid].reindex(pd.date_range(block, periods=3, freq="h", tz="UTC"))
    hours[when] = value
    with_value, without = hours.mean(), hours.drop(when).mean()
    print(f"    its 3-hour block: mean {with_value:.1f} with the value, {without:.1f} without; one hour moves the block mean by "
          f"{with_value - without:.1f}")
    print(f"    rule: sustained={row['sustained']}, corroborated={row['corroborated']} -> {row['action']}")
    return {"station_id": sid, "timestamp_utc": when, "value": value, "split": split_of(sid, when),
            "block_mean_with": round(with_value, 1), "block_mean_without": round(without, 1),
            "other_stations_max_pm25": " ".join(near_text), "eea_flags": str(flags), "sustained": row["sustained"],
            "corroborated": row["corroborated"], "action": row["action"]}


def sensitivity(canonical: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Persistence forecast on the development year with the checked values missing (as stored) and put back."""
    restored = canonical.copy()
    keys = restored.set_index(["station_id", "timestamp_utc"]).index
    for _, row in audit.iterrows():
        restored.loc[keys.get_loc((row["station_id"], row["timestamp_utc"])), "pm25"] = row["value"]
    print("\n--- Sensitivity: persistence forecast on the development year ---")
    for label, hourly in (("values missing (as stored)", canonical), ("values put back", restored)):
        rows = features_from_hourly(hourly[~hourly["station_id"].isin(HELD_OUT_STATIONS)])
        dev = rows[rows["split"] == "dev"]
        train = rows[rows["split"] == "train"]
        print(f"{label:28s} train rows {len(train):,}; dev rows {len(dev):,}; dev MAE "
              f"{(dev['pm25_current'] - dev['target_pm25']).abs().mean():.4f}")


def main():
    """Print the evidence for every value that the rule checked, and save it."""
    if not AUDIT_PATH.exists():
        raise SystemExit(f"{AUDIT_PATH} not found. Run canonicalize_raw_data.py first.")
    audit = pd.read_csv(AUDIT_PATH, parse_dates=["timestamp_utc"])
    if audit.empty:
        raise SystemExit("The audit file is empty: no value above the threshold was checked.")
    audit["timestamp_utc"] = pd.to_datetime(audit["timestamp_utc"], utc=True)
    canonical = load_canonical()
    verify_against_pipeline(canonical)
    stations = load_station_reference().set_index("station_id")
    pivot = canonical.pivot_table(index="timestamp_utc", columns="station_id", values="pm25")
    for _, checked in audit.iterrows():   # the checked values are missing in the canonical table: show them as they were
        pivot.loc[checked["timestamp_utc"], checked["station_id"]] = checked["value"]
    medians = canonical[canonical["timestamp_utc"] < MEDIAN_UNTIL_UTC].groupby("station_id")["pm25"].median()
    print(f"{len(audit)} value(s) above the threshold were checked by canonicalize_raw_data.py.")
    rows = [show_value(row, canonical, pivot, stations, medians) for _, row in audit.iterrows()]
    out = report_dir("high_values")
    pd.DataFrame(rows).to_csv(out / "high_value_evidence.csv", index=False)
    sensitivity(canonical, audit)
    print(f"\nTable saved in {out}")


if __name__ == "__main__":
    main()
