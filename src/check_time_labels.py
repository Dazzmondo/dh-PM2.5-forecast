"""
Check what the `time_block` feature actually means, and whether it means the same thing all year in every time zone.

INPUT:  processed_data/features/pm25_features_with_splits.parquet   (create_splits.py)
OUTPUT: reports/time_labels/time_block_by_timezone.csv   and the same tables printed to the terminal. Nothing is changed.

Background: the 3-hour blocks are fixed on the UTC clock (00-03, 03-06, ... UTC) so that "+2 blocks = +6 hours" holds all
year. `time_block` is then computed in feature_extraction.py as the LOCAL clock hour at the block's START divided by 3,
rounded down. That is a label, not a fixed local window, and when a time zone's UTC offset changes with daylight saving the
same UTC block can receive a different label in summer. This script shows, for every time zone and UTC block, the UTC
offsets seen, the labels given, and whether the label changes.

Requires: pip install pandas pyarrow
"""
import pandas as pd

from analysis_helpers import FEATURES_PATH, report_dir


def main():
    """Print the time_block mapping per time zone and flag the zones whose label changes with daylight saving."""
    if not FEATURES_PATH.exists():
        raise SystemExit(f"{FEATURES_PATH} not found. Run the pipeline (create_splits.py) first.")
    df = pd.read_parquet(FEATURES_PATH, columns=["station_id", "timezone", "block_start_utc", "local_timestamp", "time_block", "split"])
    df["utc_hour"] = df["block_start_utc"].dt.hour
    # local offset from UTC in hours at the start of the block (local_timestamp has no time zone attached)
    df["utc_offset_hours"] = (df["local_timestamp"] - df["block_start_utc"].dt.tz_localize(None)).dt.total_seconds() / 3600

    mapping = df.groupby(["timezone", "utc_hour"]).agg(utc_offsets=("utc_offset_hours", lambda s: sorted(set(s))),
                                                       labels=("time_block", lambda s: sorted(set(s))))
    print("--- time_block per time zone and UTC block start ---")
    print(mapping.to_string())

    changing = mapping[mapping["labels"].map(len) > 1]
    print("\nTime zones and UTC blocks whose label changes between winter and summer:")
    print(changing if len(changing) else "none")
    zones_changing = sorted(changing.index.get_level_values("timezone").unique())

    training = df[df["split"].isin(["train", "dev"])]
    share = 100 * training["timezone"].isin(zones_changing).mean() if zones_changing else 0.0
    print(f"\nStations by time zone: {df.groupby('timezone')['station_id'].unique().apply(list).to_dict()}")
    print(f"Share of the training and development rows in a zone whose label changes: {share:.1f}% ({', '.join(zones_changing) or 'none'})")

    print("\nWhat label 0 covers (the 3-hour block that starts at 00:00-02:59 local), by time zone:")
    for zone, group in df.groupby("timezone"):
        first = group[group["time_block"] == 0]
        offsets = sorted(set(first["utc_offset_hours"]))
        windows = sorted({f"{int(h):02d}:00-{int(h) + 3:02d}:00 local" for h in (first["local_timestamp"].dt.hour)})
        print(f"  {zone:18s} UTC offsets {offsets}: block windows {windows}")

    out = report_dir("time_labels")
    mapping.to_csv(out / "time_block_by_timezone.csv")
    print(f"\nTable saved in {out}")


if __name__ == "__main__":
    main()
