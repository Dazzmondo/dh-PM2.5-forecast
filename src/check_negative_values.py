"""
Check the negative PM2.5 readings: how many there are, where, how large, what the source flags say, and what each way of
handling them would do to the data and to a simple forecast.

INPUT:  processed_data/canonical_hourly.parquet   (canonicalize_raw_data.py)
        processed_data/features/pm25_features_with_splits.parquet   (only to check the rebuild below)
        raw_data/eea/                              (optional: the EEA Validity flags of the raw negative values)
OUTPUT: reports/negative_values/*.csv   and the same tables printed to the terminal. Nothing in processed_data/ is changed.

Why: a negative concentration is physically impossible, but air-quality analysers report small negatives when the true
value is near zero (instrument noise or a baseline offset). Three ways to handle them are possible and each has a cost:
set them to missing (one-sided censoring: it removes low readings only), clip them to 0 (adds an upward bias), or keep them.

Decision rule, written BEFORE the numbers were looked at (a judgement, not a fact):
  - Every choice uses only the decision pool (non-held-out stations, 2018-2024); other rows are only counted.
  - If fewer than TAIL_SHARE_LIMIT percent of the pool's negative hours lie below TAIL_LIMIT, the negatives cluster near
    zero and are treated as noise: keep them in the stored data, and clip only inside a model's own feature step.
  - The policies are compared two ways, because a cleaned target cannot be compared fairly with a raw one:
    (a) MAE against each policy's own target, (b) MAE against the unchanged ("keep") target on the rows valid under every policy.

Requires: pip install pandas pyarrow
"""
import pandas as pd

from analysis_helpers import (POOL_START, features_from_hourly, in_decision_pool, load_canonical, report_dir,
                              verify_against_pipeline)
from config import PROCESSED_DIR, RAW_DIR
from create_splits import HELD_OUT_STATIONS

TAIL_LIMIT = -5.0          # ug/m3: a negative hour below this is "in the tail"
TAIL_SHARE_LIMIT = 1.0     # percent of the pool's negative hours that may lie in the tail for "cluster near zero"
TWO_TIER_LIMITS = (2, 5, 10)   # the two-tier policy: -T <= x < 0 becomes 0, x < -T becomes missing
MAGNITUDE_BINS = [-1e9, -20, -10, -5, -2, -1, -0.5, 0]
MAGNITUDE_LABELS = ["< -20", "-20..-10", "-10..-5", "-5..-2", "-2..-1", "-1..-0.5", "-0.5..0"]
HIGH_NEGATIVE_STATION_SHARE = 2.0   # percent of a station-year's valid hours that are negative before it is listed


def apply_policy(pm25: pd.Series, name: str, limit: float = None) -> pd.Series:
    """Return the PM2.5 values after one negative-value policy: keep, nan_all, clip0 or two_tier (with `limit`)."""
    if name == "keep":
        return pm25
    if name == "nan_all":
        return pm25.mask(pm25 < 0)
    if name == "clip0":
        return pm25.clip(lower=0)
    if name == "two_tier":
        return pm25.mask(pm25 < -limit).where(~((pm25 < 0) & (pm25 >= -limit)), 0.0)
    raise ValueError(f"unknown policy {name}")


def count_by_source(canonical: pd.DataFrame, out) -> None:
    """How many valid hours are negative, per source, over all years and stations (counts only)."""
    valid = canonical[canonical["pm25"].notna()]
    table = pd.DataFrame({"valid_hours": valid.groupby("source").size(),
                          "negative_hours": valid[valid["pm25"] < 0].groupby("source").size()}).fillna(0).astype(int)
    table["percent_negative"] = (100 * table["negative_hours"] / table["valid_hours"]).round(2)
    print("\n--- Negative hours by source (all years and stations, counts only) ---")
    print(table)
    print(f"Total negative hours: {int(table['negative_hours'].sum()):,}")
    table.to_csv(out / "negatives_by_source.csv")


def magnitude_of_pool_negatives(pool_valid: pd.DataFrame, out) -> float:
    """Show how large the negatives are, and return the share (percent) of pool negatives below TAIL_LIMIT."""
    negatives = pool_valid[pool_valid["pm25"] < 0]
    print(f"\n--- Decision pool: {len(negatives):,} negative hours out of {len(pool_valid):,} valid hours ---")
    print(negatives.groupby("source")["pm25"].describe(percentiles=[0.01, 0.05, 0.5]).round(2))
    bins = pd.cut(negatives["pm25"], bins=MAGNITUDE_BINS, labels=MAGNITUDE_LABELS, right=False)
    table = pd.crosstab(bins, negatives["source"], margins=True)
    table["percent_of_pool_negatives"] = (100 * table["All"] / table.loc["All", "All"]).round(1)
    print("\nMagnitude of the pool's negatives (hours):")
    print(table)
    table.to_csv(out / "negatives_magnitude.csv")
    tail_share = 100 * float((negatives["pm25"] < TAIL_LIMIT).mean())
    print(f"\nRule check: {tail_share:.2f}% of the pool's negative hours are below {TAIL_LIMIT:g} ug/m3 "
          f"(limit {TAIL_SHARE_LIMIT:g}%) -> "
          + ("they cluster near zero: keep them in the stored data."
             if tail_share < TAIL_SHARE_LIMIT else "there is a tail: consider a two-tier rule."))
    return tail_share


def negatives_by_station(pool_valid: pd.DataFrame, out) -> None:
    """Negative share per station and per station-year in the decision pool."""
    flagged = pool_valid.assign(negative=pool_valid["pm25"] < 0)
    by_station = flagged.groupby("station_id").agg(valid_hours=("pm25", "size"), negative_hours=("negative", "sum"),
                                                   median=("pm25", "median"), minimum=("pm25", "min"))
    by_station["percent_negative"] = (100 * by_station["negative_hours"] / by_station["valid_hours"]).round(2)
    print("\n--- Negative share by station (decision pool) ---")
    print(by_station.round(2))
    by_station.to_csv(out / "negatives_by_station.csv")

    flagged = flagged.assign(year=flagged["timestamp_utc"].dt.year)
    by_year = flagged.groupby(["station_id", "year"]).agg(valid_hours=("pm25", "size"), negative_hours=("negative", "sum"))
    by_year["percent_negative"] = (100 * by_year["negative_hours"] / by_year["valid_hours"]).round(1)
    by_year.to_csv(out / "negatives_by_station_year.csv")
    print(f"\nStation-years with at least {HIGH_NEGATIVE_STATION_SHARE:g}% negative hours:")
    print(by_year[by_year["percent_negative"] >= HIGH_NEGATIVE_STATION_SHARE])


def eea_flags(out) -> None:
    """For the raw EEA files: which Validity flags do the negative values carry? (The pipeline keeps flags 1, 2, 3.)"""
    files = sorted((RAW_DIR / "eea").rglob("*.parquet"))
    if not files:
        print("\n--- EEA flags: raw_data/eea is empty, section skipped ---")
        return
    frames = []
    for path in files:
        raw = pd.read_parquet(path, columns=["Value", "AggType", "Validity", "Verification"])
        frames.append(raw[raw["AggType"] == "hour"])
    raw = pd.concat(frames)
    negative = raw[raw["Value"] < 0]
    table = negative.groupby("Validity")["Value"].agg(rows="size", minimum="min", median="median")
    print("\n--- EEA raw hourly files: negative values by Validity flag ---")
    print(table)
    kept = negative[negative["Validity"].isin([1, 2, 3])]
    print("Negative values with a flag the pipeline keeps (Validity, Verification):")
    print(pd.crosstab(kept["Validity"], kept["Verification"]))
    print("(Validity -1 rows carry sentinel values such as -999 and -9999 and become missing; 1 = valid, "
          "2 = valid below the detection limit with the measured value given.)")
    table.to_csv(out / "eea_negative_values_by_validity_flag.csv")


def compare_policies(canonical: pd.DataFrame, out) -> None:
    """Apply each policy to the pool stations' hourly data and compare valid data and a persistence forecast on dev."""
    # non-held-out stations from 2018 on (2025 is included only so that the rows get their split labels)
    hourly = canonical[~canonical["station_id"].isin(HELD_OUT_STATIONS) & (canonical["timestamp_utc"] >= POOL_START)]
    policies = [("keep", None), ("nan_all", None), ("clip0", None)] + [("two_tier", t) for t in TWO_TIER_LIMITS]
    results = {}
    for name, limit in policies:
        label = name if limit is None else f"{name}(T={limit})"
        changed = hourly.assign(pm25=apply_policy(hourly["pm25"], name, limit))
        results[label] = (changed, features_from_hourly(changed))

    def dev_rows(rows):
        return rows[rows["split"] == "dev"].set_index(["station_id", "block_start_utc"])

    common = None
    for _, rows in results.values():
        index = dev_rows(rows).index
        common = index if common is None else common.intersection(index)
    keep_dev = dev_rows(results["keep"][1])
    table = []
    for label, (changed, rows) in results.items():
        dev = dev_rows(rows)
        on_common = dev.loc[common]
        keep_target = keep_dev.loc[common, "target_pm25"]
        table.append({
            "policy": label,
            "valid_hours": int(changed["pm25"].notna().sum()),
            "train_rows": int((rows["split"] == "train").sum()),
            "dev_rows": len(dev),
            "dev_MAE_own_target": round((dev["pm25_current"] - dev["target_pm25"]).abs().mean(), 3),
            "common_dev_rows": len(common),
            "MAE_vs_keep_target_on_common": round((on_common["pm25_current"] - keep_target).abs().mean(), 3),
            "bias_vs_keep_target": round((on_common["pm25_current"] - keep_target).mean(), 3),
        })
    table = pd.DataFrame(table).set_index("policy")
    print("\n--- Effect of each policy (non-held-out stations; persistence forecast = the current block) ---")
    print(table)
    table.to_csv(out / "policy_effects.csv")

    cork = {label: {"train_rows": int(((rows["station_id"] == "IE004BP") & (rows["split"] == "train")).sum()),
                    "dev_rows": int(((rows["station_id"] == "IE004BP") & (rows["split"] == "dev")).sum()),
                    "valid_hours": int(changed[changed["station_id"] == "IE004BP"]["pm25"].notna().sum())}
            for label, (changed, rows) in results.items()}
    cork = pd.DataFrame(cork).T
    print("\nCork UCC (IE004BP), the station that holds almost all the negatives:")
    print(cork)
    cork.to_csv(out / "policy_effects_cork.csv")
    pool_cork = hourly[(hourly["station_id"] == "IE004BP") & (hourly["timestamp_utc"] < pd.Timestamp("2025-01-01", tz="UTC"))]
    pool_cork = pool_cork["pm25"].dropna()
    print(f"Cork 2022-2024: mean {pool_cork.mean():.3f}, mean if clipped to 0 {pool_cork.clip(lower=0).mean():.3f} "
          f"(upward shift {pool_cork.clip(lower=0).mean() - pool_cork.mean():.3f} ug/m3), median {pool_cork.median():.2f} either way")


def block_level_counts() -> None:
    """How many 3-hour block means are negative (log1p cannot take a value of -1 or less)."""
    blocks = pd.read_parquet(PROCESSED_DIR / "three_hour_aggregated.parquet")
    blocks = blocks[in_decision_pool(blocks, "block_start_utc") & blocks["pm25_3h_mean"].notna()]
    print("\n--- 3-hour block means in the decision pool ---")
    print(f"blocks with a value: {len(blocks):,}; below 0: {int((blocks['pm25_3h_mean'] < 0).sum()):,}; "
          f"at or below -1 (log1p undefined): {int((blocks['pm25_3h_mean'] <= -1).sum()):,}")


def main():
    """Print and save every table about negative PM2.5 readings."""
    canonical = load_canonical()
    out = report_dir("negative_values")
    count_by_source(canonical, out)
    pool_valid = canonical[in_decision_pool(canonical, "timestamp_utc") & canonical["pm25"].notna()]
    magnitude_of_pool_negatives(pool_valid, out)
    negatives_by_station(pool_valid, out)
    eea_flags(out)
    block_level_counts()
    verify_against_pipeline(canonical)
    compare_policies(canonical, out)
    print(f"\nTables saved in {out}")


if __name__ == "__main__":
    main()
