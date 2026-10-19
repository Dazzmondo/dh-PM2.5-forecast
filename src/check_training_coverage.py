"""
Check which cities and countries the training stations cover, and what that means for how a model can be validated.

INPUT:  metadata/station_reference.csv
        processed_data/features/pm25_features_with_splits.parquet   (create_splits.py)
OUTPUT: reports/training_coverage/*.csv   and the same tables printed to the terminal. Nothing is changed.

Why: the held-out units are whole cities, and stations in one city are strongly correlated. A validation that keeps a city's
stations together (grouping by city) therefore needs enough distinct cities. This script counts them, shows how many
training rows each city has, and whether the countries of the held-out cities also appear in training.

Requires: pip install pandas pyarrow
"""
import pandas as pd

from analysis_helpers import FEATURES_PATH, report_dir
from create_splits import HELD_OUT_STATIONS
from stations import load_station_reference


def main():
    """Print and save the city and country coverage of the training stations."""
    stations = load_station_reference()
    training = stations[~stations["station_id"].isin(HELD_OUT_STATIONS)]
    held_out = stations[stations["station_id"].isin(HELD_OUT_STATIONS)]
    out = report_dir("training_coverage")

    print(f"Training stations: {len(training)}; cities: {training['city'].nunique()}; countries: {training['country'].nunique()}")
    by_country = training.groupby("country").agg(stations=("station_id", "size"), cities=("city", lambda s: sorted(set(s))))
    print(by_country)
    print(f"\nHeld-out stations: {len(held_out)} in the cities {sorted(held_out['city'].unique())}")
    for country in sorted(held_out["country"].unique()):
        names = sorted(training[training["country"] == country]["station_name"])
        print(f"  {country}: " + (f"also in training ({', '.join(names)})" if names else "NOT in training"))
    print(f"\nMost folds a leave-one-city-out validation can have: {training['city'].nunique()}")

    if not FEATURES_PATH.exists():
        raise SystemExit(f"{FEATURES_PATH} not found. Run the pipeline (create_splits.py) first.")
    rows = pd.read_parquet(FEATURES_PATH, columns=["station_id", "city", "country", "split"])
    usable = rows[rows["split"].isin(["train", "dev"])]
    per_city = usable.groupby(["country", "city"]).agg(stations=("station_id", "nunique"), rows=("split", "size"),
                                                       train_rows=("split", lambda s: int((s == "train").sum())))
    per_city["percent_of_rows"] = (100 * per_city["rows"] / per_city["rows"].sum()).round(1)
    print("\nTraining and development rows per city (the rows a model can use):")
    print(per_city)
    by_country.to_csv(out / "training_countries.csv")
    per_city.to_csv(out / "training_rows_per_city.csv")
    print(f"\nTables saved in {out}")


if __name__ == "__main__":
    main()
