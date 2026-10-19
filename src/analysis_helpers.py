"""
Shared pieces for the check_*.py and baseline_persistence.py scripts. Not a pipeline step: it is only imported.

Two ideas live here so that every analysis script uses them in the same way:

1. The DECISION POOL. Every threshold, rule or comparison that leads to a choice may use only the non-held-out stations
   and only the years 2018-2024 (train and dev). 2025 and the held-out stations may be counted, never used to decide.
2. features_from_hourly(): rebuild the feature table from an hourly series using the pipeline's OWN functions
   (aggregate_station, ensure_full_grid, add_lag_rolling_and_target, assign_splits), so a "what if" run (another
   negative-value policy, another valid-hours rule) differs from the real pipeline only in the one thing being tested.
   verify_against_pipeline() proves it reproduces the pipeline's feature table exactly.

The eda_*.py scripts also use the small plotting helpers at the end of this file (one look for every figure: the validated
default categorical colours in fixed order, a light surface, thin marks, a light grid).

Requires: pip install pandas pyarrow numpy matplotlib
"""
import pandas as pd

import aggregate_to_3hour
from aggregate_to_3hour import aggregate_station
from config import PROCESSED_DIR, REPORTS_DIR
from create_splits import HELD_OUT_STATIONS, STUDY_START_YEAR, TEST_YEAR, assign_splits
from feature_extraction import add_lag_rolling_and_target, ensure_full_grid
from stations import load_station_reference

CANONICAL_PATH = PROCESSED_DIR / "canonical_hourly.parquet"
FEATURES_PATH = PROCESSED_DIR / "features" / "pm25_features_with_splits.parquet"

# The decision pool: hours from the start of the study to the end of the development year (2024).
POOL_START = pd.Timestamp(f"{STUDY_START_YEAR}-01-01", tz="UTC")
POOL_END = pd.Timestamp(f"{TEST_YEAR}-01-01", tz="UTC")   # exclusive: 2025 is the test year

pd.set_option("display.width", 200)         # print wide tables on one line instead of cutting columns
pd.set_option("display.max_columns", 30)

STATIC_COLUMNS = ["city", "country", "timezone", "latitude", "longitude"]
# a feature row needs ALL of these (the same list feature_extraction.py drops on)
REQUIRED_COLUMNS = ["pm25_current", "pm25_lag_1", "pm25_lag_2", "pm25_lag_3",
                    "pm25_rolling_mean", "pm25_rolling_std", "target_pm25"]


def report_dir(topic: str):
    """The folder reports/<topic>/ (created if needed): where a script writes its CSV tables."""
    path = REPORTS_DIR / topic
    path.mkdir(parents=True, exist_ok=True)
    return path


def in_decision_pool(df: pd.DataFrame, time_column: str) -> pd.Series:
    """True for rows of non-held-out stations whose time falls in 2018-2024 (the only rows a decision may use)."""
    return (~df["station_id"].isin(HELD_OUT_STATIONS)) & (df[time_column] >= POOL_START) & (df[time_column] < POOL_END)


def load_canonical() -> pd.DataFrame:
    """Read the canonical hourly table, or stop with a clear message if canonicalize_raw_data.py has not been run."""
    if not CANONICAL_PATH.exists():
        raise SystemExit(f"{CANONICAL_PATH} not found. Run canonicalize_raw_data.py first.")
    return pd.read_parquet(CANONICAL_PATH)


def features_from_hourly(hourly: pd.DataFrame, min_valid_hours: int = aggregate_to_3hour.MIN_VALID_HOURS) -> pd.DataFrame:
    """Build the labelled feature rows (the rows a model could use) from an hourly table.

    `hourly` needs station_id, timestamp_utc, pm25 and the station columns city, country, timezone, latitude, longitude.
    It goes through the pipeline's own aggregation, grid, lag/rolling/target and split functions. `min_valid_hours` replaces
    the 2-of-3 rule for this call only. Returns the rows with every required value present, with a `split` column.
    """
    original_rule = aggregate_to_3hour.MIN_VALID_HOURS
    aggregate_to_3hour.MIN_VALID_HOURS = min_valid_hours   # aggregate_station() reads this module-level constant
    try:
        parts = []
        for station_id, group in hourly.groupby("station_id"):
            blocks = aggregate_station(group)
            blocks["station_id"] = station_id
            for column in STATIC_COLUMNS:
                blocks[column] = group[column].iloc[0]
            parts.append(blocks)
    finally:
        aggregate_to_3hour.MIN_VALID_HOURS = original_rule   # always put the real rule back
    df = add_lag_rolling_and_target(ensure_full_grid(pd.concat(parts, ignore_index=True)))
    return label_splits(df.dropna(subset=REQUIRED_COLUMNS))


def label_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Add the `split` column to feature rows exactly as create_splits.py does (same function, same train-start rule)."""
    stations = load_station_reference().set_index("station_id")
    train_start = df["station_id"].map(stations["availability_start"].dt.year).clip(lower=STUDY_START_YEAR)
    df = df.copy()
    df["split"] = assign_splits(df, train_start)
    return df


def verify_against_pipeline(canonical: pd.DataFrame) -> None:
    """Stop unless features_from_hourly(canonical) gives the same rows and split counts as the pipeline's own feature file."""
    if not FEATURES_PATH.exists():
        raise SystemExit(f"{FEATURES_PATH} not found. Run the pipeline (create_splits.py) first.")
    rebuilt = features_from_hourly(canonical)
    actual = pd.read_parquet(FEATURES_PATH, columns=["station_id", "block_start_utc", "split"])
    same_rows = len(rebuilt) == len(actual)
    same_splits = rebuilt["split"].value_counts().to_dict() == actual["split"].value_counts().to_dict()
    if not (same_rows and same_splits):
        raise SystemExit("The rebuilt feature table does not match the pipeline's: "
                         f"{len(rebuilt)} against {len(actual)} rows. Re-run the pipeline before trusting these checks.")
    print(f"Check: the rebuilt feature table matches the pipeline's ({len(actual):,} rows, same split counts) - OK.")


# ----------------------------------------------------------------------------------------------------------------
# Plotting helpers for the eda_*.py scripts

FIGURES_DIR = REPORTS_DIR / "figures"
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
# categorical colours, assigned in this fixed order (the first three pass the colour-vision checks for any pair)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SOURCE_COLORS = {"eea": BLUE, "epa": ORANGE, "uk_air": AQUA}
SOURCE_LABELS = {"eea": "EEA", "epa": "EPA Ireland", "uk_air": "UK-AIR"}
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # light = low, dark = high


def get_pyplot():
    """Import matplotlib without a screen (figures are only saved to files) and apply the project's figure style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 100,
    })
    return plt


def save_figure(fig, name: str):
    """Save a figure as reports/figures/<name> (PNG, 150 dpi) and print where it went."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {path}")
    return path


def print_conclusion(finding: str, decision: str, limitation: str) -> None:
    """Print the three lines every EDA script ends with: what the data show, what we do about it, and what that cannot tell."""
    print("\n=== Conclusion ===")
    print(f"FINDING:    {finding}")
    print(f"DECISION:   {decision}")
    print(f"LIMITATION: {limitation}")


def load_features() -> pd.DataFrame:
    """Read the labelled feature table (stops with a clear message if the pipeline has not been run)."""
    if not FEATURES_PATH.exists():
        raise SystemExit(f"{FEATURES_PATH} not found. Run the pipeline (create_splits.py) first.")
    return pd.read_parquet(FEATURES_PATH)
