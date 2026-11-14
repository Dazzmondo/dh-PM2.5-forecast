"""
Error analysis of a saved model: where does it go wrong, by how much, and where does it lose to persistence?

USAGE:  python src/error_analysis.py --model-dir models/<model>/v<N> [--final --confirm-freeze] [--no-tracking]
INPUT:  a saved model folder (train.py / train_gru.py), and via m3_data.py the train and dev rows
        with --final --confirm-freeze: also the 2025 test rows and the held-out cities (see the rule below)
OUTPUT: reports/m3/error_analysis/<model>_<version>_<dev|final>/   CSV tables and error_analysis.png, and one MLflow run in
        experiment `m3_error_analysis` (the same files as artefacts)

Two modes:
  default  the development year (2024) is the analysed set. Used during selection: it may inform choices.
  --final  the 2025 test split and the held-out cities (Derry, London, Paris) are the analysed set. It REFUSES to run unless
           --confirm-freeze is given AND the final evaluation (final_evaluation.py) has already been recorded in MLflow for this
           model: the analysis is reported, never used to change the model (a result seen here may not change a choice).

Every table has the model's MAE next to persistence's MAE on the same rows and the skill score 1 - MAE_model / MAE_persistence, so
"where the model loses to persistence" (skill below 0) is visible in each slice. Bias is the mean signed error (forecast minus
actual): negative means under-prediction. Slices: station, city, country; local month, local hour of the forecast origin (in 3-hour bands) and weekday;
concentration band (band edges are fixed from the TRAINING target, never from the analysed rows); year (training years are shown for
reference and are marked in_sample, because the model was fitted on them); with and without Cork UCC. The worst 1% of absolute errors
are listed with their inputs and compared with all rows (level, spread, recent change, hour, month, station). The figure shows the
main patterns. The causes of what is found are hypotheses and are to be written as such in the report.

Requires: pip install pandas pyarrow numpy matplotlib mlflow
"""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from config import REPORTS_DIR
from final_evaluation import EXPERIMENT as FINAL_EXPERIMENT, load_frozen
from m3_data import CORK_UCC, load_rows
from m3_tracking import setup_experiment

BAND_QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.95)    # concentration band edges, taken from the training target
MIN_ROWS_FOR_LOSS_LIST = 100                      # a slice needs this many rows to be listed as "loses to persistence"


def slice_table(df: pd.DataFrame, by) -> pd.DataFrame:
    """n, MAE, RMSE, bias, persistence MAE and skill for every group of `by` (a column name or list of names)."""
    grouped = df.groupby(by, observed=True)
    table = pd.DataFrame({"n": grouped.size(), "mae": grouped["abs_error"].mean(), "rmse": np.sqrt(grouped["sq_error"].mean()),
                          "bias": grouped["error"].mean(), "persistence_mae": grouped["persistence_abs_error"].mean()})
    table["skill"] = 1 - table["mae"] / table["persistence_mae"]
    table["share_of_rows"] = table["n"] / table["n"].sum()
    return table.reset_index()


def make_bands(train_target: pd.Series) -> tuple:
    """Band edges from the training target and the labels (ordered, numbered so that they sort)."""
    edges = [float(e) for e in np.quantile(train_target, BAND_QUANTILES)]
    names = [f"1: <= {edges[0]:.1f}"] + [f"{i + 2}: {edges[i]:.1f} to {edges[i + 1]:.1f}" for i in range(len(edges) - 1)] + [f"{len(edges) + 1}: > {edges[-1]:.1f}"]
    return edges, names


def build_frame(rows: pd.DataFrame, forecast: np.ndarray, band_edges: list, band_names: list) -> pd.DataFrame:
    """One row per forecast with the error columns and the slicing columns added."""
    df = rows.copy()
    df["forecast"] = forecast
    df["error"] = df["forecast"] - df["target_pm25"]                  # negative = under-prediction
    df["abs_error"] = df["error"].abs()
    df["sq_error"] = df["error"] ** 2
    df["persistence_abs_error"] = (df["pm25_current"] - df["target_pm25"]).abs()
    # the local clock hour at the start of the origin block, in 3-hour bands: blocks start every 3 h UTC and the time zones differ, so single
    # hours would mix different stations and zigzag
    df["local_hour"] = (df["local_timestamp"].dt.hour // 3) * 3
    df["year"] = df["block_start_utc"].dt.year
    df["in_sample"] = df["split"] == "train"
    df["band"] = pd.cut(df["target_pm25"], bins=[-np.inf] + band_edges + [np.inf], labels=band_names)
    df["recent_change"] = df["pm25_current"] - df["pm25_lag_1"]
    return df


def worst_cases(focus: pd.DataFrame) -> tuple:
    """The worst 1% of absolute errors, and how their inputs compare with all analysed rows."""
    cutoff = focus["abs_error"].quantile(0.99)
    worst = focus[focus["abs_error"] >= cutoff].sort_values("abs_error", ascending=False)
    columns = ["station_id", "city", "block_start_utc", "target_pm25", "forecast", "pm25_current", "pm25_lag_1", "pm25_lag_2", "pm25_lag_3",
               "pm25_rolling_mean", "pm25_rolling_std", "error", "abs_error", "persistence_abs_error"]
    features = ["pm25_current", "pm25_rolling_mean", "pm25_rolling_std", "recent_change", "target_pm25", "local_hour", "month"]
    compare = pd.DataFrame({"worst_1pct_median": worst[features].median(), "all_rows_median": focus[features].median(),
                            "worst_1pct_mean": worst[features].mean(), "all_rows_mean": focus[features].mean()})
    stations = pd.DataFrame({"share_of_worst_1pct": worst["station_id"].value_counts(normalize=True),
                             "share_of_all_rows": focus["station_id"].value_counts(normalize=True)}).fillna(0)
    stations["ratio"] = stations["share_of_worst_1pct"] / stations["share_of_all_rows"]
    return worst[columns], compare, stations.sort_values("ratio", ascending=False), cutoff


def losing_slices(tables: dict) -> pd.DataFrame:
    """Every listed slice (n large enough) where the model's MAE is above persistence's."""
    parts = []
    for name, table in tables.items():
        if name in ("by_year", "overall", "by_group") or "skill" not in table:
            continue
        losing = table[(table["skill"] < 0) & (table["n"] >= MIN_ROWS_FOR_LOSS_LIST)].copy()
        if len(losing):
            losing.insert(0, "slice", name)
            losing.insert(1, "group", losing.drop(columns=["n", "mae", "rmse", "bias", "persistence_mae", "skill", "share_of_rows", "slice"],
                                                  errors="ignore").astype(str).agg(" / ".join, axis=1))
            parts.append(losing[["slice", "group", "n", "mae", "persistence_mae", "skill", "bias"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["slice", "group", "n", "mae", "persistence_mae", "skill", "bias"])


def draw_figure(focus: pd.DataFrame, tables: dict, path: Path, title: str) -> None:
    """Six panels: MAE by month, by local hour, bias by concentration band, MAE by station against persistence, residuals, forecast vs actual."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, key, label in ((axes[0, 0], "by_month", "local month"), (axes[0, 1], "by_local_hour", "local hour of the forecast origin (3-hour bands)")):
        axis.plot(tables[key][key.replace("by_", "")], tables[key]["mae"], marker="o", label="model")
        axis.plot(tables[key][key.replace("by_", "")], tables[key]["persistence_mae"], marker="o", linestyle="--", label="persistence")
        axis.set_xlabel(label)
        axis.set_ylabel("MAE (ug/m3)")
        axis.legend()
    band = tables["by_band"]
    axes[0, 2].bar(range(len(band)), band["bias"], color="#d62728")
    axes[0, 2].set_xticks(range(len(band)), [b.split(":")[0] for b in band["band"]])
    axes[0, 2].axhline(0, color="black", linewidth=0.8)
    axes[0, 2].set_xlabel("concentration band (edges from the training target)")
    axes[0, 2].set_ylabel("mean signed error (negative = under-prediction)")
    station = tables["by_station"].sort_values("mae")
    positions = np.arange(len(station))
    axes[1, 0].barh(positions - 0.2, station["mae"], height=0.4, label="model")
    axes[1, 0].barh(positions + 0.2, station["persistence_mae"], height=0.4, label="persistence", alpha=0.6)
    axes[1, 0].set_yticks(positions, station["station_id"])
    axes[1, 0].set_xlabel("MAE (ug/m3)")
    axes[1, 0].legend()
    axes[1, 1].hist(focus["error"].clip(-30, 30), bins=80, color="#1f77b4")
    axes[1, 1].set_xlabel("error: forecast minus actual (clipped at +/-30)")
    axes[1, 1].set_ylabel("rows")
    axes[1, 2].hexbin(focus["target_pm25"].clip(0, 80), focus["forecast"].clip(0, 80), gridsize=50, bins="log", cmap="viridis")
    axes[1, 2].plot([0, 80], [0, 80], color="white", linewidth=0.8)
    axes[1, 2].set_xlabel("actual (ug/m3, clipped at 80)")
    axes[1, 2].set_ylabel("forecast (ug/m3)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    """Load the frozen model, build the error frame, write every table and the figure, and log the run."""
    parser = argparse.ArgumentParser(description="Error analysis of a saved model.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--final", action="store_true", help="analyse the 2025 test split and the held-out cities (after the final evaluation)")
    parser.add_argument("--confirm-freeze", action="store_true")
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    folder = args.model_dir if args.model_dir.is_absolute() else Path.cwd() / args.model_dir
    model_id = f"{folder.parent.name}/{folder.name}"
    if args.final:
        if not args.confirm_freeze:
            raise SystemExit("STOPPED: --final reads the 2025 test rows and the held-out cities; it needs --confirm-freeze.")
        setup_experiment(FINAL_EXPERIMENT)
        if not len(mlflow.search_runs(experiment_names=[FINAL_EXPERIMENT], filter_string=f"tags.model_dir = '{model_id}'")):
            raise SystemExit(f"STOPPED: the final evaluation of {model_id} has not been recorded. Run final_evaluation.py first; the error "
                             f"analysis of the test rows is reported after it and never used to change the model.")

    _, forecast = load_frozen(folder)
    reference = load_rows(("train", "dev"))
    parts = [reference] + ([load_rows(("test", "held_out_test"), allow_final=True)] if args.final else [])
    rows = pd.concat(parts, ignore_index=True)
    edges, band_names = make_bands(reference.loc[reference["split"] == "train", "target_pm25"])
    df = build_frame(rows, forecast(rows), edges, band_names)
    focus_splits = ["test", "held_out_test"] if args.final else ["dev"]
    focus = df[df["split"].isin(focus_splits)].copy()
    label = "final" if args.final else "dev"
    print(f"Error analysis of {model_id} on {label}: {len(focus):,} rows ({', '.join(focus_splits)}); band edges from the training target: "
          f"{[round(e, 1) for e in edges]}")

    tables = {"overall": pd.concat([slice_table(focus.assign(all_rows="all rows"), "all_rows").rename(columns={"all_rows": "group"}),
                                    slice_table(focus[focus["station_id"] != CORK_UCC].assign(all_rows="without Cork UCC"), "all_rows").rename(columns={"all_rows": "group"})]),
              "by_station": slice_table(focus, ["station_id", "city", "country"]), "by_city": slice_table(focus, "city"),
              "by_country": slice_table(focus, "country"), "by_month": slice_table(focus, "month"),
              "by_local_hour": slice_table(focus, "local_hour"), "by_weekday": slice_table(focus, "day_of_week"),
              "by_band": slice_table(focus, "band"), "by_year": slice_table(df, ["year", "in_sample"])}
    if args.final:
        tables["by_group"] = slice_table(focus, "split")
    worst, compare, worst_stations, cutoff = worst_cases(focus)
    cork = focus.groupby("station_id").agg(n=("target_pm25", "size"), negative_target_share=("target_pm25", lambda s: float((s < 0).mean())))
    percentiles = focus["error"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).rename("error").to_frame()
    losing = losing_slices(tables)

    print("\nOverall:"); print(tables["overall"].round(3).to_string(index=False))
    print("\nBy station (model vs persistence):"); print(tables["by_station"].round(3).to_string(index=False))
    print("\nBy concentration band (bias shows under-prediction of peaks when negative):"); print(tables["by_band"].round(3).to_string(index=False))
    print("\nBy year (in_sample years are training years, shown for reference only):"); print(tables["by_year"].round(3).to_string(index=False))
    print(f"\nSlices where the model loses to persistence (n >= {MIN_ROWS_FOR_LOSS_LIST}): {len(losing)}"); print(losing.round(3).to_string(index=False))
    print(f"\nWorst 1% of absolute errors (cutoff {cutoff:.1f} ug/m3), the ten largest:"); print(worst.head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nWorst 1% against all rows (medians/means):"); print(compare.round(2).to_string())
    print("\nStations over-represented in the worst 1% (ratio of shares):"); print(worst_stations.head(5).round(3).to_string())

    out = REPORTS_DIR / "m3" / "error_analysis" / f"{folder.parent.name}_{folder.name}_{label}"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    worst.to_csv(out / "worst_1pct_cases.csv", index=False)
    compare.to_csv(out / "worst_1pct_vs_all_rows.csv")
    worst_stations.to_csv(out / "worst_1pct_station_shares.csv")
    cork.to_csv(out / "negative_target_share_by_station.csv")
    percentiles.to_csv(out / "residual_percentiles.csv")
    losing.to_csv(out / "slices_where_model_loses_to_persistence.csv", index=False)
    draw_figure(focus, tables, out / "error_analysis.png", f"{model_id} on {label}: {len(focus):,} rows")
    print(f"\nTables and figure saved in {out}")
    if args.no_tracking:
        return

    setup_experiment("m3_error_analysis")
    overall = tables["overall"].iloc[0]
    with mlflow.start_run(run_name=f"error_analysis_{folder.parent.name}_{folder.name}_{label}"):
        mlflow.set_tags({"model_dir": model_id, "focus": label, "kind": "error_analysis"})
        mlflow.log_metrics({"n": float(overall["n"]), "mae": float(overall["mae"]), "rmse": float(overall["rmse"]), "bias": float(overall["bias"]),
                            "persistence_mae": float(overall["persistence_mae"]), "skill": float(overall["skill"]),
                            "worst_1pct_cutoff": float(cutoff), "slices_losing_to_persistence": float(len(losing))})
        mlflow.log_artifacts(str(out))
    print("Logged the run in experiment m3_error_analysis.")


if __name__ == "__main__":
    main()
