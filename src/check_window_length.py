"""
Window-length check (Milestone 3, E4d): do rolling statistics over 16 or 24 blocks help XGBoost more than over 8 blocks?

USAGE:  python src/check_window_length.py --config configs/window_length.yaml [--no-tracking]
INPUT:  the configuration, and via m3_data.py the train and dev rows and the gap-free block table (the test split and the held-out
        stations are never read)
OUTPUT: one MLflow run per variant and a summary run in experiment `m3_windows`; reports/m3/window_length/window_length.csv

What is done: the stored features hold the rolling mean and standard deviation over 8 blocks. Here they are rebuilt from the
gap-free block table for every window in the configuration (a window is only valid when all its blocks are valid, as in the
pipeline), and the rebuilt 8-block values are checked against the stored ones. A longer window needs more valid blocks, so rows
without a complete window of the LONGEST length are dropped for every variant: all variants are trained and scored on the same rows,
and the only thing that changes is the window. The variants are: the reference (current block, lags 1-3, mean/std over the first
window, plus the time features), the same with the mean/std over each longer window, and optionally the reference plus the longest
window's mean/std as extra columns. The model is the starting XGBoost configuration; the objective is the equal-weight mean, over the
three selection folds, of the MAE on seen-station rows, and the 2024 MAE is reported beside it. A variant is adopted only under the
pre-stated rule: at least `adoption_threshold_pct` better in at least 2 of the 3 folds AND the 2024 MAE not worse. Nothing here changes
the pipeline or a model file.

Requires: pip install pandas pyarrow numpy scikit-learn xgboost mlflow pyyaml
"""
import argparse
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from config import REPORTS_DIR
from feature_extraction import ROLLING_WINDOW_BLOCKS
from m3_data import expanding_folds, load_rows, read_block_grid, rows_hash, time_features
from m3_tracking import lineage_tags, setup_experiment
from train import fit_predict, score_fold

DEFAULTS = {"experiment": "m3_windows", "time_features": "cyclical", "target": "absolute", "windows": [8, 16, 24], "add_longest": True,
            "adoption_threshold_pct": 1.0, "seed": 0,
            "params": {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 6, "objective": "reg:absoluteerror"}}
SELECTION = ("F2021", "F2022", "F2023")
BASE_COLUMNS = ["pm25_current", "pm25_lag_1", "pm25_lag_2", "pm25_lag_3"]


def add_window_statistics(rows: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Add roll_mean_<w> and roll_std_<w> for every window, rebuilt from the gap-free block table (NaN if any block in it is missing)."""
    grid = read_block_grid(rows["station_id"].unique())
    by_station = grid.groupby("station_id")["pm25_3h_mean"]
    for w in windows:
        grid[f"roll_mean_{w}"] = by_station.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        grid[f"roll_std_{w}"] = by_station.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
    keep = ["station_id", "block_start_utc"] + [c for c in grid.columns if c.startswith("roll_")]
    return rows.merge(grid[keep], on=["station_id", "block_start_utc"], how="left", validate="1:1")


def variant_columns(windows: list, add_longest: bool) -> dict:
    """The input columns of every variant, by name: w<first> (the reference), w<other> and, optionally, w<first>+w<longest>."""
    first, longest = windows[0], max(windows)
    variants = {f"w{w}": BASE_COLUMNS + [f"roll_mean_{w}", f"roll_std_{w}"] for w in windows}
    if add_longest and longest != first:
        variants[f"w{first}+w{longest}"] = (BASE_COLUMNS + [f"roll_mean_{first}", f"roll_std_{first}",
                                                          f"roll_mean_{longest}", f"roll_std_{longest}"])
    return variants


def main():
    """Rebuild the windows, score every variant on the same rows, print the table and the decision, log to MLflow."""
    parser = argparse.ArgumentParser(description="Window-length check for XGBoost.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **yaml.safe_load(args.config.read_text(encoding="utf-8"))}
    windows = [int(w) for w in cfg["windows"]]
    if windows[0] != ROLLING_WINDOW_BLOCKS:
        raise SystemExit(f"The first window must be the stored one ({ROLLING_WINDOW_BLOCKS} blocks), not {windows[0]}.")

    t0 = time.perf_counter()
    rows = add_window_statistics(load_rows(), windows)
    first, longest = windows[0], max(windows)
    # the rebuilt reference window must equal the stored features (pandas' rolling std leaves about 1e-6 for identical values)
    if not np.allclose(rows[f"roll_mean_{first}"], rows["pm25_rolling_mean"], atol=1e-9, rtol=0) or \
            not np.allclose(rows[f"roll_std_{first}"], rows["pm25_rolling_std"], atol=1e-5, rtol=0):
        raise SystemExit("WINDOW CHECK FAILED: the rebuilt reference window differs from the stored rolling statistics.")
    complete = rows[f"roll_mean_{longest}"].notna() & rows[f"roll_std_{longest}"].notna()
    by_split = rows.assign(complete=complete).groupby("split")["complete"].agg(rows="size", kept="sum")
    print(f"Rows with a complete {longest}-block window: {int(complete.sum()):,} of {len(rows):,} "
          f"(dropped {int((~complete).sum()):,}, {(~complete).mean():.1%}); every variant uses these rows.\n{by_split.to_string()}")
    rows = rows[complete].reset_index(drop=True)

    model_cfg = {"model": "xgboost", "target": cfg["target"], "scaling": "none", "place": "none", "seed": cfg["seed"], "threads": None,
                 "params": cfg["params"]}
    timef = time_features(rows, cfg["time_features"])
    folds = expanding_folds(rows)
    results, persistence = {}, {}
    for name, columns in variant_columns(windows, cfg["add_longest"]).items():
        X = pd.concat([rows[columns], timef], axis=1)
        results[name] = {}
        for fold in folds:
            forecast, _, _ = fit_predict(model_cfg, rows, X, fold)
            metrics, _ = score_fold(rows, fold, forecast)
            results[name][fold.name] = metrics["mae_seen"]
            persistence[fold.name] = metrics["persistence_mae_seen"]
            results[name][f"n_{fold.name}"] = metrics["n_seen"]
        results[name]["objective"] = float(np.mean([results[name][k] for k in SELECTION]))
        print(f"  {name:8s} " + "  ".join(f"{k} {results[name][k]:.4f}" for k in (*SELECTION, "D")) +
              f"   objective {results[name]['objective']:.4f}", flush=True)

    reference = f"w{first}"
    records = []
    for name, r in results.items():
        change = {k: 100 * (r[k] / results[reference][k] - 1) for k in (*SELECTION, "D")}
        folds_gaining = sum(change[k] <= -cfg["adoption_threshold_pct"] for k in SELECTION)
        adopted = name != reference and folds_gaining >= 2 and change["D"] <= 0
        records.append({"variant": name, **{f"mae_{k}": r[k] for k in (*SELECTION, "D")}, "objective": r["objective"],
                        **{f"change_{k}_pct": change[k] for k in (*SELECTION, "D")},
                        "change_objective_pct": 100 * (r["objective"] / results[reference]["objective"] - 1),
                        "folds_with_gain_ge_threshold": folds_gaining, "D_worse_than_reference": change["D"] > 0, "adopted": adopted})
    table = pd.DataFrame(records)
    print("\nPersistence MAE on the same rows: " + ", ".join(f"{k} {persistence[k]:.4f}" for k in (*SELECTION, "D")))
    print(table.round(4).to_string(index=False))
    adopted = table.loc[table["adopted"], "variant"].tolist()
    print(f"\nAdopted under the rule (>= {cfg['adoption_threshold_pct']:g}% in >= 2 of 3 folds, 2024 not worse): "
          f"{', '.join(adopted) if adopted else 'none; the reference window is kept'}")

    out = REPORTS_DIR / "m3" / "window_length"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "window_length.csv", index=False)
    if args.no_tracking:
        return

    setup_experiment(cfg["experiment"])
    for record in records:
        with mlflow.start_run(run_name=f"window_{record['variant']}"):
            mlflow.set_tags({"kind": "variant", "model": "xgboost", "variant": record["variant"]})
            mlflow.log_metrics({k: float(v) for k, v in record.items() if isinstance(v, (float, int, np.floating, np.integer))
                                and not isinstance(v, bool)})
    with mlflow.start_run(run_name="window_length_summary"):
        mlflow.set_tags({**lineage_tags(), "kind": "summary", "model": "xgboost", "adopted": ",".join(adopted) or "none",
                         "evaluation_rows_sha256": rows_hash(rows[rows["split"] == "dev"])})
        mlflow.log_params({k: str(v) for k, v in cfg.items()})
        mlflow.log_metrics({"rows_kept": float(complete.sum()), "rows_dropped": float((~complete).sum()),
                            "time_total_seconds": time.perf_counter() - t0})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            table.to_csv(tmp / "window_length.csv", index=False)
            (tmp / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            mlflow.log_artifacts(str(tmp))
    print(f"Logged the check in experiment {cfg['experiment']}.")


if __name__ == "__main__":
    main()
