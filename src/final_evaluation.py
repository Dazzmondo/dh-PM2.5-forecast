"""
THE FINAL EVALUATION: score a frozen model once on the 2025 test split and on the held-out cities (Derry, London, Paris).

USAGE:  python src/final_evaluation.py --model-dir models/<model>/v<N> --confirm-freeze
INPUT:  a saved model folder written by train.py or train_gru.py (model file + config.yaml [+ preprocessing.json]), and via
        m3_data.py the `test` and `held_out_test` rows (the ONLY script allowed to read them)
OUTPUT: one MLflow run in experiment `m3_final` (tags, metrics, per-station table) and reports/final/<model>_v<N>_final.csv

This is the one place where the test year and the unseen cities are touched. Rules enforced here:
  - it refuses to start without --confirm-freeze: every choice (features, target form, hyperparameters, rounds or epochs, which
    model) must already be frozen and recorded, because a result seen here may never change a choice;
  - it runs at most ONCE per model folder: if `m3_final` already holds a run for the folder, it stops (to repeat on purpose,
    delete that run in MLflow first, and say so in the report);
  - it only LOADS the frozen model; nothing is fitted, tuned or selected here (no refit on 2024: that belongs to Milestone 4);
  - the baselines (persistence, rolling mean, seasonal naive) are scored on the same rows for context.
Results are reported per station and per city with row counts: the 2025 test rows (stations seen in training) and the held-out rows
(cities never seen in training), with and without Cork UCC. Copenhagen has no valid 2025 hours and Kilmainham few; the row counts show it.

Requires: pip install pandas pyarrow numpy scikit-learn xgboost torch mlflow pyyaml
"""
import argparse
import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from xgboost import XGBRegressor

from config import REPORTS_DIR
from m3_data import Fold, TEST_PERIOD_START, build_features, load_rows, mae_rmse, rows_hash, time_features
from m3_tracking import lineage_tags, pip_freeze_text, setup_experiment
from train import BASELINE_COLUMNS, from_model_target, score_fold
from train_gru import GRUForecaster, build_sequences, to_tensors, predict as predict_gru

EXPERIMENT = "m3_final"
STATIC_COLUMNS = {"cyclical": 4, "onehot_utc": 8, "none": 0}     # how many time-feature columns each variant has


def load_frozen(folder: Path):
    """Load a saved model folder; return (configuration, function rows -> forecast in ug/m3). Nothing is fitted."""
    cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    if (folder / "model.pt").exists():                                     # the GRU
        hp = cfg["hp"]
        stats = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
        model = GRUForecaster(STATIC_COLUMNS[cfg["time_features"]], hp["hidden_size"], hp["num_layers"], hp["dropout"])
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))

        def forecast(rows: pd.DataFrame) -> np.ndarray:
            """GRU forecast for these rows, back on the ug/m3 scale."""
            seq, sta, _ = to_tensors(build_sequences(rows), time_features(rows, cfg["time_features"]).to_numpy(dtype=float), None, stats, "cpu")
            return predict_gru(model, seq, sta) * stats["target_std"] + stats["target_mean"]
        return cfg, forecast
    if cfg.get("place", "none") != "none":
        raise SystemExit("Frozen models with place features are not supported by the final evaluation (the frozen baseline has none).")
    if cfg["model"] == "xgboost":
        model = XGBRegressor()
        model.load_model(folder / "model.json")
    else:
        model = pickle.loads((folder / "model.pkl").read_bytes())

    def forecast(rows: pd.DataFrame) -> np.ndarray:
        """Ridge/XGBoost forecast for these rows, back on the ug/m3 scale."""
        X = build_features(rows, cfg["feature_set"], cfg["time_features"])
        return from_model_target(model.predict(X), rows["pm25_current"].to_numpy(), cfg["target"])
    return cfg, forecast


def score_group(rows: pd.DataFrame, forecast: np.ndarray, group: str, seen: bool) -> tuple:
    """Metrics and the per-station table of one group of rows; `seen` only labels whether the stations were in the training data."""
    n = len(rows)
    fold = Fold(group, np.zeros(n, dtype=bool), np.ones(n, dtype=bool), np.ones(n, dtype=bool), TEST_PERIOD_START)   # every row is scored
    metrics, table = score_fold(rows.reset_index(drop=True), fold, forecast)
    table["group"] = group
    table["seen_in_training"] = seen
    return metrics, table


def main():
    """Check the freeze rules, score the frozen model (and the baselines) on the test and held-out rows, and log it once."""
    parser = argparse.ArgumentParser(description="Final evaluation of a frozen model (run once).")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--confirm-freeze", action="store_true",
                        help="confirm that every choice is frozen and recorded; without it this script does nothing")
    args = parser.parse_args()
    if not args.confirm_freeze:
        raise SystemExit("STOPPED: the final evaluation reads the 2025 test rows and the held-out cities. Run it only after every choice is "
                         "frozen and recorded, once, with --confirm-freeze.")
    folder = args.model_dir if args.model_dir.is_absolute() else Path.cwd() / args.model_dir
    if not (folder / "config.yaml").exists():
        raise SystemExit(f"{folder} is not a saved model folder (no config.yaml).")
    model_id = f"{folder.parent.name}/{folder.name}"

    setup_experiment(EXPERIMENT)
    existing = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string=f"tags.model_dir = '{model_id}'")
    if len(existing):
        raise SystemExit(f"STOPPED: {model_id} was already evaluated (run {existing.iloc[0]['run_id']}). The final evaluation runs once per "
                         f"model; to repeat on purpose, delete that run in MLflow first and say so in the report.")

    cfg, forecast = load_frozen(folder)
    rows = load_rows(("test", "held_out_test"), allow_final=True)
    print(f"Final evaluation of {model_id} ({cfg['model'] if 'model' in cfg else 'gru'}): {rows['split'].value_counts().to_dict()}")
    tables, metrics = [], {}
    for split, seen, label in (("test", True, "test"), ("held_out_test", False, "heldout")):
        part = rows[rows["split"] == split].reset_index(drop=True)
        scored, table = score_group(part, forecast(part), label, seen)
        tables.append(table.assign(model=model_id))
        metrics.update({f"{label}_{k}": v for k, v in {"n": scored["n_val"], "mae": scored["mae_all"], "rmse": scored["rmse_all"],
                        "persistence_mae": scored["persistence_mae_seen"], "skill": scored["skill_seen"],
                        "mae_without_cork": scored["mae_seen_without_cork"], "persistence_mae_without_cork": scored["persistence_mae_seen_without_cork"],
                        "stations_beaten": scored["stations_beaten"], "stations_scored": scored["stations_scored"]}.items()})
        for column in BASELINE_COLUMNS.values():                              # the baselines on the same rows, for context
            mae, rmse = mae_rmse(part["target_pm25"], part[column])
            metrics[f"{label}_baseline_{column}_mae"] = mae
        for city, group in table.groupby("city"):
            metrics[f"{label}_city_{city}_mae"] = float(np.average(group["mae"], weights=group["n"]))
            metrics[f"{label}_city_{city}_n"] = int(group["n"].sum())
        print(f"  {label}: rows {metrics[f'{label}_n']:,}  MAE {metrics[f'{label}_mae']:.3f}  RMSE {metrics[f'{label}_rmse']:.3f}  "
              f"persistence {metrics[f'{label}_persistence_mae']:.3f}  skill {metrics[f'{label}_skill']:+.3f}  "
              f"stations beating persistence {metrics[f'{label}_stations_beaten']}/{metrics[f'{label}_stations_scored']}")
    result = pd.concat(tables, ignore_index=True)
    print(result.round(3).to_string(index=False))

    out = REPORTS_DIR / "final"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{folder.parent.name}_{folder.name}_final.csv"
    result.to_csv(csv, index=False)
    with mlflow.start_run(run_name=f"final_{folder.parent.name}_{folder.name}"):
        mlflow.set_tags({**lineage_tags(), "model_dir": model_id, "kind": "final_evaluation", "frozen_confirmed": "true",
                         "evaluation_rows_sha256": rows_hash(rows)})
        mlflow.log_params({k: str(v) for k, v in cfg.items() if k != "params"})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
        mlflow.log_artifact(str(csv))
        mlflow.log_text(pip_freeze_text(), "pip_freeze.txt")
    print(f"Saved {csv} and logged the run in experiment {EXPERIMENT}.")


if __name__ == "__main__":
    main()
