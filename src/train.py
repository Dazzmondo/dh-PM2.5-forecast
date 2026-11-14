"""
Run ONE Milestone 3 experiment from a configuration file, score it, and log everything to MLflow.

USAGE:  python src/train.py --config configs/<experiment>.yaml [--save-model] [--no-tracking]
INPUT:  the configuration (see configs/), and via m3_data.py: processed_data/features/pm25_features_with_splits.parquet (train and
        dev rows only) and processed_data/three_hour_aggregated.parquet (blocks before 2025)
OUTPUT: one MLflow run in mlruns/ (parameters, metrics, timings, per-station tables, lineage tags, the configuration, pip freeze);
        with --save-model, also models/<model>/v<N>/ (model file, config.yaml, metrics.json, metadata.json)

Configuration keys (all but `experiment`, `run_name` and `model` are optional):
  experiment      MLflow experiment name, e.g. m3_baselines
  run_name        name of this run
  model           persistence | rolling_mean | seasonal_naive | ridge | xgboost
  settings        any of F (three expanding selection folds), D (train 2018-2023, score 2024), L (leave-one-city-out); default [F, D]
  feature_set     A | A+S | B                       (default A)
  time_features   cyclical | onehot_utc | none      (default cyclical)
  target          absolute | change | log1p         (default absolute; scores are always on the raw ug/m3 scale)
  scaling         none | standard                   (the scaler is fitted inside each fold's training rows; for XGBoost it is only a
                                                     control, as trees ignore monotone rescaling of a feature)
  place           none | latlon | country | all     (default none; used with setting L)
  seed            integer (default 0)
  params          model hyperparameters (Ridge: alpha; XGBoost: any XGBRegressor argument, plus early_stopping_rounds)
  threads         number of CPU threads for XGBoost (default: all)

What is scored, and how (the rules fixed before any result was seen):
  - F: for each validation year 2021, 2022, 2023, fit on the train-split rows whose TARGET ends before that year, score the
    validation year. The decision metric is the MAE on SEEN-station rows (stations that have training rows in the fold); rows
    of unseen stations are reported separately. F_mean_mae_seen is the equal-weight mean over the three folds.
  - D: fit on the whole train split (2018-2023), score the 31,192 development rows (2024). The persistence MAE on these rows is
    recomputed on every D run and must equal the Milestone 2 benchmark (3.869) or the run stops.
  - L: for each development city, fit on the other cities (targets up to the end of 2022), score the left-out city in 2023.
  - The test split and the held-out stations are never read here (m3_data.load_rows refuses them).
  - Early stopping (XGBoost, early_stopping_rounds) uses an inner window: the last 12 months of the fold's training rows by
    target time; the stopping round is then reused when refitting on all of the fold's training rows.

Requires: pip install pandas pyarrow numpy scikit-learn xgboost mlflow pyyaml
"""
import argparse
import json
import pickle
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import REPO_ROOT
from m3_data import (CORK_UCC, EXPECTED_PERSISTENCE_DEV_MAE, build_features, expanding_folds, inner_early_stopping_split,
                     load_rows, loco_folds, mae_rmse, place_features, rows_hash)
from fit_scalers import SCALER_PATH
from m3_tracking import lineage_tags, peak_ram_mb, pip_freeze_text, setup_experiment, sha256_of_file

BASELINE_COLUMNS = {"persistence": "pm25_current",          # the current block is the forecast
                    "rolling_mean": "pm25_rolling_mean",    # the existing 8-block mean is the forecast
                    "seasonal_naive": "pm25_lag_6"}         # the same clock time as the target, one day earlier (block t-6)
MODELS = list(BASELINE_COLUMNS) + ["ridge", "xgboost"]
DEFAULTS = {"settings": ["F", "D"], "feature_set": "A", "time_features": "cyclical", "target": "absolute", "scaling": "none",
            "place": "none", "seed": 0, "params": {}, "threads": None}
XGBOOST_DEFAULTS = {"tree_method": "hist", "n_estimators": 500, "learning_rate": 0.1, "max_depth": 6, "objective": "reg:squarederror"}
MODELS_DIR = REPO_ROOT / "models"


def check_scaler_consistency(model, columns: list, fit_rows: int) -> dict:
    """Prove that a Ridge model's in-model scaler equals the stored scalers.json for the columns they share; return lineage tags.

    The scalers are fitted INSIDE the model (so cross-validation can fit them on each fold's training rows only). For the final fit,
    on the whole training split, they must equal the stored artefact (fit_scalers.py, `standard` variant: concentration and spread
    groups, population standard deviation, the same 123,364 rows). The run stops if the row count or any mean/std differs.
    """
    artefact = json.loads(SCALER_PATH.read_text(encoding="utf-8"))
    if artefact["fitted_on"]["rows"] != fit_rows:
        raise SystemExit(f"SCALER CHECK FAILED: scalers.json was fitted on {artefact['fitted_on']['rows']} rows, the model on {fit_rows}.")
    stored = {}
    for group in ("concentration", "spread"):
        stored.update(artefact["variants"]["standard"][group])
    scaler = model.named_steps["standardscaler"]
    checked, worst = [], 0.0
    for index, column in enumerate(columns):
        if column not in stored:                           # the sin/cos columns have no stored statistics
            continue
        for fitted, saved, what in ((scaler.mean_[index], stored[column]["mean"], "mean"), (scaler.scale_[index], stored[column]["std"], "std")):
            if not np.isclose(fitted, saved, rtol=1e-9, atol=1e-12):
                raise SystemExit(f"SCALER CHECK FAILED: {column} {what} is {fitted!r} in the model but {saved!r} in scalers.json.")
            worst = max(worst, abs(fitted - saved))
        checked.append(column)
    print(f"Scaler check passed: the model's fitted mean and std equal scalers.json for {len(checked)} columns (largest difference {worst:.1e}).")
    return {"scaler_check": "passed", "scaler_check_columns": ",".join(checked), "scaler_check_max_abs_diff": f"{worst:.3e}",
            "scaler_artefact_sha256": sha256_of_file(SCALER_PATH), "scaler_artefact_features_file_sha256": artefact["fitted_on"]["features_file_sha256"]}


def load_config(path: Path) -> dict:
    """Read the YAML configuration, fill in the defaults and stop on a missing or unknown value."""
    cfg = {**DEFAULTS, **yaml.safe_load(Path(path).read_text(encoding="utf-8"))}
    for key in ("experiment", "run_name", "model"):
        if key not in cfg:
            raise SystemExit(f"The configuration needs a `{key}` entry.")
    if cfg["model"] not in MODELS:
        raise SystemExit(f"Unknown model {cfg['model']!r} (use one of {MODELS}).")
    if not set(cfg["settings"]) <= {"F", "D", "L"}:
        raise SystemExit(f"Unknown setting in {cfg['settings']} (use F, D, L).")
    if cfg["target"] not in ("absolute", "change", "log1p"):
        raise SystemExit(f"Unknown target {cfg['target']!r} (use absolute, change or log1p).")
    if cfg["scaling"] not in ("none", "standard"):
        raise SystemExit(f"Unknown scaling {cfg['scaling']!r} (use none or standard).")
    return cfg


def to_model_target(target, current, form: str):
    """The quantity the model is trained to predict: the target, its change from the current block, or log1p (negatives clipped to 0)."""
    if form == "absolute":
        return np.asarray(target, dtype=float)
    if form == "change":
        return np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    return np.log1p(np.clip(np.asarray(target, dtype=float), 0, None))


def from_model_target(predicted, current, form: str):
    """Turn a model's output back into a forecast in ug/m3 (every score is computed on this raw scale)."""
    if form == "absolute":
        return np.asarray(predicted, dtype=float)
    if form == "change":
        return np.asarray(current, dtype=float) + np.asarray(predicted, dtype=float)
    return np.expm1(np.asarray(predicted, dtype=float))


def build_estimator(cfg: dict, n_estimators=None, early_stopping=None):
    """A Ridge pipeline (optionally with an in-fold StandardScaler) or an XGBoost regressor, from the configuration."""
    if cfg["model"] == "ridge":
        steps = [StandardScaler()] if cfg["scaling"] == "standard" else []
        return make_pipeline(*steps, Ridge(alpha=cfg["params"].get("alpha", 1.0)))
    params = {**XGBOOST_DEFAULTS, **cfg["params"]}
    params.pop("early_stopping_rounds", None)               # handled in fit_predict (inner window)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if early_stopping is not None:
        params.update(early_stopping_rounds=early_stopping, eval_metric="mae")
    return XGBRegressor(random_state=cfg["seed"], n_jobs=cfg["threads"], **params)


def fit_predict(cfg: dict, rows: pd.DataFrame, X: pd.DataFrame, fold):
    """Fit on the fold's training rows and forecast its validation rows. Returns (forecast in ug/m3, info dict, fitted model)."""
    val = rows[fold.val]
    if cfg["model"] in BASELINE_COLUMNS:                    # nothing is fitted
        return val[BASELINE_COLUMNS[cfg["model"]]].to_numpy(), {"fit_seconds": 0.0, "predict_seconds": 0.0}, None

    X = pd.concat([X, place_features(rows, fold.fit, cfg["place"])], axis=1)
    if cfg["model"] == "xgboost" and cfg["scaling"] == "standard":
        # the E5 control: standardise with the fold's training rows only, then train as usual (results should match the unscaled run)
        scaler = StandardScaler().fit(X[fold.fit])
        X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    y = to_model_target(rows["target_pm25"], rows["pm25_current"], cfg["target"])     # whole column; masked below
    info, rounds = {}, None
    t0 = time.perf_counter()
    stopping = cfg["params"].get("early_stopping_rounds") if cfg["model"] == "xgboost" else None
    if stopping:
        core, inner = inner_early_stopping_split(rows, fold.fit)
        probe = build_estimator(cfg, early_stopping=stopping)
        probe.fit(X[core], y[core], eval_set=[(X[core], y[core]), (X[inner], y[inner])], verbose=False)
        rounds = int(probe.best_iteration) + 1
        curve = probe.evals_result()
        info["rounds"] = rounds
        info["curve"] = pd.DataFrame({"round": np.arange(1, len(curve["validation_0"]["mae"]) + 1),
                                      "train_mae": curve["validation_0"]["mae"], "inner_validation_mae": curve["validation_1"]["mae"]})
    model = build_estimator(cfg, n_estimators=rounds)
    model.fit(X[fold.fit], y[fold.fit])
    info["fit_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    raw = model.predict(X[fold.val])
    info["predict_seconds"] = time.perf_counter() - t0
    return from_model_target(raw, val["pm25_current"].to_numpy(), cfg["target"]), info, model


def score_fold(rows: pd.DataFrame, fold, forecast: np.ndarray) -> tuple:
    """Metrics of one fold (overall, seen-station rows, unseen rows, without Cork UCC) and the per-station table."""
    val = rows[fold.val].copy()
    val["forecast"] = forecast
    val["seen"] = fold.seen[fold.val]
    val["persistence"] = val["pm25_current"]
    seen = val[val["seen"]]
    unseen = val[~val["seen"]]
    mae_all, rmse_all = mae_rmse(val["target_pm25"], val["forecast"])
    mae_seen, rmse_seen = mae_rmse(seen["target_pm25"], seen["forecast"])
    persistence_seen = mae_rmse(seen["target_pm25"], seen["persistence"])[0]
    no_cork = seen[seen["station_id"] != CORK_UCC]
    out = {"n_val": len(val), "mae_all": mae_all, "rmse_all": rmse_all, "n_seen": len(seen), "mae_seen": mae_seen,
           "rmse_seen": rmse_seen, "persistence_mae_seen": persistence_seen, "skill_seen": 1 - mae_seen / persistence_seen,
           "n_unseen": len(unseen), "mae_unseen": mae_rmse(unseen["target_pm25"], unseen["forecast"])[0] if len(unseen) else np.nan,
           "mae_seen_without_cork": mae_rmse(no_cork["target_pm25"], no_cork["forecast"])[0] if len(no_cork) else np.nan,
           "persistence_mae_seen_without_cork": mae_rmse(no_cork["target_pm25"], no_cork["persistence"])[0] if len(no_cork) else np.nan}
    table = []
    for station, group in val.groupby("station_id"):
        mae, rmse = mae_rmse(group["target_pm25"], group["forecast"])
        persistence = mae_rmse(group["target_pm25"], group["persistence"])[0]
        table.append({"fold": fold.name, "station_id": station, "city": group["city"].iloc[0], "seen": bool(group["seen"].iloc[0]),
                      "n": len(group), "mae": mae, "rmse": rmse, "persistence_mae": persistence, "beats_persistence": mae < persistence})
    out["stations_beaten"] = int(sum(r["beats_persistence"] for r in table if r["seen"]))
    out["stations_scored"] = int(sum(r["seen"] for r in table))
    return out, pd.DataFrame(table)


def run_setting(cfg: dict, rows: pd.DataFrame, X: pd.DataFrame, setting: str) -> dict:
    """Run every fold of one setting (F, D or L); return the fold metrics, tables, timings and the setting's summary numbers."""
    folds = {"F": lambda: [f for f in expanding_folds(rows) if f.name != "D"],
             "D": lambda: [f for f in expanding_folds(rows) if f.name == "D"],
             "L": lambda: loco_folds(rows)}[setting]()
    result = {"metrics": {}, "tables": [], "info": {}, "model": None, "forecasts": {}}
    for fold in folds:
        forecast, info, model = fit_predict(cfg, rows, X, fold)
        metrics, table = score_fold(rows, fold, forecast)
        result["metrics"][fold.name] = metrics
        result["tables"].append(table)
        result["info"][fold.name] = info
        result["model"] = model
        n = metrics["n_seen"]
        print(f"  {setting}/{fold.name:10s} rows {metrics['n_val']:6,} (seen {n:6,})  MAE seen {metrics['mae_seen']:.3f}  "
              f"RMSE {metrics['rmse_seen']:.3f}  persistence {metrics['persistence_mae_seen']:.3f}  skill {metrics['skill_seen']:+.3f}  "
              f"fit {info['fit_seconds']:.1f}s" + (f"  rounds {info['rounds']}" if "rounds" in info else ""))
    return result


def summarise(setting: str, result: dict) -> dict:
    """The headline numbers of a setting: equal-weight mean over folds (F), the development scores (D), pooled and per-city means (L)."""
    m = pd.DataFrame(result["metrics"]).T
    if setting == "F":
        return {"F_mean_mae_seen": m["mae_seen"].mean(), "F_mean_rmse_seen": m["rmse_seen"].mean(),
                "F_mean_skill_seen": m["skill_seen"].mean(), "F_mean_persistence_mae_seen": m["persistence_mae_seen"].mean()}
    if setting == "D":
        d = m.loc["D"]
        return {"D_mae": d["mae_seen"], "D_rmse": d["rmse_seen"], "D_skill": d["skill_seen"], "D_persistence_mae": d["persistence_mae_seen"],
                "D_mae_without_cork": d["mae_seen_without_cork"], "D_persistence_mae_without_cork": d["persistence_mae_seen_without_cork"],
                "D_stations_beaten": d["stations_beaten"], "D_stations_scored": d["stations_scored"], "D_n": d["n_val"]}
    pooled = (m["mae_seen"] * m["n_seen"]).sum() / m["n_seen"].sum()
    return {"L_pooled_mae": pooled, "L_mean_city_mae": m["mae_seen"].mean(),
            "L_pooled_persistence_mae": (m["persistence_mae_seen"] * m["n_seen"]).sum() / m["n_seen"].sum()}


def check_persistence_benchmark(rows: pd.DataFrame) -> None:
    """Stop if the persistence MAE on the 31,192 development rows is not the Milestone 2 figure (3.869): the row set would be wrong."""
    dev = rows[rows["split"] == "dev"]
    mae = mae_rmse(dev["target_pm25"], dev["pm25_current"])[0]
    if len(dev) != 31_192 or round(mae, 3) != EXPECTED_PERSISTENCE_DEV_MAE:
        raise SystemExit(f"BENCHMARK CHECK FAILED: persistence MAE {mae:.4f} on {len(dev)} rows, expected 3.869 on 31,192.")


def next_version(model_name: str) -> int:
    """The next free version number under models/<model>/."""
    existing = [int(p.name[1:]) for p in (MODELS_DIR / model_name).glob("v*") if p.name[1:].isdigit()] if (MODELS_DIR / model_name).exists() else []
    return max(existing, default=0) + 1


def save_model_files(cfg: dict, model, metrics: dict, tags: dict, run_id: str) -> tuple:
    """Write models/<model>/v<N>/ (model file, config.yaml, metrics.json, metadata.json); return (folder, model size in bytes)."""
    folder = MODELS_DIR / cfg["model"] / f"v{next_version(cfg['model'])}"
    folder.mkdir(parents=True)
    if cfg["model"] == "xgboost":
        model.save_model(folder / "model.json")
        model_file = folder / "model.json"
    else:
        model_file = folder / "model.pkl"
        model_file.write_bytes(pickle.dumps(model))
    (folder / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    metadata = {"mlflow_run_id": run_id, "run_name": cfg["run_name"], "experiment": cfg["experiment"], **tags}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return folder, model_file.stat().st_size


def main():
    """Read the configuration, run its settings, print the scores and log the run to MLflow."""
    parser = argparse.ArgumentParser(description="Run one Milestone 3 experiment from a configuration file.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--save-model", action="store_true", help="also write models/<model>/v<N>/ (needs setting D)")
    parser.add_argument("--no-tracking", action="store_true", help="print the scores without writing an MLflow run")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.save_model and "D" not in cfg["settings"]:
        raise SystemExit("--save-model needs setting D (the model fitted on the whole training split).")
    if args.save_model and cfg["model"] == "xgboost" and cfg["scaling"] == "standard":
        raise SystemExit("--save-model is not available for the standardised XGBoost control (the scaler would not be saved with the model).")

    t_start = time.perf_counter()
    rows = load_rows()                                      # train and dev only; windows rebuilt and verified
    X = build_features(rows, cfg["feature_set"], cfg["time_features"]) if cfg["model"] not in BASELINE_COLUMNS else None
    load_seconds = time.perf_counter() - t_start
    print(f"{cfg['experiment']} / {cfg['run_name']}: model {cfg['model']}, feature set {cfg['feature_set']}, time {cfg['time_features']}, "
          f"target {cfg['target']}, scaling {cfg['scaling']}, place {cfg['place']}, seed {cfg['seed']}  (data load {load_seconds:.1f}s)")

    results, metrics, tables, scaler_tags = {}, {}, [], {}
    for setting in cfg["settings"]:
        results[setting] = run_setting(cfg, rows, X, setting)
        metrics.update(summarise(setting, results[setting]))
        tables.extend(results[setting]["tables"])
        if setting == "D":
            check_persistence_benchmark(rows)
            if cfg["model"] == "ridge" and cfg["scaling"] == "standard":
                scaler_tags = check_scaler_consistency(results["D"]["model"], list(X.columns), int((rows["split"] == "train").sum()))
    fit_seconds = sum(i["fit_seconds"] for r in results.values() for i in r["info"].values())
    predict_seconds = sum(i["predict_seconds"] for r in results.values() for i in r["info"].values())
    n_predicted = sum(m["n_val"] for r in results.values() for m in r["metrics"].values())
    metrics.update({"time_load_seconds": load_seconds, "time_fit_seconds": fit_seconds, "time_predict_seconds": predict_seconds,
                    "inference_ms_per_1000_rows": 1000 * 1000 * predict_seconds / max(n_predicted, 1)})
    print("\nSummary: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k.startswith(("F_mean_mae", "D_mae", "D_skill", "L_pooled_mae"))))
    if args.no_tracking:
        return

    setup_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        tags = {**lineage_tags(), **scaler_tags}
        mlflow.set_tags(tags)
        mlflow.log_params({k: str(v) for k, v in cfg.items() if k != "params"})
        mlflow.log_params({f"param.{k}": str(v) for k, v in cfg["params"].items()})
        mlflow.set_tag("evaluation_rows_sha256", rows_hash(rows[rows["split"] == "dev"]))
        for setting, result in results.items():
            for fold, fold_metrics in result["metrics"].items():
                mlflow.log_metrics({f"{setting}_{fold}_{k}": float(v) for k, v in fold_metrics.items() if pd.notna(v)})
            for fold, info in result["info"].items():
                if "rounds" in info:
                    mlflow.log_metric(f"{setting}_{fold}_rounds", info["rounds"])
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
        model_size = None
        if args.save_model and results["D"]["model"] is not None:
            folder, model_size = save_model_files(cfg, results["D"]["model"], metrics, tags, run.info.run_id)
            mlflow.log_metric("model_size_bytes", model_size)
            print(f"Model saved in {folder}")
        elif results.get("D", {}).get("model") is not None:
            model_size = len(pickle.dumps(results["D"]["model"]))
            mlflow.log_metric("model_size_bytes", model_size)
        mlflow.log_metric("time_total_seconds", time.perf_counter() - t_start)
        peak = peak_ram_mb()
        if peak is not None:
            mlflow.log_metric("peak_ram_mb", peak)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pd.concat(tables, ignore_index=True).to_csv(tmp / "per_station_results.csv", index=False)
            (tmp / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            (tmp / "pip_freeze.txt").write_text(pip_freeze_text(), encoding="utf-8")
            for setting, result in results.items():
                for fold, info in result["info"].items():
                    if "curve" in info:
                        info["curve"].to_csv(tmp / f"learning_curve_{setting}_{fold}.csv", index=False)
            mlflow.log_artifacts(str(tmp))
        print(f"Logged MLflow run {run.info.run_id} in experiment {cfg['experiment']}")


if __name__ == "__main__":
    main()
