"""
Hyperparameter search for XGBoost: random search against TPE (Bayesian optimisation), same space, folds, metric and budget.

USAGE:  python src/tune_xgboost.py --config configs/hpo_random_vs_tpe.yaml [--n-trials N] [--seeds 0 1] [--samplers random tpe] [--no-tracking]
INPUT:  the configuration, and via m3_data.py the train and dev rows (the test split and held-out stations are never read)
OUTPUT: MLflow runs in mlruns/ (experiment `m3_hpo`): one run per (sampler, seed) study with one nested run per trial, a run for the
        default configuration of each seed, and a summary run holding the best-so-far curves, the hyperparameter importance and the tables.

The protocol (fixed before any result was seen):
  - Objective (minimised): the equal-weight mean, over the three selection folds (validation years 2021, 2022, 2023), of the MAE on
    SEEN-station rows, on the raw ug/m3 scale. The development year (2024) is NOT part of the objective.
  - Inside each fold the number of boosting rounds is set by early stopping on an inner window (the last 12 months of the fold's
    training rows by target time) and then reused when refitting on all the fold's training rows (train.fit_predict).
  - Both samplers get the same search space, the same folds, the same metric and the same number of trials; there is no pruner, so
    a trial always costs the same budget. The sampler seed and the XGBoost seed are the same number (0, 1, 2 ...).
  - The default configuration is evaluated the same way (one reference per seed).
  - After each search, the best configuration (and the default) is fitted on 2018-2023 and scored ONCE on the development year;
    that score is reported, never used to choose a configuration.

TPE draws its first 10 trials at random (Optuna's n_startup_trials), so with the same seed they are the SAME configurations as
random search's first 10 trials; the two samplers differ from trial 11 on, and that is where the comparison is made.

Search space (the default below; a configuration can replace it with a `space:` entry, see "Configuration"):
               learning_rate 0.01-0.3 (log)   max_depth 3-10   min_child_weight 1-20   subsample 0.5-1.0   colsample_bytree 0.5-1.0
               reg_lambda 1e-3-10 (log)       reg_alpha 1e-3-10 (log)                   gamma 0-5

Configuration (YAML), besides experiment, feature_set, time_features, target, objective, samplers, seeds, n_trials,
early_stopping_rounds and n_estimators (the cap on boosting rounds):
  space:     {parameter: {low: .., high: .., log: true|false, type: float|int}, ...}   the search space of this run; every parameter
             must be an XGBoost hyperparameter; the whole space is logged with the run
  evaluate:  {name: {parameter: value, ...}, ...}   NO search: each named configuration is scored with every seed in `seeds` (the
             objective on the selection folds, then 2024 once) to measure how much the score moves with the seed alone
Hyperhyperparameter importance (Optuna) describes how the sampled trials varied with each hyperparameter; it is not a causal effect.

Requires: pip install pandas pyarrow numpy scikit-learn xgboost optuna mlflow pyyaml matplotlib
"""
import argparse
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml

from m3_data import build_features, expanding_folds, load_rows, rows_hash
from m3_tracking import lineage_tags, peak_ram_mb, pip_freeze_text, setup_experiment
from train import XGBOOST_DEFAULTS, fit_predict, score_fold

optuna.logging.set_verbosity(optuna.logging.WARNING)
SAMPLERS = {"random": optuna.samplers.RandomSampler, "tpe": optuna.samplers.TPESampler}
HPO_DEFAULTS = {"experiment": "m3_hpo", "feature_set": "A", "time_features": "cyclical", "target": "absolute",
                "samplers": ["random", "tpe"], "seeds": [0, 1, 2], "n_trials": 50, "early_stopping_rounds": 50,
                "n_estimators": 2000, "threads": None, "objective": None, "space": None, "evaluate": None}
DEFAULT_PARAMS = {k: XGBOOST_DEFAULTS[k] for k in ("learning_rate", "max_depth")}     # the reference configuration: lr 0.1, depth 6


DEFAULT_SPACE = {"learning_rate": {"low": 0.01, "high": 0.3, "log": True},
                 "max_depth": {"low": 3, "high": 10, "type": "int"},
                 "min_child_weight": {"low": 1, "high": 20, "type": "int"},
                 "subsample": {"low": 0.5, "high": 1.0},
                 "colsample_bytree": {"low": 0.5, "high": 1.0},
                 "reg_lambda": {"low": 1e-3, "high": 10, "log": True},
                 "reg_alpha": {"low": 1e-3, "high": 10, "log": True},
                 "gamma": {"low": 0.0, "high": 5.0}}
XGBOOST_PARAMETERS = {"learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode",
                      "reg_lambda", "reg_alpha", "gamma", "max_delta_step", "max_bin"}


def check_space(space: dict) -> None:
    """Stop with a clear message if a search space is malformed (unknown parameter, empty range, log scale from zero)."""
    for name, spec in space.items():
        if name not in XGBOOST_PARAMETERS:
            raise SystemExit(f"space: {name!r} is not a hyperparameter this script searches (use one of {sorted(XGBOOST_PARAMETERS)}).")
        if not {"low", "high"} <= set(spec) or not spec["low"] < spec["high"]:
            raise SystemExit(f"space: {name} needs low and high with low < high (got {spec}).")
        if spec.get("log") and spec["low"] <= 0:
            raise SystemExit(f"space: {name} is on a log scale, so low must be above 0 (got {spec['low']}).")
        if spec.get("type", "float") not in ("float", "int"):
            raise SystemExit(f"space: {name} has type {spec['type']!r} (use float or int).")


def suggest(trial: optuna.Trial, space: dict) -> dict:
    """The hyperparameters of one trial, drawn from the search space of the configuration."""
    params = {}
    for name, spec in space.items():
        if spec.get("type", "float") == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log")))
        else:
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log")))
    return params


def model_config(hpo: dict, params: dict, seed: int) -> dict:
    """The train.py-style configuration for XGBoost with these hyperparameters, early stopping on and the given seed."""
    extra = {"objective": hpo["objective"]} if hpo["objective"] else {}        # e.g. reg:absoluteerror; default is squared error
    return {"model": "xgboost", "target": hpo["target"], "scaling": "none", "place": "none", "seed": seed, "threads": hpo["threads"],
            "params": {**params, **extra, "n_estimators": hpo["n_estimators"], "early_stopping_rounds": hpo["early_stopping_rounds"]}}


def evaluate(hpo: dict, params: dict, seed: int, rows: pd.DataFrame, X: pd.DataFrame, folds: list) -> dict:
    """Fit and score every fold in `folds` with these hyperparameters; return the per-fold seen-station MAE, rounds and time."""
    cfg = model_config(hpo, params, seed)
    maes, rounds, fit_seconds = {}, {}, 0.0
    for fold in folds:
        forecast, info, _ = fit_predict(cfg, rows, X, fold)
        metrics, _ = score_fold(rows, fold, forecast)
        maes[fold.name], rounds[fold.name] = metrics["mae_seen"], info["rounds"]
        fit_seconds += info["fit_seconds"]
    if "D" in maes:
        return {"mae_by_fold": maes, "rounds": rounds, "fit_seconds": fit_seconds, "D_mae": maes["D"], "D_rmse": metrics["rmse_seen"],
                "D_skill": metrics["skill_seen"]}
    return {"mae_by_fold": maes, "rounds": rounds, "fit_seconds": fit_seconds, "objective": float(np.mean(list(maes.values())))}


def run_study(sampler: str, seed: int, hpo: dict, rows, X, selection_folds, dev_fold, tracking: bool) -> dict:
    """One study (sampler, seed): n_trials trials, then the best configuration scored once on the development year."""
    study = optuna.create_study(direction="minimize", sampler=SAMPLERS[sampler](seed=seed), pruner=optuna.pruners.NopPruner())
    if tracking:
        mlflow.start_run(run_name=f"hpo_{sampler}_seed{seed}")      # closed with mlflow.end_run() after the search
        mlflow.set_tags({"sampler": sampler, "seed": str(seed), "kind": "study", "n_trials": str(hpo["n_trials"])})

    def objective(trial: optuna.Trial) -> float:
        """Evaluate one trial on the selection folds and log it as a nested MLflow run."""
        params = suggest(trial, hpo["space"])
        result = evaluate(hpo, params, seed, rows, X, selection_folds)
        trial.set_user_attr("mae_by_fold", result["mae_by_fold"])
        trial.set_user_attr("rounds", result["rounds"])
        trial.set_user_attr("fit_seconds", result["fit_seconds"])
        if tracking:
            with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
                mlflow.set_tags({"sampler": sampler, "seed": str(seed), "kind": "trial"})
                mlflow.log_params(params)
                mlflow.log_metrics({"objective": result["objective"], "fit_seconds": result["fit_seconds"],
                                    **{f"mae_{k}": v for k, v in result["mae_by_fold"].items()}})
        return result["objective"]

    t0 = time.perf_counter()
    for number in range(hpo["n_trials"]):
        study.optimize(objective, n_trials=1)
        if tracking:
            mlflow.log_metric("best_so_far", study.best_value, step=number)
        print(f"    {sampler:6s} seed {seed} trial {number + 1:3d}/{hpo['n_trials']}  value {study.trials[-1].value:.4f}  "
              f"best {study.best_value:.4f}  ({time.perf_counter() - t0:.0f}s elapsed)")
    best = evaluate(hpo, study.best_params, seed, rows, X, [dev_fold])      # scored once on 2024; not used to choose anything
    out = {"sampler": sampler, "seed": seed, "best_objective": study.best_value, "best_params": study.best_params,
           "D_mae": best["D_mae"], "D_rmse": best["D_rmse"], "D_skill": best["D_skill"], "D_rounds": best["rounds"]["D"],
           "search_seconds": time.perf_counter() - t0, "trials": study.trials_dataframe(), "study": study}
    if tracking:
        mlflow.log_metrics({"best_objective": out["best_objective"], "D_mae": out["D_mae"], "D_rmse": out["D_rmse"],
                            "D_skill": out["D_skill"], "search_seconds": out["search_seconds"]})
        mlflow.log_params({f"best.{k}": v for k, v in study.best_params.items()})
        with tempfile.TemporaryDirectory() as tmp:
            out["trials"].to_csv(Path(tmp) / "trials.csv", index=False)
            mlflow.log_artifacts(tmp)
        mlflow.end_run()
    return out


def run_default(seed: int, hpo: dict, rows, X, selection_folds, dev_fold, tracking: bool) -> dict:
    """The default configuration (learning rate 0.1, depth 6), evaluated exactly like a trial and then scored once on 2024."""
    selection = evaluate(hpo, DEFAULT_PARAMS, seed, rows, X, selection_folds)
    dev = evaluate(hpo, DEFAULT_PARAMS, seed, rows, X, [dev_fold])
    out = {"seed": seed, "objective": selection["objective"], "D_mae": dev["D_mae"], "D_rmse": dev["D_rmse"], "D_skill": dev["D_skill"]}
    if tracking:
        with mlflow.start_run(run_name=f"hpo_default_seed{seed}"):
            mlflow.set_tags({"sampler": "default", "seed": str(seed), "kind": "default"})
            mlflow.log_params(DEFAULT_PARAMS)
            mlflow.log_metrics({"best_objective": out["objective"], "D_mae": out["D_mae"], "D_rmse": out["D_rmse"], "D_skill": out["D_skill"],
                                **{f"mae_{k}": v for k, v in selection["mae_by_fold"].items()}})
    return out


def run_evaluations(hpo: dict, rows, X, selection_folds, dev_fold, tracking: bool) -> None:
    """Seed-noise mode: score each named configuration with every seed (selection-fold objective, then 2024 once); no search."""
    records = []
    for name, params in hpo["evaluate"].items():
        for seed in hpo["seeds"]:
            selection = evaluate(hpo, params, seed, rows, X, selection_folds)
            dev = evaluate(hpo, params, seed, rows, X, [dev_fold])
            record = {"config": name, "seed": seed, "objective": selection["objective"], "D_mae": dev["D_mae"], "D_rmse": dev["D_rmse"],
                      "D_skill": dev["D_skill"], **{f"mae_{k}": v for k, v in selection["mae_by_fold"].items()},
                      **{f"rounds_{k}": v for k, v in {**selection["rounds"], **dev["rounds"]}.items()}}
            records.append(record)
            print(f"    {name:28s} seed {seed}: objective {record['objective']:.4f}  2024 MAE {record['D_mae']:.4f}  "
                  f"rounds {selection['rounds']} / {dev['rounds']}")
            if tracking:
                with mlflow.start_run(run_name=f"eval_{name}_seed{seed}"):
                    mlflow.set_tags({"kind": "eval", "config": name, "seed": str(seed)})
                    mlflow.log_params(dict(params))
                    mlflow.log_metrics({k: float(v) for k, v in record.items() if k not in ("config", "seed") and not k.startswith("rounds")})
    table = pd.DataFrame(records)
    spread = table.groupby("config")[["objective", "D_mae"]].agg(["mean", "std", "min", "max"])
    spread[("objective", "range")] = spread[("objective", "max")] - spread[("objective", "min")]
    spread[("D_mae", "range")] = spread[("D_mae", "max")] - spread[("D_mae", "min")]
    print("\nSeed-to-seed spread of each configuration (std is the sample standard deviation over the seeds):")
    print(spread.round(5).to_string())
    if tracking:
        with mlflow.start_run(run_name="eval_summary"):
            mlflow.set_tags({**lineage_tags(), "kind": "summary", "evaluation_rows_sha256": rows_hash(rows[rows["split"] == "dev"])})
            mlflow.log_params({k: str(v) for k, v in hpo.items() if k != "space"})
            for name, group in table.groupby("config"):
                mlflow.log_metrics({f"{name}.objective_mean": group["objective"].mean(), f"{name}.objective_std": group["objective"].std(),
                                    f"{name}.objective_range": group["objective"].max() - group["objective"].min(),
                                    f"{name}.D_mae_mean": group["D_mae"].mean(), f"{name}.D_mae_std": group["D_mae"].std()})
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                table.to_csv(tmp / "seed_noise.csv", index=False)
                spread.to_csv(tmp / "seed_noise_summary.csv")
                (tmp / "hpo_config.yaml").write_text(yaml.safe_dump(hpo, sort_keys=False), encoding="utf-8")
                mlflow.log_artifacts(str(tmp))
        print("Logged the summary run (eval_summary).")


def importance_table(studies: list) -> pd.DataFrame:
    """Optuna hyperparameter importance per study (needs a few trials); NaN-free rows only. Association within the sampled trials, not causal."""
    rows = []
    for study_result in studies:
        try:
            importance = optuna.importance.get_param_importances(study_result["study"])
        except Exception:                                   # too few trials or no variation: skip rather than guess
            continue
        rows.extend({"sampler": study_result["sampler"], "seed": study_result["seed"], "parameter": k, "importance": v}
                    for k, v in importance.items())
    return pd.DataFrame(rows)


def best_so_far_figure(studies: list, defaults: list, path: Path) -> None:
    """Plot best-so-far objective against trial number, mean and range over seeds, per sampler, with the default as a dashed line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    colors = {"random": "#1f77b4", "tpe": "#d62728"}
    for sampler in sorted({s["sampler"] for s in studies}):
        curves = np.array([np.minimum.accumulate(s["trials"]["value"].to_numpy()) for s in studies if s["sampler"] == sampler])
        x = np.arange(1, curves.shape[1] + 1)
        axis.plot(x, curves.mean(axis=0), label=f"{sampler} (mean of {len(curves)} seeds)", color=colors.get(sampler))
        axis.fill_between(x, curves.min(axis=0), curves.max(axis=0), alpha=0.2, color=colors.get(sampler))
    axis.axhline(np.mean([d["objective"] for d in defaults]), linestyle="--", color="grey", label="default configuration")
    axis.set_xlabel("trial")
    axis.set_ylabel("best objective so far (mean seen-station MAE, ug/m3)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Run the studies for every (sampler, seed), the default references and the summary; print and log the results."""
    parser = argparse.ArgumentParser(description="Random search vs TPE for XGBoost.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n-trials", type=int, help="override the number of trials per study (e.g. 3 to time a first batch)")
    parser.add_argument("--seeds", type=int, nargs="+", help="override the seeds")
    parser.add_argument("--samplers", nargs="+", choices=list(SAMPLERS), help="override the samplers")
    parser.add_argument("--no-tracking", action="store_true", help="print results without writing MLflow runs")
    args = parser.parse_args()
    hpo = {**HPO_DEFAULTS, **yaml.safe_load(args.config.read_text(encoding="utf-8"))}
    for key, value in (("n_trials", args.n_trials), ("seeds", args.seeds), ("samplers", args.samplers)):
        if value:
            hpo[key] = value
    tracking = not args.no_tracking
    hpo["space"] = hpo["space"] or DEFAULT_SPACE
    check_space(hpo["space"])

    rows = load_rows()
    X = build_features(rows, hpo["feature_set"], hpo["time_features"])
    folds = expanding_folds(rows)
    selection_folds, dev_fold = [f for f in folds if f.name != "D"], [f for f in folds if f.name == "D"][0]
    print(f"HPO: samplers {hpo['samplers']}, seeds {hpo['seeds']}, {hpo['n_trials']} trials per study, feature set {hpo['feature_set']}, "
          f"target {hpo['target']}; objective = mean seen-station MAE over folds {[f.name for f in selection_folds]}")
    if tracking:
        setup_experiment(hpo["experiment"])

    if hpo["evaluate"]:                                     # seed-noise mode: score named configurations, no search
        run_evaluations(hpo, rows, X, selection_folds, dev_fold, tracking)
        return

    studies, defaults = [], []
    t0 = time.perf_counter()
    for seed in hpo["seeds"]:
        defaults.append(run_default(seed, hpo, rows, X, selection_folds, dev_fold, tracking))
        for sampler in hpo["samplers"]:
            studies.append(run_study(sampler, seed, hpo, rows, X, selection_folds, dev_fold, tracking))

    table = pd.DataFrame([{"sampler": s["sampler"], "seed": s["seed"], "best_objective": s["best_objective"], "D_mae": s["D_mae"],
                           "D_skill": s["D_skill"], "D_rounds": s["D_rounds"], "search_minutes": s["search_seconds"] / 60,
                           **{f"best.{k}": v for k, v in s["best_params"].items()}} for s in studies] +
                         [{"sampler": "default", "seed": d["seed"], "best_objective": d["objective"], "D_mae": d["D_mae"],
                           "D_skill": d["D_skill"]} for d in defaults])
    spread = table.groupby("sampler")[["best_objective", "D_mae"]].agg(["mean", "std"]).round(4)
    print("\nBest configuration per study (objective = selection folds; D_mae = scored once on 2024):")
    print(table.round(4).to_string(index=False))
    print("\nMean and spread over seeds:")
    print(spread.to_string())
    print(f"\nTotal time {(time.perf_counter() - t0) / 60:.1f} min")
    if not tracking:
        return

    importance = importance_table(studies)
    with mlflow.start_run(run_name="hpo_summary"):
        mlflow.set_tags({**lineage_tags(), "kind": "summary", "evaluation_rows_sha256": rows_hash(rows[rows["split"] == "dev"])})
        mlflow.log_params({k: str(v) for k, v in hpo.items()})
        for sampler, group in table.groupby("sampler"):
            mlflow.log_metrics({f"{sampler}_best_objective_mean": group["best_objective"].mean(),
                                f"{sampler}_best_objective_std": group["best_objective"].std(ddof=0),
                                f"{sampler}_D_mae_mean": group["D_mae"].mean(), f"{sampler}_D_mae_std": group["D_mae"].std(ddof=0)})
        mlflow.log_metric("time_total_seconds", time.perf_counter() - t0)
        peak = peak_ram_mb()
        if peak is not None:
            mlflow.log_metric("peak_ram_mb", peak)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            table.to_csv(tmp / "hpo_results.csv", index=False)
            importance.to_csv(tmp / "parameter_importance.csv", index=False)
            best_so_far_figure(studies, defaults, tmp / "best_so_far.png")
            (tmp / "pip_freeze.txt").write_text(pip_freeze_text(), encoding="utf-8")
            (tmp / "hpo_config.yaml").write_text(yaml.safe_dump(hpo, sort_keys=False), encoding="utf-8")
            mlflow.log_artifacts(str(tmp))
    print("Logged the summary run (hpo_summary) with the figure, results table and hyperparameter importance.")


if __name__ == "__main__":
    main()
