"""
Hyperparameter search for the GRU (Milestone 3, E9b): TPE over the architecture, the learning rate and the batch size.

USAGE:  python src/tune_gru.py --config configs/hpo_gru.yaml [--n-trials N] [--device auto|cpu|cuda] [--no-tracking]
INPUT:  the configuration, and via m3_data.py the train and dev rows (the test split and held-out stations are never read)
OUTPUT: MLflow runs in experiment `m3_hpo_gru` (one run for the search with one nested run per trial, a run for the starting
        configuration, a summary run with the figure, the trials table and the parameter importance);
        reports/figures/m3_09_gru_search.png; reports/m3/gru_search/; and configs/gru_tuned_final.yaml, the configuration of the
        best trial, ready for a final run with three seeds (python src/train_gru.py --config configs/gru_tuned_final.yaml --save-model)

The protocol is the one of the XGBoost searches (section 3 of docs/M3.md):
  - Objective (minimised): the equal-weight mean, over the three selection folds (validation years 2021, 2022, 2023), of the MAE on
    SEEN-station rows, on the raw ug/m3 scale. The development year 2024 is NOT part of the objective and is not read here.
  - Inside each fold, training uses early stopping on an inner window (the last 12 months of the fold's training rows by target time),
    then refits on all the fold's training rows for the epoch count that early stopping chose (train_gru.run_fold).
  - TPE (Optuna) with the first 10 trials drawn at random, one sampler seed; the same budget is spent on every trial (no pruner).
  - The STARTING configuration (hidden 32, 1 layer, no dropout, learning rate 0.003, batch 256) is scored in the same way first,
    so the gain of the search over the starting values is measured, not assumed.
Search space (set in the configuration; the loss is fixed at L1 because E9a showed L1 better than squared error in every pair):
  hidden_size {16, 32, 64, 128}   num_layers {1, 2}   dropout 0-0.3   learning rate 3e-4 to 1e-2 (log)   batch size {128, 256, 512, 1024}
Weight decay is not searched: the learning curves show no over-fitting for the GRU (inner validation loss at or below the training loss).

Requires: pip install pandas pyarrow numpy torch optuna mlflow pyyaml matplotlib
"""
import argparse
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import yaml

from analysis_helpers import BLUE, ORANGE, get_pyplot, save_figure
from config import REPO_ROOT, REPORTS_DIR
from m3_data import expanding_folds, load_rows, rows_hash, time_features
from m3_tracking import lineage_tags, peak_ram_mb, pip_freeze_text, setup_experiment
from train import check_persistence_benchmark
from train_gru import CFG_DEFAULTS, HP_DEFAULTS, build_sequences, choose_device, run_settings

optuna.logging.set_verbosity(optuna.logging.WARNING)
STARTING_HP = {"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "lr": 0.003, "batch_size": 256}      # the starting configuration (section 3.1)
DEFAULTS = {"experiment": "m3_hpo_gru", "n_trials": 40, "sampler_seed": 0, "model_seed": 0, "loss": "l1", "max_epochs": 50, "patience": 7,
            "time_features": "cyclical", "device": "auto", "threads": None,
            "space": {"hidden_size": [16, 32, 64, 128], "num_layers": [1, 2], "dropout": {"low": 0.0, "high": 0.3},
                      "lr": {"low": 3e-4, "high": 1e-2, "log": True}, "batch_size": [128, 256, 512, 1024]}}


def suggest(trial: optuna.Trial, space: dict) -> dict:
    """The hyperparameters of one trial: a list in the space is a categorical choice, a {low, high} entry a (log) range."""
    hp = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            hp[name] = trial.suggest_categorical(name, spec)
        else:
            hp[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log")))
    return {**STARTING_HP, **hp}


def evaluate(hp: dict, cfg: dict, device, rows, sequence, static, folds_by_name: dict) -> dict:
    """Train and score on the three selection folds; return the per-fold seen-station MAE, epochs, fit time and their mean."""
    run_cfg = {**CFG_DEFAULTS, "settings": ["F"], "loss": cfg["loss"], "max_epochs": cfg["max_epochs"], "patience": cfg["patience"],
               "time_features": cfg["time_features"]}
    result = run_settings(run_cfg, {**HP_DEFAULTS, **hp}, cfg["model_seed"], device, rows, sequence, static, folds_by_name)
    part = result["settings"]["F"]
    maes = {name: m["mae_seen"] for name, m in part["metrics"].items()}
    return {"mae_by_fold": maes, "epochs": {name: i["rounds"] for name, i in part["info"].items()},
            "fit_seconds": sum(i["fit_seconds"] for i in part["info"].values()), "objective": float(np.mean(list(maes.values())))}


def draw(trials: pd.DataFrame, reference: float, path_name: str):
    """Best-so-far objective against the trial number, with the starting configuration as a dashed line, and every trial as a dot."""
    plt = get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    best = trials["value"].cummin()
    axes[0].scatter(trials["number"] + 1, trials["value"], color="#9a9893", s=18, label="trial")
    axes[0].plot(trials["number"] + 1, best, color=BLUE, label="best so far")
    axes[0].axhline(reference, color=ORANGE, linestyle="--", label="starting configuration")
    axes[0].set(title="GRU search: objective by trial", xlabel="trial", ylabel="mean seen-station MAE over the 3 folds (ug/m3)")
    axes[0].legend(fontsize=8)
    sizes = trials["params_hidden_size"].astype(float)
    scatter = axes[1].scatter(trials["params_lr"], trials["value"], c=np.log2(trials["params_batch_size"].astype(float)), s=20 + sizes, cmap="viridis")
    axes[1].set(xscale="log", title="Objective by learning rate (colour: log2 batch size, size: hidden units)", xlabel="learning rate", ylabel="objective")
    fig.colorbar(scatter, ax=axes[1], label="log2(batch size)")
    fig.suptitle(f"GRU hyperparameter search (E9b) on {path_name}")
    fig.tight_layout()
    return fig


def main():
    """Score the starting configuration, run the TPE search, write the figure, tables and the final-run configuration, and log to MLflow."""
    parser = argparse.ArgumentParser(description="GRU hyperparameter search (TPE).")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **yaml.safe_load(args.config.read_text(encoding="utf-8"))}
    if args.n_trials:
        cfg["n_trials"] = args.n_trials
    if args.device:
        cfg["device"] = args.device
    if cfg["loss"] not in ("mse", "l1"):
        raise SystemExit("loss must be mse or l1.")
    if cfg["threads"]:
        torch.set_num_threads(int(cfg["threads"]))
    device = choose_device(cfg["device"])
    tracking = not args.no_tracking

    t0 = time.perf_counter()
    rows = load_rows()
    check_persistence_benchmark(rows)
    sequence = build_sequences(rows)
    static = time_features(rows, cfg["time_features"]).to_numpy(dtype=float)
    folds_by_name = {f.name: f for f in expanding_folds(rows)}
    print(f"GRU search on {device}: {cfg['n_trials']} trials, loss {cfg['loss']}, space {cfg['space']}; objective = mean seen-station MAE "
          f"over F2021, F2022, F2023")
    if tracking:
        setup_experiment(cfg["experiment"])

    print("\nStarting configuration:", STARTING_HP)
    reference = evaluate(STARTING_HP, cfg, device, rows, sequence, static, folds_by_name)
    print(f"  starting configuration: objective {reference['objective']:.4f}  folds {reference['mae_by_fold']}  epochs {reference['epochs']}")
    if tracking:
        with mlflow.start_run(run_name="gru_start_configuration"):
            mlflow.set_tags({"kind": "reference", "model": "gru"})
            mlflow.log_params({f"hp.{k}": v for k, v in STARTING_HP.items()} | {"loss": cfg["loss"]})
            mlflow.log_metrics({"objective": reference["objective"], "fit_seconds": reference["fit_seconds"],
                                **{f"mae_{k}": v for k, v in reference["mae_by_fold"].items()}})

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg["sampler_seed"]), pruner=optuna.pruners.NopPruner())
    if tracking:
        mlflow.start_run(run_name="gru_search")
        mlflow.set_tags({**lineage_tags(), "kind": "study", "model": "gru", "sampler": "tpe", "n_trials": str(cfg["n_trials"])})

    def objective(trial: optuna.Trial) -> float:
        """One trial: draw the hyperparameters, score the three folds, log a nested run."""
        hp = suggest(trial, cfg["space"])
        result = evaluate(hp, cfg, device, rows, sequence, static, folds_by_name)
        trial.set_user_attr("mae_by_fold", result["mae_by_fold"])
        trial.set_user_attr("epochs", result["epochs"])
        trial.set_user_attr("fit_seconds", result["fit_seconds"])
        if tracking:
            with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
                mlflow.set_tags({"kind": "trial", "model": "gru"})
                mlflow.log_params({f"hp.{k}": v for k, v in hp.items()})
                mlflow.log_metrics({"objective": result["objective"], "fit_seconds": result["fit_seconds"],
                                    **{f"mae_{k}": v for k, v in result["mae_by_fold"].items()}})
        return result["objective"]

    t_search = time.perf_counter()
    for number in range(cfg["n_trials"]):
        study.optimize(objective, n_trials=1)
        if tracking:
            mlflow.log_metric("best_so_far", study.best_value, step=number)
        print(f"  trial {number + 1:3d}/{cfg['n_trials']}  value {study.trials[-1].value:.4f}  best {study.best_value:.4f}  "
              f"({time.perf_counter() - t_search:.0f}s elapsed)")
    best_hp = {**STARTING_HP, **study.best_params}
    best_hp = {k: (int(v) if k in ("hidden_size", "num_layers", "batch_size") else float(v)) for k, v in best_hp.items()}
    change = 100 * (study.best_value / reference["objective"] - 1)
    print(f"\nBest trial {study.best_trial.number}: objective {study.best_value:.4f} against the starting configuration "
          f"{reference['objective']:.4f} ({change:+.2f}%)\nBest hyperparameters: {best_hp}")

    trials = study.trials_dataframe()
    try:
        importance = pd.DataFrame(list(optuna.importance.get_param_importances(study).items()), columns=["parameter", "importance"])
    except Exception:                                       # too few trials or no variation: skip rather than guess
        importance = pd.DataFrame(columns=["parameter", "importance"])
    out = REPORTS_DIR / "m3" / "gru_search"
    out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out / "gru_search_trials.csv", index=False)
    importance.to_csv(out / "gru_search_importance.csv", index=False)
    fig = draw(trials, reference["objective"], str(device))
    save_figure(fig, "m3_09_gru_search.png")

    final = {"experiment": "m3_neural", "run_name": "gru_tuned_final", "settings": ["F", "D"], "seeds": [0, 1, 2], "loss": cfg["loss"],
             "max_epochs": cfg["max_epochs"], "patience": cfg["patience"], "hp": {**best_hp}}
    header = (f"# E9b final: the best configuration of the GRU search (tune_gru.py, trial {study.best_trial.number}, objective "
              f"{study.best_value:.4f} against {reference['objective']:.4f} for the starting configuration); three seeds, selection folds and 2024.\n")
    (REPO_ROOT / "configs" / "gru_tuned_final.yaml").write_text(header + yaml.safe_dump(final, sort_keys=False), encoding="utf-8")
    print("Wrote configs/gru_tuned_final.yaml")
    if not tracking:
        return

    mlflow.log_metrics({"best_objective": study.best_value, "start_objective": reference["objective"], "change_vs_start_pct": change,
                        "search_seconds": time.perf_counter() - t_search, "time_total_seconds": time.perf_counter() - t0})
    mlflow.log_params({f"best.{k}": v for k, v in best_hp.items()})
    mlflow.set_tag("evaluation_rows_sha256", rows_hash(rows[rows["split"] == "dev"]))
    peak = peak_ram_mb()
    if peak is not None:
        mlflow.log_metric("peak_ram_mb", peak)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        trials.to_csv(tmp / "trials.csv", index=False)
        importance.to_csv(tmp / "parameter_importance.csv", index=False)
        fig.savefig(tmp / "gru_search.png", dpi=150)
        (tmp / "pip_freeze.txt").write_text(pip_freeze_text(), encoding="utf-8")
        (tmp / "search_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        mlflow.log_artifacts(str(tmp))
    mlflow.end_run()
    print("Logged the search in experiment m3_hpo_gru.")


if __name__ == "__main__":
    main()
