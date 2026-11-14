"""
Ridge regularisation sweep (Milestone 3, E7d): how does the strength of the penalty change the error?

USAGE:  python src/tune_ridge.py --config configs/hpo_ridge_alpha.yaml [--no-tracking]
INPUT:  the configuration, and via m3_data.py the train and dev rows (the test split and held-out stations are never read)
OUTPUT: one MLflow run per alpha and a summary run in experiment `m3_hpo_ridge`; reports/m3/ridge_sweep/ridge_alpha_sweep.csv;
        reports/figures/m3_08_ridge_alpha.png

What is done: every alpha in the configuration is scored with the same procedure as a search trial of the XGBoost searches: the
objective is the equal-weight mean, over the three selection folds, of the MAE on seen-station rows; the 2024 development MAE of
each alpha is reported next to it but is not used to choose. The inputs are standardised inside each fold's training rows (the
penalty only means the same thing for every input when they are on one scale). Ridge is deterministic, so there is one value per alpha.

The chosen alpha is the one with the lowest objective, except that the starting value (alpha = 1, always included) is kept when it
is within `tie_tolerance_pct` (default 0.01%) of the best: a difference that small is a tie, and the simpler, already used value
wins. The gain over the starting value is printed. Nothing here changes a model file; the chosen alpha goes into the Ridge
configurations by hand.

Requires: pip install pandas pyarrow numpy scikit-learn mlflow pyyaml matplotlib
"""
import argparse
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from analysis_helpers import AQUA, BLUE, ORANGE, get_pyplot, save_figure
from config import REPORTS_DIR
from m3_data import build_features, expanding_folds, load_rows, rows_hash
from m3_tracking import lineage_tags, setup_experiment
from train import fit_predict, score_fold

DEFAULTS = {"experiment": "m3_hpo_ridge", "feature_set": "A", "time_features": "cyclical", "target": "absolute", "start_alpha": 1.0, "tie_tolerance_pct": 0.01,
            "alphas": [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]}


def score_alpha(alpha: float, cfg: dict, rows, X, folds: list) -> dict:
    """Seen-station MAE of a standardised Ridge with this alpha on every fold in `folds` (and the fold-mean)."""
    model_cfg = {"model": "ridge", "target": cfg["target"], "scaling": "standard", "place": "none", "seed": 0, "threads": None,
                 "params": {"alpha": float(alpha)}}
    maes = {}
    for fold in folds:
        forecast, _, _ = fit_predict(model_cfg, rows, X, fold)
        metrics, _ = score_fold(rows, fold, forecast)
        maes[fold.name] = metrics["mae_seen"]
    return maes


def draw(table: pd.DataFrame, start_alpha: float, best_alpha: float):
    """MAE against alpha on a log axis: the selection objective (what is chosen on) and the 2024 score (reported only)."""
    plt = get_pyplot()
    fig, axis = plt.subplots(figsize=(8, 4.6))
    axis.plot(table["alpha"], table["objective"], marker="o", color=BLUE, label="selection folds (the choice is made on this)")
    axis.plot(table["alpha"], table["D_mae"], marker="s", color=ORANGE, label="2024 (reported, not used to choose)")
    axis.axvline(start_alpha, color="#9a9893", linestyle=":", label=f"starting value (alpha = {start_alpha:g})")
    axis.axvline(best_alpha, color=AQUA, linestyle="--", label=f"chosen (alpha = {best_alpha:g})")
    axis.set(xscale="log", xlabel="Ridge alpha (penalty strength)", ylabel="MAE (ug/m3)", title="Ridge regularisation sweep (E7d)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    """Score every alpha, print the table, save the figure and CSV, and log to MLflow."""
    parser = argparse.ArgumentParser(description="Ridge alpha sweep.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **yaml.safe_load(args.config.read_text(encoding="utf-8"))}
    alphas = sorted(set(float(a) for a in cfg["alphas"]) | {float(cfg["start_alpha"])})
    if any(a <= 0 for a in alphas):
        raise SystemExit("Every alpha must be above 0.")

    t0 = time.perf_counter()
    rows = load_rows()
    X = build_features(rows, cfg["feature_set"], cfg["time_features"])
    folds = expanding_folds(rows)
    selection, dev = [f for f in folds if f.name != "D"], [f for f in folds if f.name == "D"]
    print(f"Ridge sweep: {len(alphas)} alphas from {alphas[0]:g} to {alphas[-1]:g}; objective = mean seen-station MAE over "
          f"{[f.name for f in selection]}")
    records = []
    for alpha in alphas:
        maes = score_alpha(alpha, cfg, rows, X, selection)
        d_mae = score_alpha(alpha, cfg, rows, X, dev)["D"]
        records.append({"alpha": alpha, "objective": float(np.mean(list(maes.values()))), "D_mae": d_mae,
                        **{f"mae_{k}": v for k, v in maes.items()}})
        print(f"  alpha {alpha:>9g}: objective {records[-1]['objective']:.5f}  2024 MAE {d_mae:.5f}")
    table = pd.DataFrame(records)
    lowest = table.loc[table["objective"].idxmin()]
    start = table[table["alpha"] == float(cfg["start_alpha"])].iloc[0]
    tie = 100 * (start["objective"] / lowest["objective"] - 1) <= cfg["tie_tolerance_pct"]
    best = start if tie else lowest                         # a tie with the starting value is resolved in its favour
    gain = 100 * (best["objective"] / start["objective"] - 1)
    plateau = table[100 * (table["objective"] / lowest["objective"] - 1) <= cfg["tie_tolerance_pct"]]["alpha"]
    print(f"\nLowest objective: alpha {lowest['alpha']:g} ({lowest['objective']:.5f}); {len(plateau)} alphas ({plateau.min():g} to "
          f"{plateau.max():g}) are within {cfg['tie_tolerance_pct']}% of it.\nChosen alpha: {best['alpha']:g}"
          f"{' (a tie with the starting value, which is kept)' if tie else ''}; starting alpha {cfg['start_alpha']:g}: "
          f"{start['objective']:.5f}; change {gain:+.3f}% (2024: {best['D_mae']:.5f} against {start['D_mae']:.5f})")

    out = REPORTS_DIR / "m3" / "ridge_sweep"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ridge_alpha_sweep.csv", index=False)
    fig = draw(table, float(cfg["start_alpha"]), float(best["alpha"]))
    save_figure(fig, "m3_08_ridge_alpha.png")
    if args.no_tracking:
        return

    setup_experiment(cfg["experiment"])
    for record in records:
        with mlflow.start_run(run_name=f"ridge_alpha_{record['alpha']:g}"):
            mlflow.set_tags({"kind": "sweep_point", "model": "ridge"})
            mlflow.log_param("alpha", record["alpha"])
            mlflow.log_metrics({k: float(v) for k, v in record.items() if k != "alpha"})
    with mlflow.start_run(run_name="ridge_alpha_summary"):
        mlflow.set_tags({**lineage_tags(), "kind": "summary", "model": "ridge", "evaluation_rows_sha256": rows_hash(rows[rows["split"] == "dev"])})
        mlflow.log_params({k: str(v) for k, v in cfg.items()})
        mlflow.log_metrics({"best_alpha": float(best["alpha"]), "best_objective": float(best["objective"]),
                            "start_objective": float(start["objective"]), "change_vs_start_pct": float(gain),
                            "best_D_mae": float(best["D_mae"]), "start_D_mae": float(start["D_mae"]),
                            "time_total_seconds": time.perf_counter() - t0})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            table.to_csv(tmp / "ridge_alpha_sweep.csv", index=False)
            fig.savefig(tmp / "ridge_alpha.png", dpi=150)
            (tmp / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            mlflow.log_artifacts(str(tmp))
    print("Logged the sweep in experiment m3_hpo_ridge.")


if __name__ == "__main__":
    main()
