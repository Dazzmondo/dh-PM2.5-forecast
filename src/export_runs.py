"""
Export the MLflow runs to plain CSV tables, so the report's tables and plots come straight from the logged runs.

USAGE:  python src/export_runs.py [--experiments m3_baselines m3_features ...]
INPUT:  the MLflow store in mlruns/ (written by train.py, tune_xgboost.py, train_gru.py, final_evaluation.py)
OUTPUT: reports/m3/run_summary.csv    one row per run (all experiments, trials excluded): experiment, run name, run ID, status, start
                                      time, the lineage tags, every parameter and every metric
        reports/m3/hpo_trials.csv     one row per Optuna trial (the nested runs of m3_hpo): sampler, seed, the eight hyperparameters,
                                      the objective and the per-fold MAE
        and a short table of the key columns printed to the terminal
Nothing in the store is changed. Peers cannot open the MLflow store, so these tables (and screenshots of the MLflow UI) are what the
report shows; every number in docs/M3.md must be traceable to a row here.

Requires: pip install pandas mlflow
"""
import argparse

import mlflow
import pandas as pd

from config import REPORTS_DIR
from m3_tracking import TRACKING_URI

KEEP_TAGS = ["kind", "model", "sampler", "seed", "device", "git_commit", "dataset_run_id", "image_digest", "python",
             "machine_cpu", "machine_gpu", "evaluation_rows_sha256", "features_file_sha256", "model_dir"]
KEY_COLUMNS = ["experiment", "run_name", "F_mean_mae_seen", "D_mae", "D_rmse", "D_skill", "time_fit_seconds", "inference_ms_per_1000_rows",
               "model_size_bytes", "peak_ram_mb"]


def main():
    """Read every run of the chosen experiments and write the two CSV tables."""
    parser = argparse.ArgumentParser(description="Export MLflow runs to CSV.")
    parser.add_argument("--experiments", nargs="+", help="experiment names (default: all)")
    args = parser.parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    experiments = [e for e in mlflow.search_experiments() if not args.experiments or e.name in args.experiments]
    if not experiments:
        raise SystemExit("No MLflow experiments found in mlruns/. Run an experiment first.")
    names = {e.experiment_id: e.name for e in experiments}
    runs = mlflow.search_runs(experiment_ids=list(names), order_by=["start_time ASC"])
    if runs.empty:
        raise SystemExit("The selected experiments have no runs.")

    table = pd.DataFrame({"experiment": runs["experiment_id"].map(names), "run_name": runs["tags.mlflow.runName"],
                          "run_id": runs["run_id"], "status": runs["status"], "start_time": runs["start_time"]})
    for tag in KEEP_TAGS:
        if f"tags.{tag}" in runs:
            table[tag] = runs[f"tags.{tag}"]
    for column in runs.columns:
        if column.startswith(("params.", "metrics.")):
            table[column.split(".", 1)[1] if column.startswith("metrics.") else column] = runs[column]
    is_trial = table.get("kind", pd.Series("", index=table.index)) == "trial"

    out = REPORTS_DIR / "m3"
    out.mkdir(parents=True, exist_ok=True)
    table[~is_trial].to_csv(out / "run_summary.csv", index=False)
    trials = table[is_trial].dropna(axis=1, how="all")
    if len(trials):
        trials.to_csv(out / "hpo_trials.csv", index=False)
    shown = table[~is_trial][[c for c in KEY_COLUMNS if c in table.columns]]
    print(shown.round(4).to_string(index=False))
    print(f"\n{int((~is_trial).sum())} runs and {int(is_trial.sum())} trials exported to {out}")


if __name__ == "__main__":
    main()
