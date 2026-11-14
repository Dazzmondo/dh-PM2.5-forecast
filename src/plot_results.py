"""
Draw the Milestone 3 figures from the logged MLflow runs, so every plot in the report traces to a run.

USAGE:  python src/plot_results.py [--figure learning variants hpo benchmark place | all] [--store mlruns/mlflow.db]
INPUT:  the MLflow store (default: mlruns/mlflow.db), written by train.py, train_gru.py, tune_xgboost.py and benchmark.py
OUTPUT: reports/figures/m3_01_learning_curves.png   m3_02_variants.png   m3_03_hpo.png   m3_04_benchmark.png   m3_05_place.png
        reports/m3/figures_data/<figure>.csv        the numbers each figure shows
Nothing in the store is changed. A figure whose runs are not in the store is skipped with a message.

  learning   (E6a) XGBoost train and inner-validation MAE by boosting round for every fold, and the GRU's loss by epoch for every seed;
             the best round or epoch is marked. Shows under- and overfitting and what early stopping chose.
  variants   (E1-E6b) every run of the baselines and reference models: seen-station fold MAE and development MAE, with the change from
             that model's reference configuration in percent and the 1% adoption threshold marked.
  hpo        (E7) best-so-far objective by trial for random search against TPE (mean and range over the seeds) with the default
             configuration, the parameter importance, and each study's best configuration scored once on the development year.
  benchmark  (E11) XGBoost speed-up and efficiency against threads (ideal line and Amdahl fit), and fit times on CPU vs GPU for XGBoost and
             the GRU, per machine.
  place      (E8) leave-one-city-out MAE per city for each place-feature variant, with persistence, from the runs of setting L.

Colours: the project's fixed categorical palette (blue, orange, aqua, then greys), as in the Milestone 2 figures.

Requires: pip install pandas numpy matplotlib mlflow
"""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from analysis_helpers import AQUA, BLUE, INK_SECONDARY, ORANGE, get_pyplot, save_figure
from config import REPORTS_DIR, REPO_ROOT

DATA_DIR = REPORTS_DIR / "m3" / "figures_data"
GREY = "#9a9893"
FIGURES = ["learning", "variants", "hpo", "benchmark", "place"]
REFERENCE_RUNS = {"ridge": "ridge_A_standard_absolute", "xgboost": "xgboost_A_default"}     # the references of the variant comparison
ADOPTION_THRESHOLD = 1.0                                                                    # percent, the pre-stated 1% rule


def connect(store: Path) -> None:
    """Point MLflow at the store; stop if the file is missing."""
    if not store.exists():
        raise SystemExit(f"{store} not found. Run an experiment first.")
    mlflow.set_tracking_uri(f"sqlite:///{store.as_posix()}")


def get_runs(experiment: str) -> pd.DataFrame:
    """All runs of an experiment (oldest first), or an empty table if the experiment does not exist."""
    try:
        runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time ASC"])
    except mlflow.exceptions.MlflowException:
        return pd.DataFrame()
    return runs


def latest_per_name(runs: pd.DataFrame) -> pd.DataFrame:
    """One run per run name: the most recent."""
    return runs.drop_duplicates("tags.mlflow.runName", keep="last") if len(runs) else runs


def artifact_csv(run_id: str, name: str):
    """A CSV artefact of a run as a DataFrame, or None if the run has no such artefact."""
    try:
        return pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name))
    except Exception:
        return None


def save_data(name: str, frame: pd.DataFrame) -> None:
    """Save the numbers a figure shows."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_DIR / f"{name}.csv", index=False)


def fig_learning(plt, store) -> bool:
    """Learning curves: XGBoost by round for each fold, GRU by epoch for each seed."""
    features, neural = get_runs("m3_features"), get_runs("m3_neural")
    xgb = latest_per_name(features[features["tags.mlflow.runName"] == "xgboost_A_earlystop"]) if len(features) else features
    gru = neural[neural["tags.mlflow.runName"].str.contains("seed", na=False) & ~neural["tags.mlflow.runName"].str.contains("grid", na=False)] \
        if len(neural) else neural
    if xgb.empty and gru.empty:
        print("learning: no early-stopping XGBoost run (xgboost_A_earlystop) or GRU seed runs in the store; skipped.")
        return False
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    data = []
    folds = [("F_F2021", "F2021"), ("F_F2022", "F2022"), ("F_F2023", "F2023"), ("D_D", "D")]
    if len(xgb):
        run_id = xgb.iloc[0]["run_id"]
        for axis, (key, label) in zip(axes.flat, folds):
            curve = artifact_csv(run_id, f"learning_curve_{key}.csv")
            if curve is None:
                axis.set_visible(False)
                continue
            best = int(curve["inner_validation_mae"].idxmin())
            axis.plot(curve["round"], curve["train_mae"], color=BLUE, label="training (core rows)")
            axis.plot(curve["round"], curve["inner_validation_mae"], color=ORANGE, label="inner validation window")
            axis.axvline(curve["round"][best], color=INK_SECONDARY, linestyle=":", label=f"best round {int(curve['round'][best])}")
            axis.set(title=f"XGBoost, fold {label}", xlabel="boosting round", ylabel="MAE (ug/m3)")
            axis.legend()
            data.append(curve.assign(model="xgboost", fold=label))
    axis = axes[1, 1]
    axis.set(title="GRU, loss by epoch (standardised units)", xlabel="epoch", ylabel="loss")
    plotted_gru = False
    for i, (_, run) in enumerate(gru.iterrows()):
        for name in ("D", "F2023"):
            curve = artifact_csv(run["run_id"], f"learning_curve_{name}.csv")
            if curve is not None and curve["validation_loss"].notna().any():
                axis.plot(curve["epoch"], curve["train_loss"], color=BLUE, alpha=0.7, label="training" if i == 0 else None)
                axis.plot(curve["epoch"], curve["validation_loss"], color=ORANGE, alpha=0.7, label="inner validation" if i == 0 else None)
                data.append(curve.assign(model="gru", fold=name, run=run["tags.mlflow.runName"]))
                plotted_gru = True
                break
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend() if plotted_gru else axis.text(0.5, 0.5, "no GRU seed runs with validation loss", ha="center", transform=axis.transAxes)
    summary = axes[1, 2]
    summary.axis("off")
    if len(xgb):
        lines = ["chosen rounds: " + ", ".join(f"{label} {int(xgb.iloc[0].get(f'metrics.{key}_rounds', np.nan))}"
                                               for key, label in folds if pd.notna(xgb.iloc[0].get(f"metrics.{key}_rounds", np.nan)))]
        summary.text(0, 0.7, "\n".join(lines), fontsize=10, va="top")
    summary.text(0, 0.3, "Gap between training and validation = overfitting;\nboth high and flat = underfitting.\nThe dotted line is where early stopping stopped.",
                 fontsize=9, va="top", color=INK_SECONDARY)
    fig.suptitle("Learning curves (E6a)")
    fig.tight_layout()
    save_figure(fig, "m3_01_learning_curves.png")
    if data:
        save_data("m3_01_learning_curves", pd.concat(data, ignore_index=True))
    return True


def fig_variants(plt, store) -> bool:
    """Seen-station fold MAE and development MAE of every baseline / reference / variant run, against each model's reference."""
    frames = [get_runs(e) for e in ("m3_baselines", "m3_features")]
    runs = latest_per_name(pd.concat([f for f in frames if len(f)], ignore_index=True)) if any(len(f) for f in frames) else pd.DataFrame()
    if runs.empty:
        print("variants: no runs in m3_baselines / m3_features; skipped.")
        return False
    table = pd.DataFrame({"run": runs["tags.mlflow.runName"], "model": runs["params.model"], "F_mean_mae": runs["metrics.F_mean_mae_seen"],
                          "D_mae": runs["metrics.D_mae"], "D_skill": runs["metrics.D_skill"]})
    table["family"] = np.where(table["model"].isin(REFERENCE_RUNS), table["model"], "baseline")
    for column in ("F_mean_mae", "D_mae"):
        reference = table[table["run"].isin(REFERENCE_RUNS.values())].set_index("family")[column]
        table[f"{column}_vs_reference_pct"] = [100 * (v / reference[f] - 1) if f in reference.index else np.nan
                                               for v, f in zip(table[column], table["family"])]
    table = table.sort_values(["family", "F_mean_mae"], ascending=[True, False]).reset_index(drop=True)
    colors = {"baseline": GREY, "ridge": AQUA, "xgboost": BLUE}
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.32 * len(table) + 1.8)), sharey=True)
    for axis, column, title in ((axes[0], "F_mean_mae", "Selection folds (seen-station MAE, mean of 3)"), (axes[1], "D_mae", "Development year 2024 (MAE)")):
        # dots, not bars: the axis does not start at zero, and a truncated bar would exaggerate the differences
        axis.scatter(table[column], range(len(table)), color=[colors[f] for f in table["family"]], s=55, zorder=3)
        span = table[column].max() - table[column].min()
        axis.set(title=title, xlabel="MAE (ug/m3)", xlim=(table[column].min() - 0.06 * span, table[column].max() + 0.25 * span))
        axis.set_yticks(range(len(table)), table["run"])
        for i, row in table.iterrows():
            pct = row[f"{column}_vs_reference_pct"]
            if pd.notna(pct):
                axis.text(row[column], i, f"  {pct:+.1f}%", va="center", fontsize=8,
                          color="#b00020" if pct > ADOPTION_THRESHOLD else (AQUA if pct < -ADOPTION_THRESHOLD else INK_SECONDARY))
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c) for c in colors.values()]
    fig.legend(handles, ["baseline", "ridge", "xgboost"], loc="lower center", ncol=3)
    fig.suptitle(f"Baselines and variants (percent = change from the model's reference run; the adoption threshold is {ADOPTION_THRESHOLD:.0f}%)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, "m3_02_variants.png")
    save_data("m3_02_variants", table)
    return True


def fig_hpo(plt, store) -> bool:
    """Best-so-far curves, parameter importance, and the dev-year score of each study's best configuration."""
    runs = get_runs("m3_hpo")
    if runs.empty or "tags.kind" not in runs:
        print("hpo: no m3_hpo runs; skipped.")
        return False
    studies = runs[runs["tags.kind"] == "study"]
    defaults = runs[runs["tags.kind"] == "default"]
    summary = runs[runs["tags.kind"] == "summary"]
    curves = {}
    for _, run in studies.iterrows():
        trials = artifact_csv(run["run_id"], "trials.csv")
        if trials is not None:
            curves.setdefault(run["tags.sampler"], []).append(np.minimum.accumulate(trials["value"].to_numpy()))
    if not curves:
        print("hpo: no finished studies (trials.csv) yet; skipped.")
        return False
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {"random": BLUE, "tpe": ORANGE}
    data = []
    for sampler, lines in curves.items():
        length = min(len(line) for line in lines)
        matrix = np.array([line[:length] for line in lines])
        x = np.arange(1, length + 1)
        axes[0].plot(x, matrix.mean(axis=0), color=colors.get(sampler), label=f"{sampler} (mean of {len(lines)} seed{'s' if len(lines) > 1 else ''})")
        axes[0].fill_between(x, matrix.min(axis=0), matrix.max(axis=0), color=colors.get(sampler), alpha=0.2)
        data.append(pd.DataFrame({"sampler": sampler, "trial": x, "best_so_far_mean": matrix.mean(axis=0),
                                  "best_so_far_min": matrix.min(axis=0), "best_so_far_max": matrix.max(axis=0)}))
    if len(defaults):
        axes[0].axhline(defaults["metrics.best_objective"].mean(), color=GREY, linestyle="--", label="default configuration")
    axes[0].set(title="Best objective so far", xlabel="trial", ylabel="mean seen-station MAE over the 3 folds (ug/m3)")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend()
    importance = artifact_csv(summary.iloc[-1]["run_id"], "parameter_importance.csv") if len(summary) else None
    if importance is not None and len(importance):
        pivot = importance.groupby(["parameter", "sampler"])["importance"].mean().unstack().fillna(0)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        height = 0.38
        for k, sampler in enumerate(pivot.columns):
            axes[1].barh(np.arange(len(pivot)) + (k - 0.5) * height, pivot[sampler], height=height, color=colors.get(sampler), label=sampler)
        axes[1].set_yticks(range(len(pivot)), pivot.index)
        axes[1].set(title="Parameter importance (Optuna)", xlabel="importance (association within the sampled trials, not causal)")
        axes[1].legend()
        save_data("m3_03_hpo_importance", importance)
    else:
        axes[1].text(0.5, 0.5, "no importance table yet\n(logged by the summary run)", ha="center", transform=axes[1].transAxes)
    scored = pd.concat([studies[["tags.sampler", "tags.seed", "metrics.D_mae"]].rename(columns={"tags.sampler": "sampler", "tags.seed": "seed", "metrics.D_mae": "D_mae"}),
                        defaults[["tags.sampler", "tags.seed", "metrics.D_mae"]].rename(columns={"tags.sampler": "sampler", "tags.seed": "seed", "metrics.D_mae": "D_mae"})])
    for k, sampler in enumerate(["default", "random", "tpe"]):
        part = scored[scored["sampler"] == sampler]
        axes[2].scatter(np.full(len(part), k) + np.linspace(-0.1, 0.1, max(len(part), 1))[:len(part)], part["D_mae"],
                        color={"default": GREY, "random": BLUE, "tpe": ORANGE}[sampler])
    axes[2].set_xticks(range(3), ["default", "random", "tpe"])
    axes[2].set(title="Best configuration scored once on 2024", ylabel="development MAE (ug/m3)")
    fig.suptitle("Hyperparameter search (E7): random search vs TPE, same space, folds, metric and budget")
    fig.tight_layout()
    save_figure(fig, "m3_03_hpo.png")
    save_data("m3_03_hpo_curves", pd.concat(data, ignore_index=True))
    save_data("m3_03_hpo_dev_scores", scored)
    return True


def amdahl_curve(threads: np.ndarray, p: float) -> np.ndarray:
    """Speed-up predicted by Amdahl's law for a parallel fraction p."""
    return 1 / ((1 - p) + p / threads)


def fig_benchmark(plt, store) -> bool:
    """Thread scaling (speed-up and efficiency) and CPU vs GPU fit times, per machine."""
    runs = get_runs("m3_benchmark")
    if runs.empty:
        print("benchmark: no m3_benchmark runs; skipped.")
        return False
    repeats = []
    for _, run in runs.iterrows():
        frame = artifact_csv(run["run_id"], "benchmark_repeats.csv")
        if frame is not None:
            repeats.append(frame.assign(machine=run.get("tags.machine_type", "machine")))
    if not repeats:
        print("benchmark: no benchmark_repeats.csv artefacts; skipped.")
        return False
    data = pd.concat(repeats, ignore_index=True)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    palette = [BLUE, ORANGE, AQUA, GREY]
    machines = list(dict.fromkeys(data["machine"]))
    threads = data[data["test"] == "xgb_threads"].copy()
    if len(threads):
        threads["threads"] = threads["variant"].str.extract(r"threads_(\d+)", expand=False).astype(int)
        for m, machine in enumerate(machines):
            for config, part in threads[threads["machine"] == machine].groupby("config"):
                median = part.groupby("threads")["fit_seconds"].median()
                speedup = median.iloc[0] / median
                label = f"{machine}, {config}"
                axes[0].plot(median.index, speedup, marker="o", color=palette[m % 4], linestyle="-" if config == "default" else "--", label=label)
                axes[1].plot(median.index, speedup / (median.index / median.index[0]), marker="o", color=palette[m % 4],
                             linestyle="-" if config == "default" else "--", label=label)
                top = median.index.max()
                if top > median.index.min():
                    p = (1 - 1 / speedup[top]) / (1 - 1 / (top / median.index.min()))
                    axes[0].plot(median.index, amdahl_curve(median.index / median.index.min(), p), color=palette[m % 4], alpha=0.3, linewidth=1)
        top = threads["threads"].max()
        axes[0].plot([1, top], [1, top], color=INK_SECONDARY, linestyle=":", label="ideal")
        axes[0].set(title="XGBoost speed-up (faint line: Amdahl fit)", xlabel="CPU threads", ylabel="speed-up T1/TN")
        axes[0].legend(fontsize=7)
        axes[1].set(title="Parallel efficiency", xlabel="CPU threads", ylabel="speed-up / threads")
        for axis in axes[:2]:
            axis.set_xticks(sorted(threads["threads"].unique()))
    for axis, test, title in ((axes[2], "xgb_device", "XGBoost: CPU vs GPU"), (axes[3], "gru_device", "GRU: CPU vs GPU")):
        part = data[data["test"] == test]
        if part.empty:
            axis.text(0.5, 0.5, "no runs of this test", ha="center", transform=axis.transAxes)
            continue
        groups = part.groupby(["machine", "config", "variant"])["fit_seconds"].agg(["median", "min", "max"]).reset_index()
        labels = [(f"{r.machine}" + chr(10) if len(machines) > 1 else "") + f"{r.config}" + chr(10) + f"{r.variant}" for r in groups.itertuples()]
        axis.bar(range(len(groups)), groups["median"], color=[BLUE if v == "cpu" else ORANGE for v in groups["variant"]],
                 yerr=[groups["median"] - groups["min"], groups["max"] - groups["median"]], capsize=3)
        axis.set_xticks(range(len(groups)), labels, fontsize=8)
        axis.set(title=title + " (median, min-max)", ylabel="fit time (s)")
    fig.suptitle("Compute benchmark (E11)")
    fig.tight_layout()
    save_figure(fig, "m3_04_benchmark.png")
    save_data("m3_04_benchmark_repeats", data)
    return True


def fig_place(plt, store) -> bool:
    """Leave-one-city-out MAE per city for each place-feature variant, with persistence."""
    runs = get_runs("m3_place")
    if runs.empty or "metrics.L_pooled_mae" not in runs:
        print("place: no m3_place runs with setting L; skipped.")
        return False
    runs = runs[runs["metrics.L_pooled_mae"].notna()]
    metric_columns = [c for c in runs.columns if c.startswith("metrics.L_") and c.endswith("_mae_seen") and "persistence" not in c and "without" not in c]
    cities = sorted({c[len("metrics.L_"):-len("_mae_seen")] for c in metric_columns})
    rows = []
    for _, run in runs.iterrows():
        for city in cities:
            rows.append({"model": run["params.model"], "place": run["params.place"], "city": city, "mae": run[f"metrics.L_{city}_mae_seen"],
                         "persistence_mae": run.get(f"metrics.L_{city}_persistence_mae_seen", np.nan)})
    table = pd.DataFrame(rows)
    models = sorted(table["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 4.8), squeeze=False)
    variants = ["none", "latlon", "country", "all"]
    colors = dict(zip(variants, [GREY, BLUE, ORANGE, AQUA]))
    for axis, model in zip(axes[0], models):
        part = table[table["model"] == model]
        width = 0.2
        for k, variant in enumerate(variants):
            sub = part[part["place"] == variant].set_index("city").reindex(cities)
            axis.bar(np.arange(len(cities)) + (k - 1.5) * width, sub["mae"], width=width, color=colors[variant], label=f"place: {variant}")
        persistence = part.groupby("city")["persistence_mae"].first().reindex(cities)
        axis.scatter(np.arange(len(cities)), persistence, color="black", marker="_", s=300, zorder=3, label="persistence")
        axis.set_xticks(range(len(cities)), cities, rotation=30)
        axis.set(title=f"{model}: leave-one-city-out, validated in 2023", ylabel="MAE (ug/m3)")
        axis.legend(fontsize=8)
    fig.suptitle("Place ablation (E8): country is unseen for Athens, Belfast, Copenhagen and Madrid")
    fig.tight_layout()
    save_figure(fig, "m3_05_place.png")
    save_data("m3_05_place", table)
    return True


def main():
    """Draw the requested figures from the MLflow store."""
    parser = argparse.ArgumentParser(description="Draw the Milestone 3 figures from MLflow.")
    parser.add_argument("--figure", nargs="+", default=["all"], choices=FIGURES + ["all"])
    parser.add_argument("--store", type=Path, default=REPO_ROOT / "mlruns" / "mlflow.db")
    args = parser.parse_args()
    connect(args.store)
    plt = get_pyplot()
    wanted = FIGURES if "all" in args.figure else args.figure
    functions = {"learning": fig_learning, "variants": fig_variants, "hpo": fig_hpo, "benchmark": fig_benchmark, "place": fig_place}
    drawn = [name for name in wanted if functions[name](plt, args.store)]
    print(f"\nDrawn: {drawn or 'nothing'}; skipped: {[n for n in wanted if n not in drawn]}")


if __name__ == "__main__":
    main()
