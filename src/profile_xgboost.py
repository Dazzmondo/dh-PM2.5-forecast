"""
Profile one fit of the selected XGBoost model (Milestone 3 bonus): which internal operations take the time?

USAGE:  python src/profile_xgboost.py [--model-dir models/xgboost/v3] [--repeats 3]
INPUT:  the saved model folder (config.yaml, metrics.json and model.json, for the settings and the number of boosting rounds), and via
        m3_data.py the train and dev rows (the test split and held-out stations are never read)
OUTPUT: reports/m3/profiling/xgboost_cpu/   runs.csv (every run with XGBoost's timers), breakdown.csv (the parts of one fit),
        profile.png, timer_log.txt (the raw output of every run) and lineage.json (code, data, library and machine identifiers).
        Nothing is saved as a model and nothing is logged to MLflow.

How: XGBoost trains inside its compiled library, so a Python profiler cannot see its internal operations. With verbosity=3 the library
prints its own timers when a fitted model is freed. Each fit therefore runs in a separate process whose output is captured and parsed.
The fit is the saved model's: the same settings and number of rounds on the 2018-2023 training rows. Its 2024 error is checked against
the saved model's, so that the profiled workload is known to be the right one.

Settings compared (the model's objective is absolute error):
  all threads, with timers      the breakdown
  one thread, with timers       which parts speed up with more threads
  all threads, squared error    what the absolute-error objective costs (same settings otherwise; not a candidate model)
  all threads, without timers   how long the fit takes when nothing is printed

The parts of a fit are XGBoost's timers where it has them. Two parts are differences: the time outside `BoostNewTrees` (the
tree-building step) and the time inside it but outside `UpdateTree` (the tree updater).

Close other programs before running: the timings depend on what else the machine is doing.

Requires: pip install pandas pyarrow numpy xgboost matplotlib pyyaml
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from xgboost import Booster, XGBRegressor

from analysis_helpers import AQUA, BLUE, get_pyplot
from config import REPORTS_DIR
from m3_data import build_features, load_rows
from m3_tracking import lineage_tags

OUT_DIR = REPORTS_DIR / "m3" / "profiling" / "xgboost_cpu"
VERBOSE = 3                                  # XGBoost prints its timers at this level
ABSOLUTE, SQUARED = "reg:absoluteerror", "reg:squarederror"
TIMER_LINE = re.compile(r"\] (\w+): ([0-9.e+-]+)s, (\d+) calls")
NEEDED = ["BoostNewTrees", "UpdateTree", "InitRoot", "BuildHistogram", "UpdatePosition", "LeafPartition", "EvaluateSplits"]
# the parts of one fit, in the order they are tabulated and drawn
PARTS = ["Outside the tree-building step (no timer)", "Tree-building step, outside the tree updater (no timer)",
         "Building histograms (InitRoot, BuildHistogram)", "Moving rows to child nodes (UpdatePosition)",
         "Assigning rows to leaves (LeafPartition)", "Choosing splits (EvaluateSplits)", "Set-up and the rest of the updater"]


def one_fit(model_dir: Path, threads: int, verbosity: int, objective: str):
    """Child process: fit the saved model's settings once, time it, and print one RESULT line (XGBoost prints its timers after it)."""
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
    saved = Booster()
    saved.load_model(model_dir / "model.json")
    rounds = saved.num_boosted_rounds()                       # the rounds the saved model really has (chosen by early stopping)

    t0 = time.perf_counter()
    rows = load_rows(("train", "dev"))
    train, dev = rows[rows["split"] == "train"], rows[rows["split"] == "dev"]
    X_train, X_dev = (build_features(part, cfg["feature_set"], cfg["time_features"]) for part in (train, dev))
    load_seconds = time.perf_counter() - t0

    # the rounds are fixed, so the early-stopping settings of the configuration are left out
    params = {k: v for k, v in cfg["params"].items() if k not in ("n_estimators", "early_stopping_rounds")}
    params["objective"] = objective
    model = XGBRegressor(random_state=cfg["seed"], n_jobs=threads, n_estimators=rounds, verbosity=verbosity, **params)
    t0 = time.perf_counter()
    model.fit(X_train, train["target_pm25"])
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    forecast = model.predict(X_dev)
    predict_seconds = time.perf_counter() - t0

    result = {"threads": threads, "verbosity": verbosity, "objective": objective, "rounds": rounds, "train_rows": len(train),
              "load_seconds": load_seconds, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
              "dev_mae": float(np.abs(forecast - dev["target_pm25"].to_numpy()).mean())}
    print("RESULT " + json.dumps(result), flush=True)
    del model                                                  # freeing the model makes XGBoost print its timer totals


def parse_output(text: str) -> dict:
    """The RESULT line and the timer totals (seconds) in one child's output."""
    found = re.search(r"^RESULT (\{.*\})$", text, flags=re.MULTILINE)
    if not found:
        raise SystemExit(f"STOPPED: a fit printed no result. Its output ends:\n{text[-2000:]}")
    record = json.loads(found.group(1))
    record.update({name: float(seconds) for name, seconds, _ in TIMER_LINE.findall(text)})
    missing = [name for name in NEEDED if name not in record]
    if record["verbosity"] >= VERBOSE and missing:
        raise SystemExit(f"STOPPED: XGBoost did not print these timers: {', '.join(missing)}. The library version may have changed them.")
    return record


def split_fit(record: pd.Series) -> list:
    """The seconds of each of PARTS in one timed fit; they add up to the fit time."""
    parts = [record["fit_seconds"] - record["BoostNewTrees"],
             record["BoostNewTrees"] - record["UpdateTree"],
             record["InitRoot"] + record["BuildHistogram"],
             record["UpdatePosition"], record["LeafPartition"], record["EvaluateSplits"]]
    return parts + [record["fit_seconds"] - sum(parts)]


def median_run(runs: pd.DataFrame, setting: str) -> pd.Series:
    """The run of a setting whose fit time is the median, so that its parts belong to one real run."""
    chosen = runs[runs["setting"] == setting].sort_values("fit_seconds")
    return chosen.iloc[len(chosen) // 2]


def draw(breakdown: pd.DataFrame, labels: dict, path: Path):
    """Horizontal bars: the seconds of each part with one thread and with all threads."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    positions = np.arange(len(breakdown))
    for offset, column, colour in ((-0.2, "one_thread_seconds", AQUA), (0.2, "all_threads_seconds", BLUE)):
        bars = ax.barh(positions + offset, breakdown[column], height=0.38, label=labels[column], color=colour)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_yticks(positions, breakdown["part"])
    ax.invert_yaxis()
    ax.set_xlabel("seconds in one fit")
    ax.set_title("Where one XGBoost fit spends its time (CPU)", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    """Run every setting in turn, repeat, then write the runs, the breakdown of the median runs, the figure and the lineage."""
    parser = argparse.ArgumentParser(description="Profile one fit of the saved XGBoost model with XGBoost's own timers.")
    parser.add_argument("--model-dir", default=Path("models/xgboost/v3"), type=Path)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--child", nargs=3, metavar=("THREADS", "VERBOSITY", "OBJECTIVE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        one_fit(args.model_dir, int(args.child[0]), int(args.child[1]), args.child[2])
        return
    if args.repeats < 1 or args.repeats % 2 == 0:
        raise SystemExit("STOPPED: --repeats must be an odd number, so that there is a median run.")

    threads = os.cpu_count()
    settings = {"all_threads": (threads, VERBOSE, ABSOLUTE), "one_thread": (1, VERBOSE, ABSOLUTE),
                "all_threads_squared_error": (threads, VERBOSE, SQUARED), "all_threads_no_timers": (threads, 0, ABSOLUTE)}
    saved_mae = json.loads((args.model_dir / "metrics.json").read_text(encoding="utf-8"))["D_mae"]
    records, log = [], []
    for repeat in range(1, args.repeats + 1):                  # settings are interleaved, so slow drift touches all of them
        for setting, (n_threads, verbosity, objective) in settings.items():
            command = [sys.executable, "-u", __file__, "--model-dir", str(args.model_dir),
                       "--child", str(n_threads), str(verbosity), objective]
            done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                       "MLFLOW_DISABLE_AGENT_HINT": "1"})    # keeps a library notice out of the log
            output = done.stdout + done.stderr
            if done.returncode != 0:
                raise SystemExit(f"STOPPED: the fit '{setting}' failed:\n{output[-2000:]}")
            log.append(f"===== {setting}, repeat {repeat} =====\n{output}")
            record = {"setting": setting, "repeat": repeat, **parse_output(output)}
            records.append(record)
            print(f"repeat {repeat}  {setting:26s} fit {record['fit_seconds']:6.1f} s   2024 MAE {record['dev_mae']:.4f}", flush=True)

    runs = pd.DataFrame(records)
    check = runs[runs["setting"].isin(["all_threads", "all_threads_no_timers"])]["dev_mae"]
    if (check - saved_mae).abs().max() > 0.003:                # the reproduction tolerance of docs/M3.md section 7
        raise SystemExit(f"STOPPED: the profiled fit does not reproduce the saved model (2024 MAE {check.iloc[0]:.4f} against {saved_mae:.4f}).")

    one, every = median_run(runs, "one_thread"), median_run(runs, "all_threads")
    breakdown = pd.DataFrame({"part": PARTS, "one_thread_seconds": split_fit(one), "all_threads_seconds": split_fit(every)})
    breakdown["speed_up"] = breakdown["one_thread_seconds"] / breakdown["all_threads_seconds"]
    breakdown["share_at_all_threads"] = breakdown["all_threads_seconds"] / every["fit_seconds"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT_DIR / "runs.csv", index=False)
    breakdown.to_csv(OUT_DIR / "breakdown.csv", index=False)
    (OUT_DIR / "timer_log.txt").write_text("\n".join(log), encoding="utf-8")
    (OUT_DIR / "lineage.json").write_text(json.dumps(lineage_tags(), indent=2), encoding="utf-8")
    draw(breakdown, {"one_thread_seconds": f"1 thread ({one['fit_seconds']:.1f} s)",
                     "all_threads_seconds": f"{threads} threads ({every['fit_seconds']:.1f} s)"}, OUT_DIR / "profile.png")

    print("\nParts of the median fit (seconds):")
    print(breakdown.round(2).to_string(index=False))
    print("\nFit seconds by setting (min / median / max):")
    print(runs.groupby("setting")[["fit_seconds", "predict_seconds"]].agg(["min", "median", "max"]).round(2).to_string())
    print(f"\nResults written to {OUT_DIR}")


if __name__ == "__main__":
    main()
