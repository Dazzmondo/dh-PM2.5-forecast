"""
Compute benchmark (Milestone 3): how long do XGBoost and the GRU take on this machine, on how many CPU threads, on CPU or GPU?

USAGE:  python src/benchmark.py --config configs/compute_benchmark.yaml [--only xgb_threads xgb_device gru_device] [--no-tracking]
                                [--machine-type n1-standard-8] [--region europe-west4] [--physical-cores 4] [--hourly-rate 0.38] [--cpu-only-rate 0.38]
INPUT:  the configuration, and via m3_data.py the train and dev rows (the test split and held-out stations are never read)
OUTPUT: one MLflow run in experiment `m3_benchmark` (every repeat as a metric step, the summary tables as artefacts, the machine and
        image as tags) and the summary tables printed. Run it on each machine being compared, from the same image.

Three tests (each: a warm-up run that is discarded, then `repeats` timed runs; every run fits the model on the whole training split,
2018-2023, and predicts the 31,192 development rows; no early stopping, so every run is exactly as long as the others):
  xgb_threads   XGBoost on the CPU with 1, 2, 4, 8 ... threads (the list in the configuration). Reported: median, min, max and std of
                the fit time, the speed-up S(N) = T1 / TN, the parallel efficiency S(N) / N, and the parallel fraction p that
                Amdahl's law implies, S(N) = 1 / ((1 - p) + p / N), from the largest N. Cloud vCPUs are usually hyperthreads, so a curve
                that flattens beyond the physical core count is expected.
  xgb_device    XGBoost with device="cpu" and device="cuda" (skipped, and said so, when no CUDA device or CUDA build is available).
  gru_device    the GRU for a fixed number of epochs and a fixed batch size on the CPU (explicit torch thread count) and on the GPU.
Each XGBoost test is run for every named configuration in the file (for example a small default and a larger one: a flat curve on a
small workload is a valid result, and a larger configuration shows whether the flatness is the workload's or the machine's).

Always recorded, as separate numbers: data loading, fitting, prediction and total time; inference time per 1,000 rows; peak process
memory (the whole-process peak so far, so it only grows), GPU memory where it can be read; model size. With --hourly-rate (the machine's
price per hour in your billing currency, from the current price list) the approximate cost of one fit and of the whole benchmark.
Everything on the machine is also tagged: CPU, logical and (with --physical-cores) physical cores, GPU, CUDA, library versions, image.

Requires: pip install pandas pyarrow numpy scikit-learn xgboost torch mlflow pyyaml
"""
import argparse
import json
import pickle
import subprocess
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from xgboost import XGBRegressor

from m3_data import build_features, expanding_folds, load_rows, rows_hash, time_features
from m3_tracking import lineage_tags, peak_ram_mb, pip_freeze_text, setup_experiment
from train_gru import build_sequences, predict, sync, to_tensors, train_network

TESTS = ["xgb_threads", "xgb_device", "gru_device"]
DEFAULTS = {"experiment": "m3_benchmark", "feature_set": "A", "time_features": "cyclical", "repeats": 5, "seed": 0,
            "threads": [1, 2, 4, 8],
            "xgb_configs": {"default": {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 6},
                            "large": {"n_estimators": 2000, "learning_rate": 0.05, "max_depth": 10}},
            "gru": {"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "lr": 0.003, "batch_size": 256},
            "gru_epochs": 10, "gru_repeats": 3, "gru_cpu_threads": 4, "loss": "mse", "max_epochs": 50, "patience": 7}


def gpu_memory_used_mb():
    """GPU memory in use according to nvidia-smi, in MB; None when there is no NVIDIA tool or GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        return float(out.stdout.strip().splitlines()[0]) if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def device_used(model: XGBRegressor) -> str:
    """The device a fitted XGBoost model really used ('cpu' or 'cuda:0'). XGBoost silently falls back to the CPU, with only a
    warning, when device='cuda' is requested on a machine without a GPU, so the requested device proves nothing."""
    return json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]


def xgboost_cuda_works() -> bool:
    """True only if a tiny fit with device='cuda' really runs on a CUDA device (a CUDA build of XGBoost and a GPU are both needed)."""
    try:
        model = XGBRegressor(tree_method="hist", device="cuda", n_estimators=2)
        model.fit(np.random.rand(50, 3), np.random.rand(50))
        return device_used(model).startswith("cuda")
    except Exception:
        return False


def xgb_run(params: dict, device: str, threads, X_fit, y_fit, X_val, seed: int) -> dict:
    """One timed XGBoost fit and prediction; returns the timings, model size and memory readings."""
    model = XGBRegressor(tree_method="hist", device=device, n_jobs=threads, random_state=seed, **params)
    t0 = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.predict(X_val)
    predict_seconds = time.perf_counter() - t0
    used = device_used(model)
    if not used.startswith(device):                         # never record a CPU run as a GPU run (or the reverse)
        raise SystemExit(f"STOPPED: XGBoost was asked for device={device!r} but ran on {used!r}.")
    return {"fit_seconds": fit, "predict_seconds": predict_seconds, "inference_ms_per_1000_rows": 1e6 * predict_seconds / len(X_val),
            "model_size_bytes": len(pickle.dumps(model)), "gpu_used_mb": gpu_memory_used_mb() if device == "cuda" else None}


def gru_run(hp: dict, cfg: dict, device: torch.device, train, val_seq, val_sta, seed: int) -> dict:
    """One timed GRU fit (a fixed number of epochs) and prediction on the device."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    t0 = time.perf_counter()
    model, _, _ = train_network(hp, cfg, seed, device, train, epochs=cfg["gru_epochs"])
    sync(device)
    fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    predict(model, val_seq, val_sta)
    sync(device)
    predict_seconds = time.perf_counter() - t0
    return {"fit_seconds": fit, "predict_seconds": predict_seconds, "inference_ms_per_1000_rows": 1e6 * predict_seconds / len(val_seq),
            "model_size_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "gpu_used_mb": torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None}


def repeat(function, repeats: int, label: str) -> list:
    """Run `function` once as a discarded warm-up, then `repeats` times; return the timed results."""
    function()                                              # warm-up: caches, thread pools and GPU context start-up are not measured
    results = []
    for i in range(repeats):
        results.append(function())
        print(f"    {label}: repeat {i + 1}/{repeats}  fit {results[-1]['fit_seconds']:.2f}s  predict {results[-1]['predict_seconds']:.3f}s")
    return results


def summarise(records: list) -> pd.DataFrame:
    """Median, min, max and std of the fit time per (test, configuration, variant), plus speed-up and efficiency for thread tests."""
    frame = pd.DataFrame(records)
    group = ["test", "config", "variant"]
    table = frame.groupby(group, sort=False).agg(
        repeats=("fit_seconds", "size"), fit_median=("fit_seconds", "median"), fit_min=("fit_seconds", "min"),
        fit_max=("fit_seconds", "max"), fit_std=("fit_seconds", "std"), predict_median=("predict_seconds", "median"),
        inference_ms_per_1000_rows=("inference_ms_per_1000_rows", "median"), model_size_bytes=("model_size_bytes", "median"),
        gpu_used_mb=("gpu_used_mb", "max")).reset_index()
    table["threads"] = table["variant"].str.extract(r"threads_(\d+)", expand=False).astype(float)
    for (test, config), part in table[table["test"] == "xgb_threads"].groupby(["test", "config"]):
        base = part.loc[part["threads"].idxmin()]
        table.loc[part.index, "speedup"] = base["fit_median"] / part["fit_median"]
        table.loc[part.index, "efficiency"] = table.loc[part.index, "speedup"] / (part["threads"] / base["threads"])
    return table


def amdahl_fraction(table: pd.DataFrame) -> dict:
    """The parallel fraction p implied by the speed-up at the largest thread count, S = 1 / ((1 - p) + p / N), per configuration."""
    fractions = {}
    for config, part in table[table["test"] == "xgb_threads"].groupby("config"):
        top = part.loc[part["threads"].idxmax()]
        if top["threads"] > 1 and top["speedup"] > 0:
            fractions[config] = float((1 - 1 / top["speedup"]) / (1 - 1 / top["threads"]))
    return fractions


def main():
    """Run the chosen tests, print the summary tables and log the benchmark to MLflow."""
    parser = argparse.ArgumentParser(description="Compute benchmark: CPU thread scaling and CPU vs GPU.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--only", nargs="+", choices=TESTS, help="run only these tests (default: all)")
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument("--machine-type", default="local", help="e.g. n1-standard-8 (tagged on the run)")
    parser.add_argument("--region", default="", help="cloud region (tagged on the run)")
    parser.add_argument("--physical-cores", type=int, help="physical CPU cores (the logical count alone cannot show hyperthreading)")
    parser.add_argument("--hourly-rate", type=float, help="price of this machine per hour (the machine actually used, GPU included); adds the "
                        "approximate cost of a fit and of the run")
    parser.add_argument("--cpu-only-rate", type=float, help="price per hour of the same machine WITHOUT a GPU; the cost of CPU fits is then "
                        "taken at this rate (a CPU run would not need the GPU machine), and GPU fits at --hourly-rate")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **yaml.safe_load(args.config.read_text(encoding="utf-8"))}
    cfg["gru_hp"] = {**DEFAULTS["gru"], **cfg.get("gru", {})}
    tests = args.only or TESTS

    t_start = time.perf_counter()
    rows = load_rows()
    fold = [f for f in expanding_folds(rows) if f.name == "D"][0]
    X = build_features(rows, cfg["feature_set"], cfg["time_features"])
    y = rows["target_pm25"].to_numpy(dtype=float)
    X_fit, y_fit, X_val = X[fold.fit], y[fold.fit], X[fold.val]            # cut once, so no repeat pays for the slicing
    load_seconds = time.perf_counter() - t_start
    print(f"Benchmark on {args.machine_type}: {int(fold.fit.sum()):,} training rows, {int(fold.val.sum()):,} development rows, data load "
          f"{load_seconds:.2f}s; torch threads by default: {torch.get_num_threads()}; CUDA available to PyTorch: {torch.cuda.is_available()}")

    records, skipped = [], []
    if "xgb_threads" in tests:
        for name, params in cfg["xgb_configs"].items():
            for threads in cfg["threads"]:
                print(f"  xgb_threads / {name} / {threads} thread(s)")
                for r in repeat(lambda: xgb_run(params, "cpu", threads, X_fit, y_fit, X_val, cfg["seed"]), cfg["repeats"],
                                f"{name} {threads}t"):
                    records.append({"test": "xgb_threads", "config": name, "variant": f"threads_{threads}", **r})
    if "xgb_device" in tests:
        for name, params in cfg["xgb_configs"].items():
            for device in ("cpu", "cuda"):
                if device == "cuda" and not xgboost_cuda_works():
                    skipped.append("xgb_device / cuda: no CUDA GPU or no CUDA build of XGBoost on this machine")
                    continue
                print(f"  xgb_device / {name} / {device}")
                for r in repeat(lambda: xgb_run(params, device, None, X_fit, y_fit, X_val, cfg["seed"]), cfg["repeats"], f"{name} {device}"):
                    records.append({"test": "xgb_device", "config": name, "variant": device, **r})
    if "gru_device" in tests:
        sequence = build_sequences(rows)
        static = time_features(rows, cfg["time_features"]).to_numpy(dtype=float)
        fit, val = fold.fit, fold.val
        stats = {"seq_mean": float(sequence[fit].mean()), "seq_std": float(sequence[fit].std()),
                 "target_mean": float(y[fit].mean()), "target_std": float(y[fit].std())}
        for device_name in ("cpu", "cuda"):
            if device_name == "cuda" and not torch.cuda.is_available():
                skipped.append("gru_device / cuda: no CUDA GPU available to PyTorch on this machine")
                continue
            device = torch.device(device_name)
            torch.set_num_threads(int(cfg["gru_cpu_threads"]))
            train = to_tensors(sequence[fit], static[fit], y[fit], stats, device)
            val_seq, val_sta, _ = to_tensors(sequence[val], static[val], None, stats, device)
            print(f"  gru_device / {device_name} ({cfg['gru_epochs']} epochs, batch {cfg['gru_hp']['batch_size']}, "
                  f"{cfg['gru_cpu_threads']} torch threads on CPU)")
            for r in repeat(lambda: gru_run(cfg["gru_hp"], cfg, device, train, val_seq, val_sta, cfg["seed"]), cfg["gru_repeats"], f"gru {device_name}"):
                records.append({"test": "gru_device", "config": f"{cfg['gru_epochs']}_epochs", "variant": device_name, **r})
    if not records:
        raise SystemExit("Nothing was benchmarked: " + "; ".join(skipped))

    table = summarise(records)
    total_seconds = time.perf_counter() - t_start
    if args.hourly_rate:
        cpu_rate = args.cpu_only_rate or args.hourly_rate
        is_gpu = table["variant"].isin(["cuda"])
        table["approx_cost_per_fit"] = np.where(is_gpu, args.hourly_rate, cpu_rate) * table["fit_median"] / 3600
    amdahl = amdahl_fraction(table)
    print("\nSummary (fit times in seconds; speed-up and efficiency relative to the fewest threads):")
    print(table.drop(columns=["model_size_bytes"]).round(3).to_string(index=False))
    for config, p in amdahl.items():
        print(f"Amdahl: configuration {config}: parallel fraction p = {p:.2f} implied by the largest thread count")
    for line in skipped:
        print("SKIPPED:", line)
    print(f"Benchmark wall time {total_seconds / 60:.1f} min" + (f", approximate cost {args.hourly_rate * total_seconds / 3600:.3f}" if args.hourly_rate else ""))
    if args.no_tracking:
        return

    setup_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=f"benchmark_{args.machine_type}"):
        tags = {**lineage_tags(), "machine_type": args.machine_type, "region": args.region, "tests": ",".join(tests),
                "skipped": "; ".join(skipped), "evaluation_rows_sha256": rows_hash(rows[rows["split"] == "dev"])}
        if args.physical_cores:
            tags["machine_physical_cores"] = str(args.physical_cores)
        mlflow.set_tags(tags)
        mlflow.log_params({k: str(v) for k, v in cfg.items() if k not in ("gru",)})
        for (test, config, variant), group in pd.DataFrame(records).groupby(["test", "config", "variant"], sort=False):
            for step, value in enumerate(group["fit_seconds"]):
                mlflow.log_metric(f"{test}.{config}.{variant}.fit_seconds", float(value), step=step)
        for row in table.itertuples():
            prefix = f"{row.test}.{row.config}.{row.variant}"
            mlflow.log_metrics({f"{prefix}.fit_median": row.fit_median, f"{prefix}.inference_ms_per_1000_rows": row.inference_ms_per_1000_rows,
                                f"{prefix}.model_size_bytes": row.model_size_bytes})
            if row.test == "xgb_threads":
                mlflow.log_metrics({f"{prefix}.speedup": row.speedup, f"{prefix}.efficiency": row.efficiency})
            if pd.notna(row.gpu_used_mb):
                mlflow.log_metric(f"{prefix}.gpu_used_mb", row.gpu_used_mb)
        mlflow.log_metrics({f"amdahl_parallel_fraction.{k}": v for k, v in amdahl.items()})
        mlflow.log_metric("data_load_seconds", load_seconds)
        mlflow.log_metric("benchmark_wall_seconds", total_seconds)
        peak = peak_ram_mb()
        if peak is not None:
            mlflow.log_metric("peak_ram_mb", peak)
        if args.hourly_rate:
            mlflow.log_metric("approx_cost_of_benchmark", args.hourly_rate * total_seconds / 3600)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pd.DataFrame(records).to_csv(tmp / "benchmark_repeats.csv", index=False)
            table.to_csv(tmp / "benchmark_summary.csv", index=False)
            (tmp / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            (tmp / "pip_freeze.txt").write_text(pip_freeze_text(), encoding="utf-8")
            mlflow.log_artifacts(str(tmp))
    print("Logged the benchmark run in experiment m3_benchmark.")


if __name__ == "__main__":
    main()
