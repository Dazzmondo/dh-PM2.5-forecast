"""
Train and score the GRU on the ordered 8-block window (Milestone 3), logging everything to MLflow.

USAGE:  python src/train_gru.py --config configs/<experiment>.yaml [--save-model] [--no-tracking]
INPUT:  the configuration, and via m3_data.py the train and dev rows (the test split and held-out stations are never read)
OUTPUT: MLflow runs in mlruns/ (per-epoch training and validation loss as metric steps, metrics, timings, per-station tables,
        lineage tags); with --save-model, models/pytorch_gru/v<N>/ (model.pt, config.yaml, preprocessing.json, metrics.json,
        metadata.json), one version per seed.

What the model sees: the 8 block means t-7 ... t of one station (oldest first, one value per step), standardised with a mean and
standard deviation fitted on the fold's training rows, plus the time features only (sin/cos of local hour and month), joined to
the last hidden state. It does NOT see city, country or position (Milestone 2 keeps place out of the baseline). The target is the
standardised mean of block t+2; forecasts are turned back into ug/m3 and scored on the raw scale with the same code as train.py.

Window builder: build_sequences() takes the 8 columns from m3_data (rebuilt from the gap-free block table and already verified
against the stored lags and rolling statistics) and stops the run if the order, the last value or the shape is wrong. The window of
every row ends at its forecast origin, never crosses a station, and has no missing block (a missing block would have removed the row).

Training (per fold): phase 1 trains on the core rows with early stopping on an inner window (the last 12 months of the fold's
training rows by target time, patience `patience`, at most `max_epochs` epochs, Adam with ReduceLROnPlateau); phase 2 refits from
the same seed on ALL the fold's training rows for the epoch count phase 1 chose. With `fixed_epochs: N` there is no early stopping:
every fit trains exactly N epochs (used for the CPU vs GPU timing, so run lengths match).

Two modes, chosen by the configuration:
  tuning  (`grid:` present)  a small grid on ONE fold (train 2018-2022, validate 2023), one seed, at most 12 configurations;
                             `loss` (mse or l1) may be a grid axis, like the hyperparameters
  final   (`seeds:` list)    the chosen hyperparameters on the settings listed (F, D), one MLflow run per seed, plus a summary run

Requires: pip install pandas pyarrow numpy torch mlflow pyyaml
"""
import argparse
import copy
import itertools
import json
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from m3_data import expanding_folds, inner_early_stopping_split, load_rows, rows_hash, time_features
from m3_tracking import lineage_tags, peak_ram_mb, pip_freeze_text, setup_experiment
from train import MODELS_DIR, check_persistence_benchmark, next_version, score_fold, summarise

WINDOW_COLUMNS = [f"pm25_lag_{k}" for k in range(7, 0, -1)] + ["pm25_current"]      # t-7 ... t, oldest first
HP_DEFAULTS = {"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "lr": 0.003, "batch_size": 256}
CFG_DEFAULTS = {"settings": ["D"], "seeds": [0], "time_features": "cyclical", "loss": "mse", "max_epochs": 50, "patience": 7,
                "fixed_epochs": None, "device": "auto", "threads": None, "hp": {}}
MAX_GRID = 12
PREDICT_BATCH = 4096


class GRUForecaster(nn.Module):
    """A GRU over the 8-block window; its last hidden state is joined with the time features and mapped to one forecast."""

    def __init__(self, n_static: int, hidden_size: int, num_layers: int, dropout: float):
        """Build the recurrent layer and the small output network."""
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size + n_static, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Forecast (standardised) for a batch: sequence (batch, 8, 1) and static features (batch, n_static)."""
        _, hidden = self.gru(sequence)
        return self.head(torch.cat([self.drop(hidden[-1]), static], dim=1)).squeeze(1)


def build_sequences(rows: pd.DataFrame) -> np.ndarray:
    """The (rows, 8) window array, oldest block first; stops the run if the order or the last value is wrong."""
    sequence = rows[WINDOW_COLUMNS].to_numpy(dtype=float)
    if sequence.shape[1] != 8 or np.isnan(sequence).any():
        raise SystemExit("WINDOW CHECK FAILED: a window is not 8 complete blocks.")
    if not np.array_equal(sequence[:, -1], rows["pm25_current"].to_numpy()):
        raise SystemExit("WINDOW CHECK FAILED: the window does not end at the current block (the forecast origin).")
    for k in (1, 2, 3):
        if not np.array_equal(sequence[:, -1 - k], rows[f"pm25_lag_{k}"].to_numpy()):
            raise SystemExit(f"WINDOW CHECK FAILED: window position t-{k} is not pm25_lag_{k}.")
    return sequence


def choose_device(name: str) -> torch.device:
    """cpu, cuda, or auto (cuda when available)."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("device: cuda was requested but no CUDA GPU is available.")
    return torch.device(name)


def sync(device: torch.device) -> None:
    """Wait for queued GPU work so that timings are real (a no-op on the CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def to_tensors(sequence, static, target, stats, device):
    """Standardise with the fold's statistics and move to tensors on the device (the target may be None)."""
    seq = torch.tensor(((sequence - stats["seq_mean"]) / stats["seq_std"]), dtype=torch.float32, device=device).unsqueeze(2)
    sta = torch.tensor(static, dtype=torch.float32, device=device)
    tgt = None if target is None else torch.tensor((target - stats["target_mean"]) / stats["target_std"], dtype=torch.float32, device=device)
    return seq, sta, tgt


def train_network(hp: dict, cfg: dict, seed: int, device: torch.device, train, validation=None, epochs=None):
    """Train a GRU. With `validation`: early stopping (best state kept). With `epochs`: exactly that many epochs, no validation.

    Returns (model, history DataFrame with epoch, train_loss, validation_loss, lr, and best_epoch).
    """
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = GRUForecaster(train[1].shape[1], hp["hidden_size"], hp["num_layers"], hp["dropout"]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=2)
    loss_fn = nn.MSELoss() if cfg["loss"] == "mse" else nn.L1Loss()
    seq, sta, tgt = train
    n = len(tgt)
    max_epochs = epochs if epochs is not None else cfg["max_epochs"]
    history, best_loss, best_state, best_epoch, waited = [], np.inf, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.tensor(generator.permutation(n), device=device)
        total = 0.0
        for start in range(0, n, hp["batch_size"]):
            index = order[start:start + hp["batch_size"]]
            optimiser.zero_grad()
            loss = loss_fn(model(seq[index], sta[index]), tgt[index])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(index)
        record = {"epoch": epoch, "train_loss": total / n, "lr": optimiser.param_groups[0]["lr"], "validation_loss": np.nan}
        if validation is not None:
            model.eval()
            with torch.no_grad():
                record["validation_loss"] = loss_fn(model(validation[0], validation[1]), validation[2]).item()
            scheduler.step(record["validation_loss"])
            if record["validation_loss"] < best_loss - 1e-6:
                best_loss, best_epoch, waited = record["validation_loss"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
            history.append(record)
            if waited >= cfg["patience"]:
                break
        else:
            history.append(record)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), (best_epoch if validation is not None else max_epochs)


def predict(model: nn.Module, seq: torch.Tensor, sta: torch.Tensor) -> np.ndarray:
    """Standardised forecasts for all rows, in batches."""
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(seq[i:i + PREDICT_BATCH], sta[i:i + PREDICT_BATCH]).cpu().numpy()
                               for i in range(0, len(seq), PREDICT_BATCH)])


def run_fold(cfg: dict, hp: dict, seed: int, device, rows, sequence, static, fold) -> tuple:
    """Fit on the fold's training rows and forecast its validation rows; returns (forecast in ug/m3, info, final model, statistics)."""
    fit = fold.fit
    target = rows["target_pm25"].to_numpy(dtype=float)
    stats = {"seq_mean": float(sequence[fit].mean()), "seq_std": float(sequence[fit].std()),
             "target_mean": float(target[fit].mean()), "target_std": float(target[fit].std())}      # fitted on this fold's training rows only
    info = {"fit_seconds": 0.0}
    t0 = time.perf_counter()
    if cfg["fixed_epochs"]:
        full = to_tensors(sequence[fit], static[fit], target[fit], stats, device)
        model, history, epochs = train_network(hp, cfg, seed, device, full, epochs=cfg["fixed_epochs"])
        info["curve"] = history
    else:
        core, inner = inner_early_stopping_split(rows, fit)
        core_t = to_tensors(sequence[core], static[core], target[core], stats, device)
        inner_t = to_tensors(sequence[inner], static[inner], target[inner], stats, device)
        _, history, epochs = train_network(hp, cfg, seed, device, core_t, validation=inner_t)
        info["curve"] = history
        full = to_tensors(sequence[fit], static[fit], target[fit], stats, device)
        model, _, _ = train_network(hp, cfg, seed, device, full, epochs=epochs)          # refit for the epoch count early stopping chose
    sync(device)
    info["fit_seconds"] = time.perf_counter() - t0
    info["rounds"] = epochs
    val = fold.val
    val_t = to_tensors(sequence[val], static[val], None, stats, device)
    t0 = time.perf_counter()
    scaled = predict(model, val_t[0], val_t[1])
    sync(device)
    info["predict_seconds"] = time.perf_counter() - t0
    return scaled * stats["target_std"] + stats["target_mean"], info, model, stats


def run_settings(cfg: dict, hp: dict, seed: int, device, rows, sequence, static, folds_by_name: dict) -> dict:
    """Run every requested setting (F: three folds; D: development; F2023: the single tuning fold) and score them like train.py."""
    result = {"metrics": {}, "tables": [], "info": {}, "model": None, "stats": None}
    out = {}
    for setting in cfg["settings"]:
        names = ["F2021", "F2022", "F2023"] if setting == "F" else ["F2023"] if setting == "F2023" else ["D"]
        part = {"metrics": {}, "info": {}}
        for name in names:
            fold = folds_by_name[name]
            forecast, info, model, stats = run_fold(cfg, hp, seed, device, rows, sequence, static, fold)
            metrics, table = score_fold(rows, fold, forecast)
            part["metrics"][name], part["info"][name] = metrics, info
            result["tables"].append(table)
            if name == "D":
                result["model"], result["stats"] = model, stats
            print(f"    seed {seed} {name:6s} MAE seen {metrics['mae_seen']:.3f}  RMSE {metrics['rmse_seen']:.3f}  skill {metrics['skill_seen']:+.3f}  "
                  f"epochs {info['rounds']}  fit {info['fit_seconds']:.1f}s  predict {info['predict_seconds']:.2f}s")
        out[setting] = part
    result["settings"] = out
    return result


def log_run(cfg: dict, hp: dict, seed: int, result: dict, device, rows, load_seconds: float, t_start: float, save: bool) -> dict:
    """Write one seed's results into the ACTIVE MLflow run: metrics, per-epoch curves, tables, timings, lineage; return the metrics."""
    metrics = {}
    for setting, part in result["settings"].items():
        if setting in ("F", "D"):
            metrics.update(summarise(setting, part))
        for name, fold_metrics in part["metrics"].items():
            mlflow.log_metrics({f"{name}_{k}": float(v) for k, v in fold_metrics.items() if pd.notna(v)})
            if setting == "F2023":
                metrics["F2023_mae_seen"] = fold_metrics["mae_seen"]
        for name, info in part["info"].items():
            for record in info["curve"].itertuples():
                mlflow.log_metric(f"{name}_train_loss", record.train_loss, step=record.epoch)
                if pd.notna(record.validation_loss):
                    mlflow.log_metric(f"{name}_inner_validation_loss", record.validation_loss, step=record.epoch)
            mlflow.log_metric(f"{name}_epochs", info["rounds"])
    fits = [i["fit_seconds"] for p in result["settings"].values() for i in p["info"].values()]
    predicts = [i["predict_seconds"] for p in result["settings"].values() for i in p["info"].values()]
    n_val = sum(m["n_val"] for p in result["settings"].values() for m in p["metrics"].values())
    metrics.update({"time_load_seconds": load_seconds, "time_fit_seconds": sum(fits), "time_predict_seconds": sum(predicts),
                    "inference_ms_per_1000_rows": 1e6 * sum(predicts) / max(n_val, 1), "time_total_seconds": time.perf_counter() - t_start})
    tags = lineage_tags()
    mlflow.set_tags({**tags, "model": "gru", "device": str(device), "seed": str(seed),
                     "evaluation_rows_sha256": rows_hash(rows[rows["split"] == "dev"])})
    mlflow.log_params({**{f"hp.{k}": v for k, v in hp.items()}, "seed": seed, "loss": cfg["loss"], "max_epochs": cfg["max_epochs"],
                       "patience": cfg["patience"], "fixed_epochs": str(cfg["fixed_epochs"]), "time_features": cfg["time_features"]})
    if device.type == "cuda":
        metrics["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6
    peak = peak_ram_mb()
    if peak is not None:
        metrics["peak_ram_mb"] = peak
    if result["model"] is not None:
        metrics["model_size_bytes"] = sum(p.numel() * p.element_size() for p in result["model"].parameters())
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pd.concat(result["tables"], ignore_index=True).to_csv(tmp / "per_station_results.csv", index=False)
        (tmp / "config.yaml").write_text(yaml.safe_dump({**cfg, "hp": hp, "seed": seed}, sort_keys=False), encoding="utf-8")
        (tmp / "pip_freeze.txt").write_text(pip_freeze_text(), encoding="utf-8")
        for setting, part in result["settings"].items():
            for name, info in part["info"].items():
                info["curve"].to_csv(tmp / f"learning_curve_{name}.csv", index=False)
        mlflow.log_artifacts(str(tmp))
    if save and result["model"] is not None:
        save_gru(cfg, hp, seed, result, metrics, tags, mlflow.active_run().info.run_id)
    return metrics


def save_gru(cfg: dict, hp: dict, seed: int, result: dict, metrics: dict, tags: dict, run_id: str) -> None:
    """Write models/pytorch_gru/v<N>/ (weights, configuration, the fold's scaling statistics, metrics, metadata)."""
    folder = MODELS_DIR / "pytorch_gru" / f"v{next_version('pytorch_gru')}"
    folder.mkdir(parents=True)
    torch.save(result["model"].state_dict(), folder / "model.pt")
    (folder / "config.yaml").write_text(yaml.safe_dump({**cfg, "hp": hp, "seed": seed}, sort_keys=False), encoding="utf-8")
    (folder / "preprocessing.json").write_text(json.dumps(result["stats"], indent=2), encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"mlflow_run_id": run_id, "experiment": cfg["experiment"], "seed": seed, **tags}, indent=2),
                                         encoding="utf-8")
    print(f"    model saved in {folder}")


def main():
    """Read the configuration, then run the tuning grid or the final seeds; print the scores and log to MLflow."""
    parser = argparse.ArgumentParser(description="Train and score the GRU (Milestone 3).")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="overrides `device` in the configuration (the default there is "
                        "auto, which silently uses a GPU when there is one); cuda stops the run if no CUDA GPU is available")
    args = parser.parse_args()
    cfg = {**CFG_DEFAULTS, **yaml.safe_load(args.config.read_text(encoding="utf-8"))}
    if args.device:
        cfg["device"] = args.device
    for key in ("experiment", "run_name"):
        if key not in cfg:
            raise SystemExit(f"The configuration needs a `{key}` entry.")
    if cfg["loss"] not in ("mse", "l1"):
        raise SystemExit("loss must be mse or l1.")
    grid = cfg.get("grid")
    if grid:
        cfg["settings"], cfg["seeds"] = ["F2023"], cfg["seeds"][:1]     # tuning: one fold, one seed
    if args.save_model and "D" not in cfg["settings"]:
        raise SystemExit("--save-model needs setting D.")
    if cfg["threads"]:
        torch.set_num_threads(int(cfg["threads"]))
    device = choose_device(cfg["device"])

    t_start = time.perf_counter()
    rows = load_rows()
    check_persistence_benchmark(rows)
    sequence = build_sequences(rows)
    static = time_features(rows, cfg["time_features"]).to_numpy(dtype=float)
    folds_by_name = {f.name: f for f in expanding_folds(rows)}
    load_seconds = time.perf_counter() - t_start
    print(f"{cfg['experiment']} / {cfg['run_name']}: GRU, device {device}, threads {torch.get_num_threads()}, windows {sequence.shape}, "
          f"static {static.shape[1]} columns, loss {cfg['loss']}  (data load {load_seconds:.1f}s)")
    tracking = not args.no_tracking
    if tracking:
        setup_experiment(cfg["experiment"])

    if grid:
        keys = list(grid)
        configs = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
        if len(configs) > MAX_GRID:
            raise SystemExit(f"The grid has {len(configs)} configurations; the limit is {MAX_GRID} (a small controlled search, not a second HPO project).")
        print(f"Tuning grid: {len(configs)} configurations on fold F2023 (train 2018-2022, validate 2023), seed {cfg['seeds'][0]}")
        scores = []
        for i, point in enumerate(configs):
            hp = {**HP_DEFAULTS, **cfg["hp"], **{k: v for k, v in point.items() if k != "loss"}}
            point_cfg = {**cfg, "loss": point.get("loss", cfg["loss"])}            # the loss function is a grid axis too
            if point_cfg["loss"] not in ("mse", "l1"):
                raise SystemExit("loss must be mse or l1.")
            print(f"  config {i + 1}/{len(configs)}: {point}")
            result = run_settings(point_cfg, hp, cfg["seeds"][0], device, rows, sequence, static, folds_by_name)
            if tracking:
                with mlflow.start_run(run_name=f"{cfg['run_name']}_grid{i:02d}"):
                    metrics = log_run(point_cfg, hp, cfg["seeds"][0], result, device, rows, load_seconds, t_start, False)
            else:
                metrics = {"F2023_mae_seen": result["settings"]["F2023"]["metrics"]["F2023"]["mae_seen"]}
            scores.append({**point, "F2023_mae_seen": metrics["F2023_mae_seen"], "epochs": result["settings"]["F2023"]["info"]["F2023"]["rounds"]})
        table = pd.DataFrame(scores).sort_values("F2023_mae_seen")
        print("\nGrid results (best first; selection fold only, the development year is not used):")
        print(table.round(4).to_string(index=False))
        return

    hp = {**HP_DEFAULTS, **cfg["hp"]}
    print(f"Final mode: hyperparameters {hp}, settings {cfg['settings']}, seeds {cfg['seeds']}")
    all_metrics = []
    for seed in cfg["seeds"]:
        result = run_settings(cfg, hp, seed, device, rows, sequence, static, folds_by_name)
        if "D" in cfg["settings"]:
            check_persistence_benchmark(rows)
        if tracking:
            with mlflow.start_run(run_name=f"{cfg['run_name']}_seed{seed}"):
                all_metrics.append(log_run(cfg, hp, seed, result, device, rows, load_seconds, t_start, args.save_model))
    if tracking and len(all_metrics) > 1:
        frame = pd.DataFrame(all_metrics)
        with mlflow.start_run(run_name=f"{cfg['run_name']}_summary"):
            mlflow.set_tags({"kind": "summary", "model": "gru", "n_seeds": str(len(all_metrics))})
            for column in frame.columns:
                if column.startswith(("D_", "F_mean")) or column in ("time_fit_seconds", "inference_ms_per_1000_rows"):
                    mlflow.log_metrics({f"{column}_mean": float(frame[column].mean()), f"{column}_std": float(frame[column].std(ddof=0))})
            print("\nSeed summary: " + "  ".join(f"{c} {frame[c].mean():.4f} +/- {frame[c].std(ddof=0):.4f}" for c in ("D_mae", "D_rmse", "D_skill")
                                                 if c in frame.columns))


if __name__ == "__main__":
    main()
