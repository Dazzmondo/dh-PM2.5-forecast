"""
Write one saved model to GCS and record exactly what was written (the model's entry in the bucket's version history).

USAGE:  python src/store_models.py --model-dir models/<model>/v<N> [--dataset-run-id latest|<id>] [--dry-run]
INPUT:  a local model folder written by train.py or train_gru.py:
            model.json | model.pkl | model.pt, config.yaml, metrics.json, metadata.json   (GRU: also preprocessing.json)
        and, in the bucket, manifests/dataset_versions.jsonl (written by store_preprocessed_data.py)
WRITES  models/<model>/v<N>/<each file>        (the layout shown in the README)
        manifests/model_versions.jsonl         (one line appended per model)

Why a script of its own: store_preprocessed_data.py stores the data pipeline's stages in one run; a model is stored after it is
trained and frozen, one version at a time, from a different local folder, and gets its own manifest. Like upload_raw_data.py and
store_preprocessed_data.py, this script does one job and records it.

Rules:
  - A model version is immutable. If anything already exists under models/<model>/v<N>/ in the bucket, the script stops and writes nothing.
  - The model record carries the IDs that tie it to the data it came from: the dataset run ID (and the raw-data version) from
    manifests/dataset_versions.jsonl, the SHA-256 of that dataset's scalers.json, the MLflow run ID and the git commit from the model's
    own metadata.json. With --dataset-run-id latest (the default) the newest dataset record is used and the record says so
    (`dataset_run_id_source`); pass an explicit ID when the model was trained on an older dataset. If the model's metadata already names
    a dataset run ID and it differs from the one chosen, the script stops.
  - Preprocessing (scaling, window statistics) is fitted inside each model and stored with it (the Ridge pipeline, or
    preprocessing.json for the GRU); `scaler_artefact_sha256` is the stored scalers.json of the same dataset version, for lineage.
  - --dry-run checks the folder and prints what would be uploaded without contacting GCS.

Needs GCP_PROJECT_ID and GCS_BUCKET_NAME set as environment variables and Google Cloud application-default credentials (no credentials
are stored in the code).

Requires: pip install google-cloud-storage pyyaml
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from google.cloud import storage

from config import check_gcp_settings
from store_preprocessed_data import BUCKET_NAME, DATASET_VERSIONS_MANIFEST, PROJECT_ID, sha256_of_file, upload_file

MODEL_VERSIONS_MANIFEST = "manifests/model_versions.jsonl"   # one line appended per stored model
MODEL_FILES = ("model.json", "model.pkl", "model.pt")
REQUIRED_FILES = ("config.yaml", "metrics.json", "metadata.json")
SCALER_GCS_SUFFIX = "processed/scalers/scalers.json"
KEPT_METRICS = ("F_mean_mae_seen", "F_mean_skill_seen", "D_mae", "D_rmse", "D_skill", "D_mae_without_cork", "D_stations_beaten",
                "time_fit_seconds", "inference_ms_per_1000_rows", "model_size_bytes")


def check_folder(folder: Path) -> list:
    """The files to upload, in order; stops if the folder is not a complete saved model."""
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder.")
    models = [name for name in MODEL_FILES if (folder / name).exists()]
    if len(models) != 1:
        raise SystemExit(f"{folder} must contain exactly one model file ({', '.join(MODEL_FILES)}); found {models}.")
    missing = [name for name in REQUIRED_FILES if not (folder / name).exists()]
    if missing:
        raise SystemExit(f"{folder} is missing {missing}.")
    if models[0] == "model.pt" and not (folder / "preprocessing.json").exists():
        raise SystemExit(f"{folder} is a GRU but has no preprocessing.json (its scaling statistics).")
    return sorted(p.name for p in folder.iterdir() if p.is_file())


def latest_or_named_dataset_record(bucket, wanted: str) -> dict:
    """The dataset record with this run ID (or the newest one), read from the bucket's manifest; stops if there is none."""
    blob = bucket.blob(DATASET_VERSIONS_MANIFEST)
    if not blob.exists():
        raise SystemExit(f"{DATASET_VERSIONS_MANIFEST} not found in the bucket. Run store_preprocessed_data.py first.")
    records = [json.loads(line) for line in blob.download_as_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"{DATASET_VERSIONS_MANIFEST} is empty.")
    if wanted == "latest":
        return records[-1]
    matches = [r for r in records if r["dataset_run_id"] == wanted]
    if not matches:
        raise SystemExit(f"No dataset record with run ID {wanted!r}; known: {[r['dataset_run_id'] for r in records]}.")
    return matches[0]


def main():
    """Check the model folder, check the bucket, upload the files and append the model's record to the manifest."""
    parser = argparse.ArgumentParser(description="Store one saved model in GCS.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dataset-run-id", default="latest")
    parser.add_argument("--dry-run", action="store_true", help="check the folder and print the plan; do not contact GCS")
    args = parser.parse_args()
    folder = args.model_dir.resolve()
    files = check_folder(folder)
    model_type, version = folder.parent.name, folder.name
    prefix = f"models/{model_type}/{version}/"
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    print(f"Model {model_type}/{version}: {len(files)} files, MLflow run {metadata.get('mlflow_run_id')}, commit {metadata.get('git_commit')}")
    if args.dry_run:
        for name in files:
            print(f"  would upload {name} -> {prefix}{name}  ({(folder / name).stat().st_size:,} bytes, sha256 {sha256_of_file(folder / name)[:12]}...)")
        print(f"  would append one line to {MODEL_VERSIONS_MANIFEST}. Dry run: nothing was contacted or written.")
        return

    check_gcp_settings(PROJECT_ID, BUCKET_NAME)
    bucket = storage.Client(project=PROJECT_ID).bucket(BUCKET_NAME)
    if any(True for _ in bucket.list_blobs(prefix=prefix, max_results=1)):
        raise SystemExit(f"STOPPED: gs://{BUCKET_NAME}/{prefix} already holds files. A model version is immutable; train a new version.")
    dataset = latest_or_named_dataset_record(bucket, args.dataset_run_id)
    recorded = metadata.get("dataset_run_id", "not set")
    if args.dataset_run_id == "latest" and recorded not in ("not set", dataset["dataset_run_id"]):
        raise SystemExit(f"STOPPED: the model's metadata names dataset run {recorded}, but the newest record is {dataset['dataset_run_id']}. "
                         f"Pass --dataset-run-id {recorded} (or the right ID) explicitly.")
    scaler = next((o["sha256"] for o in dataset["outputs"] if o["gcs_path"].endswith(SCALER_GCS_SUFFIX)), None)

    local_scaler = metadata.get("scaler_artefact_sha256")
    if local_scaler and scaler and local_scaler != scaler:
        print(f"WARNING: the scalers.json the model was checked against (sha256 {local_scaler[:12]}...) is not the one in the dataset record "
              f"({scaler[:12]}...); both hashes are written to the model record.")
    outputs = [upload_file(bucket, folder / name, prefix + name) for name in files]
    entry = {
        "model_id": f"{model_type}/{version}", "model_type": model_type, "model_version": version,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": metadata.get("mlflow_run_id"), "experiment": metadata.get("experiment"), "git_commit": metadata.get("git_commit"),
        "dataset_run_id": dataset["dataset_run_id"],
        "dataset_run_id_source": "named" if args.dataset_run_id != "latest" else ("model metadata" if recorded != "not set" else "latest record at upload time"),
        "raw_data_version": dataset.get("raw_data_version"), "scaler_artefact_sha256": scaler,
        "scaler_check": metadata.get("scaler_check"),                       # "passed" when the in-model scaler equalled scalers.json
        "scaler_artefact_sha256_local_at_training": metadata.get("scaler_artefact_sha256"),
        "features_file_sha256": metadata.get("features_file_sha256"),
        "configuration": {k: config[k] for k in ("model", "feature_set", "time_features", "target", "scaling", "place", "seed", "params", "hp")
                          if k in config},
        "metrics": {k: v for k, v in json.loads((folder / "metrics.json").read_text(encoding="utf-8")).items() if k in KEPT_METRICS},
        "outputs": outputs,
    }
    blob = bucket.blob(MODEL_VERSIONS_MANIFEST)
    existing = blob.download_as_text() if blob.exists() else ""
    blob.upload_from_string(existing + json.dumps(entry) + "\n")
    print(f"Manifest entry appended to {MODEL_VERSIONS_MANIFEST}. Model stored under gs://{BUCKET_NAME}/{prefix}")


if __name__ == "__main__":
    main()
