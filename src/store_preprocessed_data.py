"""
Write every processed stage to GCS and record exactly what was written.

INPUTS (all under processed_data/, produced by the earlier scripts):
    canonical_hourly.parquet                   canonicalize_raw_data.py
    three_hour_aggregated.parquet              aggregate_to_3hour.py
    features/pm25_features_with_splits.parquet feature_extraction.py + create_splits.py

WRITES
    processed/canonical/canonical_hourly.parquet
    processed/three_hour/three_hour_aggregated.parquet
    processed/features/pm25_features.parquet            (every row, with its `split` column)
    processed/splits/{train,dev,test,held_out,future_update}/<name>.parquet
    metadata/station_reference.csv
    manifests/dataset_versions.jsonl                    (one line appended per run)

Layout choices: one file per stage (about a million rows - partitioning by year
would add complexity for no benefit), and the `split` folders are plain subsets
of the features file. Old versions are kept by bucket object versioning plus the
hashes recorded below, not by copying everything into numbered folders.

Each run's manifest line records:
    dataset_run_id, pipeline/preprocessing/feature versions, git_commit,
    station_reference_sha256, raw_data_version (SHA-256 of manifests/raw_files.jsonl,
    i.e. exactly which raw files existed), row_counts per split, the configuration
    numbers, and an `outputs` list: gcs_path, sha256, size_bytes and rows of every
    file written. So "which exact data produced this model?" is answerable.

Needs GCP_PROJECT_ID and GCS_BUCKET_NAME set as environment variables and
Google Cloud application-default credentials (no credentials are stored in
the code). Run upload_raw_data.py first so the raw manifest exists.

Requires: pip install google-cloud-storage pandas pyarrow
"""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.cloud import storage

from aggregate_to_3hour import MIN_VALID_HOURS
from config import PIPELINE_VERSION, PROCESSED_DIR, REFERENCE_PATH, check_gcp_settings, git_commit
from feature_extraction import ROLLING_WINDOW_BLOCKS, TARGET_HORIZON_BLOCKS
from validate_data_quality import COMPLETENESS_THRESHOLD

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your-unique-bucket-name")

# Local inputs (produced by the earlier scripts)
CANONICAL_PATH = PROCESSED_DIR / "canonical_hourly.parquet"
THREE_HOUR_PATH = PROCESSED_DIR / "three_hour_aggregated.parquet"
INPUT_PATH = PROCESSED_DIR / "features" / "pm25_features_with_splits.parquet"

RAW_MANIFEST = "manifests/raw_files.jsonl"                    # written by upload_raw_data.py
DATASET_VERSIONS_MANIFEST = "manifests/dataset_versions.jsonl"   # one line appended per run of this script

PREPROCESSING_VERSION = "v1"   # labels recorded in the manifest: change them when the cleaning/aggregation
FEATURE_VERSION = "v1"         # rules or the feature/target definitions change

# split label -> where that subset is stored. Rows labelled "excluded" are not stored separately:
# they appear only in the full feature table.
SPLIT_TO_FOLDER = {
    "train": "processed/splits/train/train.parquet",
    "dev": "processed/splits/dev/dev.parquet",
    "test": "processed/splits/test/test.parquet",
    "held_out_test": "processed/splits/held_out/held_out_test.parquet",
    "future_update": "processed/splits/future_update/future_update.parquet",
}


def sha256_of_file(path: Path) -> str:
    """Fingerprint of a file's contents, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_file(bucket, local_path: Path, gcs_path: str, rows: int = None) -> dict:
    """Upload one local file and return its manifest record."""
    bucket.blob(gcs_path).upload_from_filename(str(local_path))
    record = {"gcs_path": f"gs://{bucket.name}/{gcs_path}", "sha256": sha256_of_file(local_path),
              "size_bytes": local_path.stat().st_size, "rows": rows}
    print(f"Uploaded {gcs_path}" + (f" ({rows} rows)" if rows is not None else ""))
    return record


def upload_dataframe(bucket, df: pd.DataFrame, gcs_path: str) -> dict:
    """Write a DataFrame to a temporary Parquet file, upload it, and always delete the temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:   # delete=False: the file outlives this "with" block
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        return upload_file(bucket, tmp_path, gcs_path, rows=len(df))
    finally:
        tmp_path.unlink(missing_ok=True)


def raw_data_version(bucket):
    """SHA-256 of the raw-file manifest = a fingerprint of exactly which raw files were uploaded."""
    blob = bucket.blob(RAW_MANIFEST)
    if not blob.exists():
        print(f"WARNING: {RAW_MANIFEST} not found - run upload_raw_data.py first; raw_data_version will be null.")
        return None
    return hashlib.sha256(blob.download_as_text().encode("utf-8")).hexdigest()


def append_manifest_entry(bucket, entry: dict) -> None:
    """Add one JSON line to the dataset manifest (download the existing lines, add ours, upload)."""
    blob = bucket.blob(DATASET_VERSIONS_MANIFEST)
    existing = blob.download_as_text() if blob.exists() else ""
    blob.upload_from_string(existing + json.dumps(entry) + "\n")
    print(f"Manifest entry appended to {DATASET_VERSIONS_MANIFEST}")


def main():
    """Upload every processed stage to GCS and append one record of this run to the dataset manifest."""
    # stop early, naming the script to run, if an earlier stage's output is missing
    for path, hint in ((CANONICAL_PATH, "canonicalize_raw_data.py"), (THREE_HOUR_PATH, "aggregate_to_3hour.py"),
                       (INPUT_PATH, "create_splits.py")):
        if not path.exists():
            raise SystemExit(f"{path} not found. Run {hint} first.")

    check_gcp_settings(PROJECT_ID, BUCKET_NAME)
    features = pd.read_parquet(INPUT_PATH)   # every feature row, with its `split` label
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    outputs = []   # one record per uploaded file; becomes the "outputs" list in the manifest

    # 1. canonical hourly table   2. 3-hour table   (uploaded as they are; only the row count is read)
    outputs.append(upload_file(bucket, CANONICAL_PATH, "processed/canonical/canonical_hourly.parquet",
                               rows=len(pd.read_parquet(CANONICAL_PATH, columns=["station_id"]))))
    outputs.append(upload_file(bucket, THREE_HOUR_PATH, "processed/three_hour/three_hour_aggregated.parquet",
                               rows=len(pd.read_parquet(THREE_HOUR_PATH, columns=["station_id"]))))
    # 3. the full feature table (every row, including excluded ones)
    outputs.append(upload_dataframe(bucket, features, "processed/features/pm25_features.parquet"))
    # 4. one file per split
    for split_name, gcs_path in SPLIT_TO_FOLDER.items():
        subset = features[features["split"] == split_name]
        if subset.empty:
            print(f"No rows for split '{split_name}' - skipping upload.")
            continue
        outputs.append(upload_dataframe(bucket, subset, gcs_path))
    # 5. the station table that produced this dataset
    outputs.append(upload_file(bucket, REFERENCE_PATH, "metadata/station_reference.csv"))

    # One manifest line: which code, raw files, station table and settings produced these files
    now = datetime.now(timezone.utc)
    append_manifest_entry(bucket, {
        "dataset_run_id": now.strftime("%Y%m%dT%H%M%SZ"),
        "pipeline_version": PIPELINE_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_version": FEATURE_VERSION,
        "git_commit": git_commit(),
        "raw_data_version": raw_data_version(bucket),
        "station_reference_sha256": sha256_of_file(REFERENCE_PATH),
        "row_counts": {k: int(v) for k, v in features["split"].value_counts().items()},
        "config": {
            "rolling_window_blocks": ROLLING_WINDOW_BLOCKS,
            "target_horizon_blocks": TARGET_HORIZON_BLOCKS,
            "aggregation_min_valid_hours": MIN_VALID_HOURS,
            "completeness_threshold": COMPLETENESS_THRESHOLD,
        },
        "outputs": outputs,
        "created_timestamp_utc": now.isoformat(),
    })


if __name__ == "__main__":
    main()
