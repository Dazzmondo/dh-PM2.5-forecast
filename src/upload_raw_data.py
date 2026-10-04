"""
Upload the downloaded raw files to GCS, mirroring raw_data/ into
gs://<bucket>/raw/ without changing them.

INPUT:  raw_data/   (original source files only)
OUTPUT: gs://<bucket>/raw/...                   the files, same relative paths
        gs://<bucket>/manifests/raw_files.jsonl  one JSON line per uploaded file

What it does
1. Creates the bucket if it does not exist (EU location, uniform bucket-level
   access, object versioning on). If the bucket exists with versioning off,
   it switches versioning on, so an overwritten file keeps its earlier copy.
2. Computes a SHA-256 checksum of each local file BEFORE uploading and
   compares it with the manifest: a file already in the bucket with the same
   checksum is skipped, a new file is uploaded, and a file whose content has
   changed is uploaded again as an update.
3. Records every uploaded file in the manifest: local and GCS path, checksum,
   size, file modified time, upload time, pipeline version, git commit and
   source (epa / uk_air / eea).
4. Compares the number of local files with the number of objects under raw/
   and warns if they differ.

Only ORIGINAL source files belong in raw_data/. Anything derived is written
to processed_data/ so it is never uploaded as if it were source data.

Needs GCP_PROJECT_ID and GCS_BUCKET_NAME set as environment variables and
Google Cloud application-default credentials (for example after
`gcloud auth application-default login`); no credentials are stored in the code.

Known limitation (accepted at this scale): the manifest is read once and
rewritten whole, so it isn't safe for several simultaneous uploaders.

Requires: pip install google-cloud-storage
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

from config import PIPELINE_VERSION, RAW_DIR, check_gcp_settings, git_commit

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your-unique-bucket-name")
LOCATION = "EU"                 # where the bucket is created (only used if it does not exist yet)
LOCAL_RAW_DIR = RAW_DIR         # the folder that gets mirrored into the bucket

MANIFEST_BLOB = "manifests/raw_files.jsonl"   # path of the manifest INSIDE the bucket
SKIP_NAMES = {".DS_Store", "Thumbs.db", ".gitkeep"}   # operating-system clutter, never uploaded


def get_client() -> storage.Client:
    """Storage client for the configured project. Credentials come from Application Default
    Credentials (see the module docstring), never from the code."""
    return storage.Client(project=PROJECT_ID)


def create_gcs_bucket(client, bucket_name: str, location: str = "EU") -> None:
    """Make sure the bucket exists and has object versioning on (creating it if needed)."""
    existing = client.lookup_bucket(bucket_name)   # the bucket, or None if it does not exist
    if existing:
        print(f"Bucket 'gs://{bucket_name}' already exists.")
        if not existing.versioning_enabled:
            existing.versioning_enabled = True
            existing.patch()
            print("Object versioning was off - switched it on.")
        return
    bucket = client.bucket(bucket_name)
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True   # one access policy for the whole bucket
    bucket.versioning_enabled = True   # keep old generations when a file is re-uploaded
    bucket.lifecycle_rules.add_delete_rule(
        age=7,             #Number of days
        is_live=False      #False means target only archived historical versions
    )
    bucket = client.create_bucket(bucket, location=location)
    print(f"Created bucket: gs://{bucket.name}")


def sha256_of_file(path: Path) -> str:
    """Fingerprint of a file's contents. Read in 1 MiB chunks so a large file never has to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):   # 1 << 20 = 1,048,576 bytes
            h.update(chunk)
    return h.hexdigest()


def load_existing_manifest(bucket):
    """Returns ({gcs_url: checksum}, [records]). Both empty on the first run."""
    blob = bucket.blob(MANIFEST_BLOB)
    if not blob.exists():
        return {}, []
    records = [json.loads(line) for line in blob.download_as_text().splitlines() if line.strip()]
    latest = {r["gcs_path"]: r["checksum_sha256"] for r in records}  # later entries win
    return latest, records


def upload_file(bucket, local_path: Path, gcs_path: str, checksum: str, commit: str = "unknown") -> dict:
    """Upload one file and return its manifest record (the dict that becomes one JSON line)."""
    bucket.blob(gcs_path).upload_from_filename(str(local_path))
    print(f"Uploaded {local_path} -> gs://{bucket.name}/{gcs_path}")
    return {
        "local_path": str(local_path),
        "gcs_path": f"gs://{bucket.name}/{gcs_path}",
        "checksum_sha256": checksum,
        "size_bytes": local_path.stat().st_size,
        "file_modified_utc": datetime.fromtimestamp(local_path.stat().st_mtime, timezone.utc).isoformat(),
        "upload_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": commit,
    }


def upload_local_tree(client, bucket_name: str, local_root: Path) -> None:
    """Upload every new or changed file under local_root and append their records to the manifest."""
    bucket = client.bucket(bucket_name)
    # known_checksums: {gs:// path: checksum} from earlier runs; manifest: all earlier records
    known_checksums, manifest = load_existing_manifest(bucket)
    existing_objects = {b.name for b in client.list_blobs(bucket_name, prefix="raw/")}   # what is in the bucket now
    new_records, uploaded, updated, skipped = [], 0, 0, 0
    commit = git_commit()   # the same code version is recorded for every file of this run

    for local_path in sorted(local_root.rglob("*")):   # every file under raw_data/, in a fixed order
        if local_path.is_dir() or local_path.name in SKIP_NAMES:
            continue
        relative = local_path.relative_to(local_root).as_posix()   # path below raw_data/, with forward slashes
        gcs_path = f"raw/{relative}"
        gcs_url = f"gs://{bucket_name}/{gcs_path}"
        checksum = sha256_of_file(local_path)   # computed BEFORE uploading, from the exact file being sent

        previous = known_checksums.get(gcs_url)   # checksum recorded when this file was last uploaded (None = new)
        if previous == checksum and gcs_path in existing_objects:
            skipped += 1   # unchanged AND still in the bucket: nothing to do
            continue
        if previous is not None and previous != checksum:   # same path, different content: a revised file
            print(f"Checksum changed for {relative} - uploading the updated version.")
            updated += 1
        else:
            uploaded += 1

        record = upload_file(bucket, local_path, gcs_path, checksum, commit)
        record["source"] = relative.split("/")[0]   # first folder = the data source: epa, uk_air or eea
        new_records.append(record)

    print(f"\n{uploaded} new, {updated} updated, {skipped} unchanged (skipped).")
    if new_records:
        manifest.extend(new_records)   # the manifest is rewritten whole: all earlier lines plus this run's
        bucket.blob(MANIFEST_BLOB).upload_from_string("\n".join(json.dumps(r) for r in manifest) + "\n")
        print(f"Manifest updated: {len(manifest)} total entries.")


def verify_bucket(client, bucket_name: str, local_root: Path) -> None:
    """Sanity check: compare the number of local files with the number of objects under raw/."""
    local_count = sum(1 for p in local_root.rglob("*") if p.is_file() and p.name not in SKIP_NAMES)
    remote = list(client.list_blobs(bucket_name, prefix="raw/"))
    print(f"\n--- Verification ---\nLocal files:    {local_count}\nObjects in GCS: {len(remote)}")
    print("Counts match." if local_count == len(remote) else "WARNING: counts don't match - check for failures above.")


def main():
    """Check the settings, make sure the bucket exists, upload the raw files, then verify the counts."""
    if not LOCAL_RAW_DIR.exists():
        raise SystemExit(f"{LOCAL_RAW_DIR} does not exist. Put downloads under raw_data/epa|uk_air|eea first.")
    check_gcp_settings(PROJECT_ID, BUCKET_NAME)
    client = get_client()
    create_gcs_bucket(client, BUCKET_NAME, LOCATION)
    upload_local_tree(client, BUCKET_NAME, LOCAL_RAW_DIR)
    verify_bucket(client, BUCKET_NAME, LOCAL_RAW_DIR)


if __name__ == "__main__":
    main()
