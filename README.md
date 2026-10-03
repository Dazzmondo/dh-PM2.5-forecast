# Six-Hour Urban PM2.5 Forecasting Across Ireland and Europe

## Project Overview

This project develops an end-to-end AI/ML system for forecasting **3-hour mean PM2.5 concentration 6 hours ahead** at urban air-quality monitoring stations.

The project treats the problem primarily as a **time-series and structured-data ML problem**. Historical PM2.5 observations are collected from official environmental monitoring sources, stored in cloud object storage, transformed into a common canonical dataset, analysed and converted to model features, and then used to train and evaluate several models.

The main research question is:

> **Can machine learning accurately forecast PM2.5 concentrations 6 hours ahead across different Irish and European cities using historical air-quality measurements?**

Two secondary questions are:

1. Can a model trained across several cities generalise to a city not seen during training?
2. Does including geographically and environmentally diverse cities improve or affect generalisation?

The second question is a hypothesis to be tested rather than an assumption that greater diversity will necessarily improve performance.

The initial modelling progression is:

**Persistence baseline → Linear Regression → XGBoost → PyTorch GRU/LSTM (Optional)**

This allows simple, interpretable, nonlinear tabular, and explicit sequence-based approaches to be compared.

---

# Milestone 1 Rubric

| **Criterion** | **Points** | **Project specification** |
|---|---:|---|
| **1. Raw data storage** | **1.0** | Raw source data will be stored unchanged in **Google Cloud Storage (GCS)** under source-specific `raw/` paths. GCS is appropriate because the raw datasets are historical files that need durable, scalable object storage and repeated batch access rather than transactional updates. |
| **2. Processed data storage & file formats** | **1.0** | Processed canonical measurements and model features will be stored in **Parquet** in GCS. Parquet is appropriate for columnar analytical access to structured time-series data. Raw files remain in their original source formats. |
| **3. Database / object storage decision** | **0.5** | GCS will be the authoritative object/data-lake layer. **BigQuery** would provide an analytical/query layer for SQL exploration and data-quality analysis. BigQuery may be abandoned for simplicity and to reduce setup time, as pandas should be sufficient for ~324k rows on Parquet files. A transactional SQL database is not required because the project has no transactional workload. |
| **4. Data versioning** | **0.5** | Raw files are identified by source, path/filename, SHA-256 checksum, size, last-modified and upload timestamps, pipeline version and git commit. Dataset versions are linked to the raw-data version, preprocessing configuration and code version to provide data lineage. |
| **5. Data access** | **1.0** | Python ingestion scripts will access official EEA, EPA Ireland and UK-AIR sources, download raw files and upload them to GCS. Training/preprocessing code will read versioned Parquet data from GCS, with either BigQuery or pandas used for analytical queries. Google Cloud authentication will use service-account/application-default credentials rather than embedded credentials. |
| **6. Data split / validation strategy** | **2.0** | A chronological train/dev/test strategy will be used: **2018–2023 training (2022-2023 for Irish stations), 2024 development, 2025 final test**. 2026 is reserved for a future-data/model-update demonstration. Derry, London and Paris are geographical holdouts and are excluded from model development. Time-aware CV may be used within 2018–2023. Future information, test data and target-period observations will not influence training or feature construction. |
| **7. Feature description** | **1.0** | Features include historical PM2.5 lags and rolling statistics, temporal variables, and geographical information. Each feature is constructed only from information available before the forecast target. |
| **8. Data types and formats** | **0.5** | The dataset contains numerical continuous data (PM2.5, coordinates), categorical data (city, country), integer temporal variables, timestamps, station identifiers and valid hour counts. Raw sources include CSV and Parquet; processed data will use Parquet; manifests use JSONL. |
| **9. Reproducibility of data collection** | **1.0** | Collection is implemented in version-controlled Python ingestion scripts. Collection parameters (source, station, pollutant, date range) are documented in this README and metadata/station_reference.csv. Each raw file's path, checksum, size, timestamps, pipeline version and git commit are recorded in a manifest. |
| **10. Reproducibility of preprocessing** | **1.5** | Raw data will be parsed, standardised, mapped to canonical stations, checked for duplicates and quality issues, aggregated into 3-hour measurements, converted into 6-hour-ahead targets, and transformed into historical lag/rolling/temporal features. The exact configuration, completeness rules and dataset version will be recorded so the processed dataset can be regenerated from the raw data. |

**Total: 10.0 points**

---

# 1. Raw Data Storage — 1.0 point

## Storage system

All original source files will be retained in **Google Cloud Storage (GCS)**.

The proposed structure is:

```text
gs://<project-bucket>/
│
├── raw/
│   ├── eea/
│   |   ├── athens/
│   |   │   ├── athensAristotelous/
│   |   │   └── athensParaskevi/
│   |   ├── copenhagen/
│   |   ├── madrid/
│   |   │   ├── madridCuatroCaminos/
│   |   │   └── madridEscuelasAguirre/
│   |   └── paris
│   |       ├── parisGennevilliers/
│   |       └── parisSaintDenis/
|   |
│   ├── epa/
│   │   ├── cork/
│   │   ├── dublin/
│   │   │   ├── dublinKilmainham/
│   │   │   └── dublinRathmines/
│   │   ├── galway/
│   │   ├── limerick/
│   │   └── waterford/
|   |
│   └── uk_air/
│       ├── belfast/
│       ├── derry/
│       └── london/
│           ├── londonBloomsbury/
│           └── londonKensington/
│
│
├── processed/
│   ├── canonical/
│   │   └── canonical_hourly.parquet
│   │
│   ├── three_hour/
│   │   └── three_hour_aggregated.parquet
│   │
│   ├── features/
│   │   └── pm25_features.parquet
│   │
│   └── splits/
│       ├── train/
│       ├── dev/
|       ├── test/
|       ├── held_out/
│       └── future_update/
│
├── manifests/
│   ├── raw_files.jsonl
│   └── dataset_versions.jsonl
│
├── metadata/
│   ├── station_reference.csv
│   └── source_metadata/
│
└── models/
    ├── linear_regression/
    │   └── v1/
    │       ├── model.joblib
    │       ├── config.yaml
    │       ├── metrics.json
    │       └── metadata.json
    │
    ├── xgboost/
    │   └── v1/
    │       ├── model.json
    │       ├── config.yaml
    │       ├── metrics.json
    │       └── metadata.json
    │
    └── pytorch_gru/ (optional)
        └── v1/
            ├── model.pt
            ├── config.yaml
            ├── preprocessing.json
            ├── metrics.json
            └── metadata.json
```

The raw files will be preserved **unchanged**. Processing will create separate derived datasets rather than overwriting the source data.

This follows the data-lake principle from the lectures: retaining the raw data means that if a preprocessing decision is later found to be incorrect, the dataset can be regenerated without recollecting everything.

GCS is appropriate because the raw data is primarily accessed through **batch processing**, is not transactional, and needs durable, scalable object storage.

The raw data sources are:

- **European Environment Agency (EEA)** Air Quality Download Service
- **Environmental Protection Agency Ireland (EPA) / AirQuality.ie** current/provisional air-quality measurements
- **UK-AIR** automatic monitoring data

### Raw data upload code (`upload_raw_data.py`)
 
`src/upload_raw_data.py` is the script that moves the downloaded raw files into storage. It mirrors the local `raw_data/` folder into `gs://<project-bucket>/raw/` without changing the files.
 
- It creates the bucket first if it does not exist (location `EU`, uniform bucket-level access, object versioning on). If the bucket already exists with versioning off, versioning is switched on.
- For every local file it computes a SHA-256 checksum *before* uploading and compares it with the raw manifest (`manifests/raw_files.jsonl`). A file already in the bucket with the same checksum is skipped, a new file is uploaded, and a file whose content has changed is uploaded again as an update (the earlier copy is kept by bucket versioning).
- Every uploaded file adds one line to the manifest (fields in Section 4).
- At the end it compares the number of local files with the number of objects under `raw/` and warns if they differ.

```python
# src/upload_raw_data.py
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
```

---

# 2. Processed Data Storage and File Formats — 1.0 point

Processed data will be stored separately from the raw data in GCS.

The main processed format will be **Apache Parquet**.

Planned structure:

```text
processed/
├── canonical/
│   └── canonical_hourly.parquet
|
├── three_hour/
│   └── three_hour_aggregated.parquet
│
├── features/
│   └── pm25_features.parquet
│
└── splits/
    ├── train/
    ├── dev/
    ├── test/
    ├── held_out/
    └── future_update/  
```

### Why Parquet?

The project produces structured time-series data with many numerical columns and repeated analytical operations.

Parquet is appropriate because it provides:

- columnar storage;
- efficient analytical reads;
- compression;
- efficient selection of only the columns required for a particular operation;
- compatibility with Pandas, PyArrow, BigQuery and cloud-based analytical workflows.

This is a deliberate choice based on **access pattern**, rather than simply using CSV because it is familiar.

CSV will still be retained where it is the original UK-AIR and EPA source format.

The distinction between **data type and storage format** is important:

- PM2.5 is numerical time-series data.
- A timestamp represents temporal data.
- These data types can be represented using CSV, Parquet, database tables or other formats.

Therefore, the choice of Parquet is a storage/I/O decision rather than a claim that PM2.5 is itself a "Parquet data type."

### Storing the processed data (`store_preprocessed_data.py`)
 
`src/store_preprocessed_data.py` writes every processed stage to GCS as Parquet, following the structure above: the canonical hourly table, the 3-hour table, the full feature table (every row, with its `split` label) and one file per split (train, dev, test, held_out, future_update). Rows labelled `excluded` stay in the full feature file only. It also uploads `metadata/station_reference.csv`, stops with a clear message if an earlier pipeline stage has not been run, and appends one line to `manifests/dataset_versions.jsonl` (Section 4).
 
```python
# src/store_preprocessed_data.py
SPLIT_TO_FOLDER = {
    "train": "processed/splits/train/train.parquet",
    "dev": "processed/splits/dev/dev.parquet",
    "test": "processed/splits/test/test.parquet",
    "held_out_test": "processed/splits/held_out/held_out_test.parquet",
    "future_update": "processed/splits/future_update/future_update.parquet",
}
 
# excerpt from main()
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
```

---

# 3. Database / Object Storage Decision — 0.5 point

The project will use **multiple complementary storage technologies**, rather than attempting to put everything into one database.

### GCS — authoritative object storage

GCS will store:

- immutable raw data;
- processed Parquet datasets;
- model artifacts;
- manifests and metadata.

This acts as the project's object-storage/data-lake layer.

### BigQuery — analytical layer

BigQuery is considered optional. The main reason for including it would be to demonstrate understanding. 
BigQuery is not required for scalability at the current dataset size. Its potential value is providing a managed SQL analytical layer and demonstrating integration between object storage and a cloud analytical warehouse.

However, the ~324,000 rows in the Parquet files could be analysed more simply with pandas in seconds. The trade-off for implementing BigQuery would be increased setup time, including dataset creation, load jobs, and IAM permissions.

BigQuery may be used where SQL-based exploration and analysis are useful.

A possible dataset is:

```text
pm25_dataset
├── canonical_hourly
├── station_reference
└── three_hour_aggregated
```

BigQuery is useful for:

- aggregation;
- coverage analysis;
- station comparisons;
- data-quality queries;
- exploratory analysis.

The project will **not** treat BigQuery as a replacement for the raw data lake.

BigQuery is not required at this dataset scale, local Parquet queries are sufficient. However, a final decision on whether or not to implement BigQuery will be made later in the project process, based on workload and educational value rather than dataset size alone.

### Why not sharding?

The current dataset is approximately 970,000 hourly observations and is expected to remain manageable as Parquet data for the project's batch-processing workload. Therefore, extensive manual sharding is not initially necessary.

If the dataset or processing workload grows substantially, Parquet partitioning by year, station or another appropriate key can be introduced without changing the raw-data architecture.

### Why not Cloud SQL/PostgreSQL?

The project does not require:

- transactional updates;
- application-level relational transactions;
- complex operational CRUD workloads.

Its main workloads are:

- historical batch ingestion;
- analytical queries;
- feature generation;
- model training.

Therefore, a transactional relational database would add complexity without addressing the project's main access pattern.

### Why not a feature store?

A dedicated feature store is not initially justified because the project is small enough for feature definitions and transformations to be managed in version-controlled pipeline code.

However, the project will explicitly address **training-serving consistency** in M4 by reusing the same feature-generation definitions during training and inference.

---

# 4. Data Versioning — 0.5 point

Environmental datasets can change because sources may publish revised or corrected historical measurements.

Therefore, dataset versioning will be explicit.

Each raw-file manifest entry will contain information such as:

| Field | Origin |
| :--- | :--- |
| **local_path** | The file's path on your machine |
| **gcs_path** | `raw/` plus the path under `raw_data/` |
| **checksum_sha256** | Hash of the file contents, computed before upload |
| **size_bytes** | File size |
| **file_modified_utc** | The file's last-modified time |
| **upload_timestamp_utc** | Time of upload |
| **source** | First folder: `epa`, `uk_air`, or `eea` |
| **pipeline_version** | `config.py` |
| **git_commit** | `config.py` |

These fields are written automatically by `upload_file()` in `upload_raw_data.py`, one JSON line per uploaded file (nothing is typed in by hand). `pipeline_version` and `git_commit` come from `config.py`, which reads the current commit hash and adds `-dirty` if tracked files have uncommitted changes.

```python
# src/upload_raw_data.py
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
```
 
```python
# src/config.py
def git_commit() -> str:
    """Hash of the code that is running, for the manifests. Gets a "-dirty" suffix
    if tracked files have uncommitted changes, and "unknown" outside a git repo."""
    try:
        # "git rev-parse HEAD" prints the hash of the current commit
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        if sha.returncode != 0 or not sha.stdout.strip():
            return "unknown"   # not a git repository
        # "status --porcelain" lists modified tracked files; untracked files are ignored,
        # so only uncommitted edits to committed files make the run "dirty"
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        return sha.stdout.strip() + ("-dirty" if changed else "")
    except Exception:   # git missing or timed out: never let this stop a pipeline run
        return "unknown"
```

Each run of the storage step (store_preprocessed_data.py) appends one line to
manifests/dataset_versions.jsonl, linking the processed dataset to the exact raw data and code that produced it:


| Field | Origin |
| :--- | :--- |
| dataset_run_id | UTC timestamp of the run |
| pipeline_version | config.py |
| preprocessing_version, feature_version | store_preprocessed_data.py |
| git_commit | config.py ("-dirty" if uncommitted changes) |
| raw_data_version | SHA-256 of manifests/raw_files.jsonl |
| station_reference_sha256 | SHA-256 of metadata/station_reference.csv |
| row_counts | Rows per split |
| config | Rolling window, target horizon, minimum valid hours, completeness threshold |
| outputs | Per file written: gcs_path, sha256, size_bytes, rows |
| created_timestamp_utc | Time of the run |

This record is built at the end of `main()` in `store_preprocessed_data.py`. `raw_data_version` is a fingerprint of the raw manifest, so two dataset runs with the same value used exactly the same raw files.

```python
# src/store_preprocessed_data.py
def raw_data_version(bucket):
    """SHA-256 of the raw-file manifest = a fingerprint of exactly which raw files were uploaded."""
    blob = bucket.blob(RAW_MANIFEST)
    if not blob.exists():
        print(f"WARNING: {RAW_MANIFEST} not found - run upload_raw_data.py first; raw_data_version will be null.")
        return None
    return hashlib.sha256(blob.download_as_text().encode("utf-8")).hexdigest()
 
# excerpt from main()
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
```

This provides a lineage relationship such as:

```text
Raw dataset version
        ↓
Preprocessing version
        ↓
Feature dataset version
        ↓
Training run
        ↓
Model version
```

The objective is to be able to answer:

> **Which exact data, preprocessing code and feature definitions produced this model?**

Raw files will not silently be overwritten when a source publishes revised data. Bucket versioning is enabled and each file's checksum is recorded.

---

# 5. Data Access — 1.0 point

The data pipeline will use Python to access the official sources.

### EEA

The EEA Air Quality Download Service will provide European monitoring time-series data and station metadata.

### EPA Ireland

EPA hourly monitoring data will provide Irish monitoring measurements.

### UK-AIR

UK-AIR automatic monitoring files will provide measurements for the UK stations.

The ingestion code will use Python tools including:

- `requests`
- `pandas`
- `NumPy`
- `PyArrow`
- Google Cloud Storage client libraries
- BigQuery client libraries (optional)

The ML pipeline will use:

- `scikit-learn`
- `xgboost`
- `torch` (optional)

GCS access will use Google Cloud authentication through service-account/application-default credentials. Credentials will not be embedded in source code.

In the code, the project ID and bucket name are read from environment variables and the storage client uses Application Default Credentials (for example after `gcloud auth application-default login`), so no credential is stored in the repository. If the variables are not set, the scripts stop with a message (`check_gcp_settings()` in `config.py`) instead of using placeholder values.
 
```python
# src/upload_raw_data.py (store_preprocessed_data.py uses the same pattern)
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your-unique-bucket-name")
 
def get_client() -> storage.Client:
    """Storage client for the configured project. Credentials come from Application Default
    Credentials (see the module docstring), never from the code."""
    return storage.Client(project=PROJECT_ID)
```

The project will primarily use **batch processing** for data collection and training-data generation.

This is appropriate because historical data is accumulated and processed in relatively large batches.

The future M4 system can additionally demonstrate new data arriving and triggering a batch update/retraining workflow. Full continuous streaming is not required for the initial training pipeline because the forecasting task does not require online learning at every individual measurement.


## Manual raw data download guide

The aim will be to add 3 scripts to automate the raw data download process for each of the 3 sources - UKAir, EEA, and EPA, however the processes are different for each source. For now we will simply recommend downloading the raw data from each of the 3 sources manually. 

Here are the guides to downloading the raw data from each source:

- UKAir (format CSV):

1. Go to interactive map here: https://uk-air.defra.gov.uk/interactive-map
2. Zoom in and click on relevant stations (London N. Kensington, London Bloomsbury, Belfast Centre, Derry)
3. Select CSV data files for this site
4. On the new page download the All Hourly Pollutant Data CSV file for each year between 2025 and 2018.

Note: You can find all relevant metadata for each station by clicking on the station and selecting site information.


- EEA Europe (format Parquet):

1. Go to this url: https://eeadmz1-downloads-webapp.azurewebsites.net/
2. Set filters. Country to DK, FR, GR, ES. Cities to Kobenhavn, Paris (Greater City), Athina, Madrid. Pollutants to PM2.5. Dataset to Primary validated data (E1a). Type to Hourly data. Note: if data is missing, try changing Type to Hourly/Daily data.
3. Fill in email.
4. Set Temporal coverage (start date and end date). 1 January 2018 to date of your choosing (31/12/2025 recommended. Keep 2026 for new data).
5. Select Download format Parquet and select Download under Download Actions
6. Unzip downloaded files

Note: EEA hourly files are converted to UTC+1 for every country while daily files are not. EPA and UK-AIR need no time-zone conversion, but both are hour-ending, and the pipeline shifts them back one hour. This is explicitly stated in the EEA and UKAir documentation, whereas it is simply inferred from an analysis of the data in the case of EPA data.

The Parquet files contain no station names, only a sampling-point code. For Denmark, France and Greece it contains the EU id (SPO-DK0034A_06001_104 is Copenhagen, DK0034A). The Spanish files use Spain's national code instead (SP_28079038_9_47 is Cuatro Caminos, ES1525A). Section 9 lists both codes.  For any new cities or stations added you will need to figure out the station id yourself. The full mapping is in metadata/station_reference.csv (columns station_id and source_download_id).

You can find the full EEA metadata and filter by country here https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements to find its unique id under EoI code and Nat code. This metadata also displays latitude/longitude and other relevant info and can be downloaded into CSV files. 
Even for station data downloaded from UKAir and EPA Ireland, it is recommended to download the EEA metadata CSV for the relevant country to get a consistent EU station id for each station, as well as other important metadata. This is due to its metadata generally being more complete and clear than EPA Ireland/AirQuality.ie in particular.

Alternatively this metadata can be found by checking each station in the city on this interactive map - https://www.eea.europa.eu/en/analysis/maps-and-charts/index . Click on the dot and then Show details. Beside the station name will be its unique EEA id. You can find longitude/latitude by clicking view station location which will bring you to its exact location on Google Maps.


- EPA Ireland (format CSV, downloaded from airquality.ie):

1. Go to this url: https://airquality.ie/readings
2. Select each station and repeat - Rathmines, Kilmainham, University College Cork, People’s Park Limerick, Paddy Browne’s Road Waterford, Briarhill Co. Galway (close to Galway city, data only available from December 2022).
3. Change from and to dates. Start with January 2022 and go up in 6-month increments. If any stations are missing PM2.5 data start at the earliest date that station does have PM2.5 data for. 
4. Click on the 3 bars beside the diagram and download the CSV file for each 6-month increment up to the date you wish to end (31/12/2025 recommended. Keep 2026 for new data).

Note: Irish stations have only two years of training data (2022–2023), providing substantially less temporal depth for time-aware cross-validation than the non-Irish stations. This is a known limitation caused by the lack of earlier hourly PM2.5 data for the selected Irish stations. The effect of this limitation on model performance will be evaluated empirically. 

You can find the easting/northing for each station at the bottom of the station page, e.g. https://airquality.ie/station/EPA-59 . This will need to be converted to latitude/longitude. Recommended to simply use the EEA metadata instead.


## Timestamp conventions used by the pipeline:

| Source | Label | Time zone | Action |
| :--- | :--- | :--- | :--- |
| EEA hourly | interval start | UTC+1 (EEA converts) | −1 h |
| UK-AIR | hour ending | GMT | −1 h |
| EPA | hour ending | UTC (both inferred) | −1 h |

---

# 6. Data Split / Validation Strategy — 2.0 points

This is one of the most important design decisions because the project uses **time-series data**.

A random train/test split is inappropriate because it could allow future observations to influence model development.

## Primary chronological split

```text
2018 ─────────────── 2023 | 2024 | 2025 | 2026
          TRAIN             DEV    TEST   UPDATE
```
Note: Train split is 2022-2023 for Irish stations.

### Training: 2018–2023 (2022-2023 for Irish stations)

Used for fitting model parameters.

Time-aware cross-validation may be performed inside this period for model/hyperparameter development.

### Development: 2024

Used for:

- comparing models;
- tuning hyperparameters;
- selecting features;
- early stopping;
- model-development decisions.

### Final test: 2025

The 2025 test set remains untouched until final evaluation.

It must not be used to decide:

- which model to use;
- which XGBoost hyperparameters to use;
- which feature set to use;
- how many neural-network layers to use.

### 2026 future-data demonstration

2026 is deliberately reserved for M4.

It represents newly arriving data and allows the project to demonstrate:

```text
New data
   ↓
Updated dataset
   ↓
Retraining
   ↓
Model v2
   ↓
Comparison with Model v1
```

This follows the lecture principle that a project should not immediately consume every available observation and then claim to simulate future data.

The approximate split will be:
- Non-Irish stations: 75% Training for years 2018-2023, 12.5% Development/Validation for year 2024, and 12.5% Test for year 2025. 
- Irish stations: 50% Training for years 2022-2023, 25% Development/Validation for year 2024, and 25% Test for year 2025.

---

## Geographic holdout

Three cities are held out:

- Derry
- London
- Paris

These cities are excluded from training and model selection. The 2025 observations from these stations will be used to evaluate generalisation to unseen cities.

This creates a second evaluation dimension:

```text
Temporal generalisation:
known cities → later years

Spatial generalisation:
known cities → completely unseen cities
```

The held-out cities must remain unseen across **all source systems**. The held-out cities must remain excluded from model development regardless of source.

---

## Cross-validation

Ordinary random k-fold cross-validation across the entire dataset will not be used.

If cross-validation is required, it will be **time-aware and restricted to the training period**.

The 2024 development set and 2025 final test set remain outside this process.

---

## Leakage prevention

The pipeline for learned preprocessing explicitly follows:

**Split → fit learned preprocessing on training data → transform other datasets**

rather than:

**Full dataset → preprocessing → split**

This applies to operations such as:

- numerical scaling;
- categorical encoding;
- learned feature transformations;
- hyperparameter selection.

For time-series features:

- lag features use only historical observations;
- rolling features use only historical observations;
- target observations cannot enter feature calculations;
- future measurements cannot be used to construct past features.

**Deterministic vs Learned preprocessing:**

Deterministic preprocessing, such as timestamp parsing, unit standardisation, station-ID mapping, duplicate detection, and fixed 3-hour aggregation rules, can be applied before the split because it does not learn parameters from the full dataset.

Learned preprocessing, such as feature scaling, imputation parameters, normalisation, or dimensionality reduction, must be fitted using the training data only and then applied unchanged to the development and test data.

### Split implementation (`create_splits.py`)
 
`src/create_splits.py` reads the feature table and gives every row a `split` label: `train`, `dev`, `test`, `held_out_test`, `future_update` or `excluded`.
 
- **Time:** the calendar year of a row decides its period (2024 = dev, 2025 = test, 2026 = future_update). Training starts in each station's first available year, taken from `availability_start` in `metadata/station_reference.csv` (never before 2018), which is how the Irish stations start in 2022 without a special rule.
- **Boundaries:** a row is only labelled if its input block and its target block (six hours later) fall in the same period. A row whose input is in late December 2023 but whose target is in January 2024 is `excluded`, so no training label comes from the development year. New Year inside the training period is not a boundary.
- **Place:** stations in `HELD_OUT_STATIONS` never enter train, dev or test. Their 2025 rows become `held_out_test` and all their other rows are `excluded`.
- **Safety checks:** the script stops if a held-out code is not in `station_reference.csv` (a typo), if a held-out station is missing from the feature table or has no 2025 rows, or if any held-out row ends up in train, dev or test.

```python
# src/create_splits.py
HELD_OUT_STATIONS = {
    "GB1060A",  # Derry Rosemount
    "GB0566A",  # London Bloomsbury
    "GB0620A",  # London N. Kensington
    "FR04002",  # Paris Gennevilliers
    "FR04058",  # Paris Saint-Denis
}
 
def period_of(year: pd.Series, train_start_year: pd.Series) -> pd.Series:
    """Which period a calendar year belongs to, for a given station."""
    # np.select takes the FIRST condition that is true: 2026+ is "future", then 2025 "test", then 2024 "dev";
    # any earlier year from the station's own start year onwards is "train"; older years are "before_study"
    return pd.Series(np.select(
        [year >= FUTURE_UPDATE_START_YEAR, year == TEST_YEAR, year == DEV_YEAR, year >= train_start_year],
        ["future", "test", "dev", "train"], default="before_study"), index=year.index)
 
def assign_splits(df: pd.DataFrame, train_start_year: pd.Series) -> pd.Series:
    """Return the split label of every row (the rules are in the module docstring)."""
    in_period = period_of(df["block_start_utc"].dt.year, train_start_year)           # period of the input block
    target_period = period_of(df["target_block_start_utc"].dt.year, train_start_year)  # period of the block predicted
    same_period = in_period == target_period  # False only for rows straddling a split boundary
    is_held_out = df["station_id"].isin(HELD_OUT_STATIONS)
 
    # np.select takes the FIRST condition that is true, so the ORDER below matters.
    conditions = [
        same_period & (in_period == "future"),                  # 2026 rows: kept for the update demo
        same_period & is_held_out & (in_period == "test"),      # an unseen city's 2025 rows: the unseen-city test
        is_held_out,                                            # every other row of an unseen city: unused
        same_period & (in_period == "test"),
        same_period & (in_period == "dev"),
        same_period & (in_period == "train"),
    ]
    choices = ["future_update", "held_out_test", "excluded", "test", "dev", "train"]
    # a row matching nothing (before the study, or input and target in different periods) is "excluded"
    return pd.Series(np.select(conditions, choices, default="excluded"), index=df.index)
```

---

# 7. Feature Description — 1.0 point

The initial candidate feature set is:

| **Variable** | **Definition** | **Type / Notes** |
|---|---|---|
| `pm25_current` | Mean PM2.5 of the input block, the 3-hour block that has just completed when the forecast is made | Continuous, µg/m³
| `pm25_lag_1` | Mean PM2.5 of the block immediately before the input block | Continuous, µg/m³
| `pm25_lag_2` | Mean PM2.5 two blocks before the input block | Continuous, µg/m³
| `pm25_lag_3` | Mean PM2.5 three blocks before the input block | Continuous, µg/m³
| `pm25_rolling_mean` | Recent historical mean PM2.5 | Continuous; historical observations only |
| `pm25_rolling_std` | Recent historical PM2.5 variability | Continuous; historical observations only |
| `time_block` | 3-hour period within the day, derived from station-local time (time zone in station_reference.csv) | Categorical/integer, 0–7 |
| `day_of_week` | Day of week | Categorical/integer |
| `month` | Calendar month | Categorical/integer |
| `season` | Meteorological season | Categorical |
| `city` | City containing the station | Categorical/metadata |
| `country` | Country containing the station | Categorical |
| `latitude` | Monitoring-station latitude | Continuous |
| `longitude` | Monitoring-station longitude | Continuous |
| `target_pm25` | Mean PM2.5 6 hours ahead | Continuous target |

The features are deliberately based primarily on **historical PM2.5 and time/station information**.

Weather variables are excluded from the initial system to keep the first version focused on the predictive information contained in the monitoring data itself.

PM10, NO2 and O3 are also excluded from the initial model.

### Important consideration: station/city identity

A station ID may be useful for predicting known stations but could allow a model to memorise location-specific behaviour.

Because unseen-city generalisation is a core research question, station and city identifiers will be treated carefully.

The final M2 analysis will determine whether they should be model features, metadata only, or represented in a way that does not prevent spatial generalisation.

Latitude and longitude are potentially more appropriate for a model intended to generalise to unseen locations.

### Feature extraction code (`feature_extraction.py`)
 
`src/feature_extraction.py` builds the feature table from the 3-hour data. Feature extraction is an optional step for the milestone, but it is included because the features and the 6-hour-ahead target are what the splits label (`create_splits.py` needs the `target_block_start_utc` column created here).
 
1. Each station is first placed on a complete 3-hour UTC grid (`ensure_full_grid()`), so a missing block is a visible gap rather than silently joining two distant blocks.
2. The time features are derived from each station's local time (`add_temporal_features()`).
3. The current block, three lags, rolling mean and standard deviation, and the target are added (`add_lag_rolling_and_target()`). The rolling window is 8 blocks (24 hours, including the input block) and the target is 2 blocks (6 hours) ahead.
Rows without a full history or a valid target are dropped; nothing is imputed.
 
```python
# src/feature_extraction.py
ROLLING_WINDOW_BLOCKS = 8   # 8 x 3h = 24h of history, including the current block
TARGET_HORIZON_BLOCKS = 2   # 2 x 3h = 6h ahead
 
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """time_block / day_of_week / month / season from each station's LOCAL time."""
    parts = []
    for tz_name, group in df.groupby("timezone"):   # one timezone at a time (each station has its own local time)
        g = group.copy()
        # drop tz info after converting: mixing several tz-aware zones in one
        # column would collapse it to dtype 'object' and break .dt below
        g["local_timestamp"] = g["block_start_utc"].dt.tz_convert(tz_name).dt.tz_localize(None)
        parts.append(g)
    df = pd.concat(parts).sort_index()   # join the per-timezone pieces back together
 
    df["time_block"] = df["local_timestamp"].dt.hour // 3   # 0 = 00:00-02:59 local ... 7 = 21:00-23:59
    df["day_of_week"] = df["local_timestamp"].dt.dayofweek  # 0 = Monday ... 6 = Sunday
    df["month"] = df["local_timestamp"].dt.month            # 1-12
    # meteorological seasons: Dec-Feb winter, Mar-May spring, Jun-Aug summer, Sep-Nov autumn
    season_map = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    df["season"] = df["month"].map(season_map)
    return df
 
def add_lag_rolling_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add, per station: the current block, 3 lags, a 24-hour rolling mean and std, and the target.
 
    Relies on the gap-free grid from ensure_full_grid(), so a shift of N rows is exactly N blocks.
    """
    df = df.sort_values(["station_id", "block_start_utc"])   # time order within each station
    df["pm25_current"] = df["pm25_3h_mean"]   # the block that has just finished: known when the forecast is made
    # each station's own series, so a lag never reaches from one station into another
    grouped = df.groupby("station_id")["pm25_3h_mean"]
 
    for k in (1, 2, 3):
        df[f"pm25_lag_{k}"] = grouped.shift(k)   # the value k blocks (3k hours) earlier
    # mean / std of the last 8 blocks including this one; min_periods=8 gives NaN unless all 8 blocks have a value
    df["pm25_rolling_mean"] = grouped.transform(
        lambda s: s.rolling(ROLLING_WINDOW_BLOCKS, min_periods=ROLLING_WINDOW_BLOCKS).mean())
    df["pm25_rolling_std"] = grouped.transform(
        lambda s: s.rolling(ROLLING_WINDOW_BLOCKS, min_periods=ROLLING_WINDOW_BLOCKS).std())
 
    # the grid is gap-free, so shifting by N rows is exactly N blocks
    df["target_pm25"] = grouped.shift(-TARGET_HORIZON_BLOCKS)   # the value 2 blocks (6 h) LATER: what the model predicts
    df["target_block_start_utc"] = df["block_start_utc"] + pd.Timedelta(hours=3 * TARGET_HORIZON_BLOCKS)
    return df
```

---

# 8. Data Types and Formats — 0.5 point

The project contains several types of structured/time-series data.

| **Data** | **Type** | **Example representation** |
|---|---|---|
| PM2.5 | Continuous numerical | `float32`/`float64` |
| Latitude/longitude | Continuous numerical | `float32`/`float64` |
| Time block | Discrete/categorical | integer 0–7 |
| Day of week | Categorical/discrete | integer/category |
| Month | Categorical/discrete | integer/category |
| Season | Categorical | string/category |
| City/country | Categorical | string/category |
| Station ID | Identifier | string |
| Timestamp | Time-series/time data | timezone-aware datetime |
| Valid Hours | Discrete numerical | integer 0-3 |
| Raw EPA data | Structured tabular | CSV |
| Raw UK-AIR data | Structured tabular | CSV |
| Raw EEA data | Structured time-series | Parquet |
| Processed data | Structured time-series | Parquet |
| Collection manifest | Semi-structured metadata | JSONL |
| Model configuration | Semi-structured metadata | JSON/YAML |

This distinction is important: **data type and storage format are not the same thing**.

For example, PM2.5 is numerical time-series data, while Parquet is a storage format used to represent that data efficiently.

---

# 9. Reproducibility of Data Collection — 1.0 point

Though the scripts are not completed yet, data collection will eventually be automated through version-controlled Python ingestion scripts.

The collection configuration will specify:

- source;
- URL/API endpoint;
- country;
- city;
- station;
- pollutant;
- start date;
- end date;
- source-specific parameters.

The current collection scope is:

### EPA Ireland

Provisional/unvalidated Irish PM2.5 data for:
- Kilmainham, Dublin (missing data between 30 Jan 2025 and 3 Jan 2026)
- Rathmines, Dublin
- University College Cork
- Briarhill, Co. Galway (located slightly outside Galway city with data available from December 2022 onwards)
- People's Park, Limerick
- Paddy Browne's Road, Waterford

**2022–2025**

Validated data would have been preferred, however there was only validated daily data for 5 of the 6 stations (2018-2021) and no hourly data. Unvalidated provisional data was chosen as it is readily available from approximately January 2022 onwards for all selected stations except Briarhill, Co. Galway which begins in December 2022. This choice was simply due to a lack of available verified data. EPA remains the main authority on Air Pollution levels in Ireland. These measurements are treated as provisional/unvalidated and will be subject to additional data-quality checks during M2. 

If it is determined later that the 2022-2023 Training data for Irish stations is not sufficient to accurately train our model, the daily data may be used, though this is unlikely. A more plausible potential use of the daily data could be for validation by comparing daily means of the EPA hourly readings with the validated daily values.

### UK-AIR

PM2.5 data for:

- Belfast Centre
- Derry
- London Bloomsbury
- London N. Kensington

**2018–2025**

### EEA

PM2.5 data for:

- Agia Paraskevi, Athens - EEA id: GR0039A
- Aristotelous, Athens - EEA id: GR0003A
- Copenhagen - EEA id: DK0034A (no hourly data for 2025)
- Cuatro Caminos, Madrid - EEA id: ES1525A (Parquet code: 28079038)
- Escuelas Aguirre, Madrid - EEA id: ES0118A (Parquet code: 28079008)
- Gennevilliers, Paris - EEA id: FR04002
- Saint-Denis, Paris - EEA id: FR04058

**2018–2025**

Exact station candidates were initially chosen by manual analysis of their suitability based on a combination of location, prioritising large cities, and sufficient available historical data. This decision was initially made prior to ingestion. After analysing the files, it was determined that Milan's 2 stations, London Westminster, and Athens Lykovrisi were lacking the required data to make them viable candidates. Thus, they were replaced by 2 Madrid stations, London N. Kensington, and Athens Aristotelous. All other original candidates were retained.

Every collection run (each run of upload_raw_data.py) records:

- local_path;
- gcs_path;
- checksum_sha256;
- size_bytes;
- file_modified_utc;
- upload_timestamp_utc;
- source;
- pipeline_version;
- git_commit;

This means another student can rerun the collection code using the documented configuration and identify exactly which source files were used.

Until the download scripts are completed, the download step is manual (guides in Section 5). The rest of collection is scripted: the downloaded files are placed under `raw_data/<source>/<raw_folder>/` (the `raw_folder` column of `metadata/station_reference.csv`) and uploaded by `upload_raw_data.py` (Section 1), which writes the manifest record above. `metadata/station_reference.csv` also holds the station selection listed above in machine-readable form and is read by every pipeline script (Section 10, Station mapping).

The ingestion scripts will not manually edit the downloaded source files.

A requirements.txt file will also be included to streamline and simplify any necessary installations. Note that torch and google-cloud-bigquery will be included in requirements.txt, even though the decision has not yet been made on whether these will be implemented. They are currently commented out for this reason as are some M4 requirements.

---

# 10. Reproducibility of Preprocessing — 1.5 points

The preprocessing pipeline will be implemented as version-controlled code and will transform the immutable raw data into the canonical and model-ready datasets.

The complete planned sequence is:

```text
Official raw files
        ↓
Parse source-specific formats
        ↓
Standardise column names / units / timestamps
        ↓
Match files to stations
        ↓
Detect duplicates/conflicting records
        ↓
Apply source quality rules
        ↓
Calculate station/year coverage
        ↓
Aggregate hourly → 3-hour measurements
        ↓
Construct 6-hour-ahead target
        ↓
Create historical lag features
        ↓
Create historical rolling features
        ↓
Create temporal features
        ↓
Add station/geographical features
        ↓
Create chronological/geographical splits
        ↓
Write versioned Parquet datasets
```

### Code overview and run order
 
Step 1 (downloading) is currently manual (Section 5); the download scripts are still in development and are not part of this milestone. The scripts in `src/` run in this order:
 
| Order | Script | What it does | Why it is included |
|---|---|---|---|
| 1 | `upload_raw_data.py` | Uploads the raw files to GCS and records the raw manifest | **Required: moves the raw data into storage** (Section 1) |
| 2 | `canonicalize_raw_data.py` | Parses the EPA, UK-AIR and EEA files into one hourly table keyed by `station_id` with UTC timestamps | The splits and stored datasets are built from its output, and it holds the timestamp and quality rules (this section) |
| 3 | `validate_data_quality.py` | Writes completeness, duplicate and value-sanity reports | Data-quality checks, kept and documented as part of the work done (Missing observations) |
| 4 | `aggregate_to_3hour.py` | Hourly to 3-hour mean blocks | The target and features are defined on these blocks (Aggregation) |
| 5 | `feature_extraction.py` | Lag, rolling and temporal features and the 6-hour-ahead target | The splits label its output (Section 7) |
| 6 | `create_splits.py` | Labels each row train / dev / test / held-out / future | **Required: train/dev/test split** (Section 6) |
| 7 | `store_preprocessed_data.py` | Writes all processed stages to GCS and the dataset manifest | **Required: stores the preprocessed data** (Section 2) |
 
The scripts marked **Required** are the ones the milestone asks for. The others are included because the required scripts depend on their output (the split can only label features, and features can only be built from cleaned 3-hour data) and because data validation is part of the work that was done; leaving them out would make the project impossible to reproduce.
 
Supporting files used by every script: `config.py` (folder paths, pipeline version, git commit), `stations.py` (loads and validates `metadata/station_reference.csv`) and `requirements.txt`.
 
To run the pipeline from the repository root (steps 1 and 7 need `GCP_PROJECT_ID` and `GCS_BUCKET_NAME` set as environment variables and Google Cloud application-default credentials):
 
```
python src/upload_raw_data.py
python src/canonicalize_raw_data.py
python src/validate_data_quality.py
python src/aggregate_to_3hour.py
python src/feature_extraction.py
python src/create_splits.py
python src/store_preprocessed_data.py
```

## Station mapping

Every station is keyed on its EU code (station_id) in metadata/station_reference.csv.

Station mapping will contain:

```text
station_id (EU id)
station_name
city
country
timezone
latitude
longitude
source
source_download_id
raw_folder
uk_air_internal_id
source_name_check
availability_start
coordinates_source
```

Station identity is handled by two small files: `metadata/station_reference.csv` (the table above) and `src/stations.py`, which loads it and stops with a complete list of problems if it is malformed (duplicate IDs, blank fields, unknown time zones, overlapping folders and so on). EPA and UK-AIR files do not contain the EU code, so `canonicalize_raw_data.py` matches each file to its station through the `raw_folder` column; EEA files are identified by the station code inside the file (`source_download_id`).
 
```python
# excerpt from load_station_reference() in src/stations.py
folders = raw.assign(raw_folder=raw["raw_folder"].str.replace("\\", "/", regex=False).str.strip("/ "))
for source, grp in folders.groupby("source"):
    pairs = list(zip(grp["station_id"], grp["raw_folder"]))
    for a_id, a in pairs:
        for b_id, b in pairs:
            if a_id != b_id and (a == b or b.startswith(a + "/")):
                problems.append(f"{source}: raw_folder {a!r} ({a_id}) overlaps {b!r} ({b_id}) - "
                                f"each station needs its own non-nested folder")
```
 
```python
# src/canonicalize_raw_data.py
def station_for_folder(file_path: Path, source_dir: Path, folders: "pd.Series"):
    """Which station's raw_folder contains this file? (None if none does.)
    `folders` maps station_id -> raw_folder (relative to source_dir, '/' separated)."""
    rel = file_path.relative_to(source_dir).parent.as_posix()   # the file's folder, relative to raw_data/<source>/
    rel = "" if rel == "." else rel                             # a file directly in the source folder has no sub-folder
    # a file belongs to a station if its folder IS that station's raw_folder or sits inside it
    hits = [sid for sid, folder in folders.items() if rel == folder or rel.startswith(folder + "/")]
    return hits[0] if len(hits) == 1 else None   # exactly one match, otherwise the caller reports the file
```

---

## Time standardisation

Source timestamps will be converted into a consistent time representation before temporal aggregation.

Time-zone handling will be explicitly recorded because the project uses measurements from multiple countries.
Source timestamps will be preserved exactly as provided in the raw data. EEA hourly data is preserved in UTC+1. During canonicalisation, the timestamp convention and timezone/offset associated with each source will be recorded. A consistent timezone representation will then be used for temporal alignment and aggregation, with daylight-saving transitions handled explicitly.

In `canonicalize_raw_data.py` every source is converted to the same convention: hourly timestamps in UTC, labelled by the start of the hour. UK-AIR and EPA files are labelled by the end of the hour and EEA hourly files are labelled in UTC+1, so one hour is subtracted in each case. UK-AIR writes the last hour of each day as `24:00`, which is handled by adding the clock time to the date as a duration.

```python
# src/canonicalize_raw_data.py
# Timestamp conventions. A row stamped 09:00 under "hour ending" covers
# 08:00-09:00; we shift to interval-START so all sources match.
UKAIR_TIMESTAMPS_ARE_HOUR_ENDING = True  # stated in every UK-AIR file header: "GMT hour ending"
EPA_TIMESTAMPS_ARE_HOUR_ENDING = True    # INFERRED, not documented: files start each day at 01:00.
#   Could be confirmed by comparing with validated EEA data for the same station.
# EEA: Start/End bound each hour (already interval-start) but the service
# labels them UTC+1 for every country, so we subtract 1h to reach true UTC.
EEA_LABEL_UTC_OFFSET_HOURS = 1
 
# UK-AIR labels each day's last hour "24:00", which datetime parsing rejects,
# so build the timestamp as (date at midnight) + (HH:MM as a duration);
# "24:00" then correctly becomes the next day's 00:00.
day = pd.to_datetime(df[date_col].astype(str), format="%d-%m-%Y", errors="coerce")
clock = pd.to_timedelta(df[time_col].astype(str).str.strip() + ":00", errors="coerce")
timestamp = day + clock
if UKAIR_TIMESTAMPS_ARE_HOUR_ENDING:
    timestamp = timestamp - pd.Timedelta(hours=1)
```

---

## Missing observations

Monitoring stations may have:

- missing hours;
- outages;
- incomplete days;
- station changes;
- provisional observations;
- revised observations.

Some obvious patterns have already been identified by an initial analysis of the data. This is why the initial proposals of 2 Milan stations, London Westminster, and Athens Lykovrisi were rejected and replaced with 2 Madrid stations, London N. Kensington, and Athens Aristotelous. These observations will be analysed and scrutinised more thoroughly during M2.

An initial target of approximately 75% valid PM2.5 coverage per year will be used as a data-quality criterion. This is a predefined quality target rather than a result of the model evaluation.

A block is valid with at least 2 of 3 hourly values. A feature row needs the input block, all 3 lags, all 8 rolling blocks and the target to be valid with nothing imputed. The 75% figure is a diagnostic. Failing station-years are flagged, not removed.

These rules are implemented in two places. `canonicalize_raw_data.py` applies the source quality rules: EEA rows are valid only if their `Validity` flag is 1, 2 or 3 (invalid rows carry placeholder values such as -9999 and become missing), empty EPA placeholder rows stamped off the hour are dropped, and if two files give different values for the same station-hour the run stops instead of choosing one. `validate_data_quality.py` is an extra quality-check step: it measures the share of valid hours per station and year against the hours expected from `availability_start` and writes its reports to `processed_data/quality_reports/`. It is a diagnostic only; nothing is removed.
 
```python
# src/canonicalize_raw_data.py
# EEA Validity codes we accept. 1 = valid. 2 = valid but below the detection
# limit (the measured value is given). 3 = valid, below the detection limit,
# value replaced by half the limit. Everything else (-1, -99, ...) is invalid
# and carries a sentinel value (0, -999, -9999) that must never be used.
EEA_VALID_CODES = (1, 2, 3)
 
# excerpt from parse_eea_file()
value = pd.to_numeric(df[col["value"]], errors="coerce")
if "validity" in col:
    validity = pd.to_numeric(df[col["validity"]], errors="coerce")  # handles "1" vs 1
    invalid = ~validity.isin(EEA_VALID_CODES)
    below_limit = validity.isin((2, 3))
    if invalid.any():
        print(f"  {path.name}: {int(invalid.sum())} invalid rows (Validity not in {EEA_VALID_CODES}) set to NaN.")
    if below_limit.any():
        print(f"  {path.name}: kept {int(below_limit.sum())} readings flagged valid-but-below-detection-limit.")
    value = value.where(~invalid)
```
 
```python
# src/validate_data_quality.py
COMPLETENESS_THRESHOLD = 0.75   # a station-year with a smaller share of valid hours is flagged (not removed)
 
def check_completeness(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Share of hourly values that are valid, for every station and year it should have data.
 
    How many hours to EXPECT comes from availability_start in station_reference.csv, never
    from the data itself. Years below COMPLETENESS_THRESHOLD are flagged, not removed.
    """
    df = df.drop_duplicates(["station_id", "timestamp_utc"], keep="first")   # a repeated hour must not count twice
    rows = []
    for _, st in stations.iterrows():
        avail_start = max(STUDY_START, pd.Timestamp(st["availability_start"], tz="UTC"))   # first hour we expect data
        s_df = df[df["station_id"] == st["station_id"]]   # this station's rows only
        for year in range(avail_start.year, STUDY_END.year + 1):
            # the part of this year inside the expected period (shorter for the first and last year)
            win_start = max(avail_start, pd.Timestamp(f"{year}-01-01 00:00", tz="UTC"))
            win_end = min(STUDY_END, pd.Timestamp(f"{year}-12-31 23:00", tz="UTC"))
            expected = int((win_end - win_start).total_seconds() // 3600) + 1   # hourly values that window should hold
            in_win = s_df[(s_df["timestamp_utc"] >= win_start) & (s_df["timestamp_utc"] <= win_end)]
            valid = int(in_win["pm25"].notna().sum())   # hours that actually have a PM2.5 value
            rows.append({"station_id": st["station_id"], "station_name": st["station_name"], "year": year,
                         "expected_from": win_start.date(), "valid_hours": valid, "expected_hours": expected,
                         "completeness_pct": round(100 * valid / expected, 1),
                         "passes_threshold": valid / expected >= COMPLETENESS_THRESHOLD})
    report = pd.DataFrame(rows)   # one row per station-year
    failing = report[~report["passes_threshold"]]
    if failing.empty:
        print(f"All expected station-years reach {COMPLETENESS_THRESHOLD:.0%} completeness.")
    else:
        print(f"{len(failing)} expected station-year(s) below {COMPLETENESS_THRESHOLD:.0%} "
              f"(0% usually means not downloaded yet):")
        print(failing[["station_id", "station_name", "year", "valid_hours", "expected_hours", "completeness_pct"]]
              .to_string(index=False))
    print("Note: this is a diagnostic - the station set is fixed and nothing is removed automatically.")
    return report
```

---

## Aggregation

Hourly measurements will be converted into **3-hour mean PM2.5 concentrations**.

The aggregation rule defines:

- the three-hour boundaries;
- the minimum number of valid hourly observations required;
- how missing hours are handled;
- how invalid/quality-flagged measurements are treated.

The project will not silently fill missing target observations.

Implemented in `aggregate_to_3hour.py`: blocks are fixed 3-hour windows on the UTC clock, and a block's mean is kept only if at least 2 of its 3 hours are valid. The number of valid hours is stored in `valid_hours`. This script is included because the target and all features are defined on these blocks.

```python
# src/aggregate_to_3hour.py
MIN_VALID_HOURS = 2  # out of 3, per block
 
def aggregate_station(group: pd.DataFrame) -> pd.DataFrame:
    """Turn ONE station's hourly series into 3-hour mean blocks.
 
    Blocks are fixed UTC windows (00:00-03:00, 03:00-06:00, ...). A block's mean
    is kept only if at least MIN_VALID_HOURS of its 3 hours have a value.
    """
    group = group.set_index("timestamp_utc").sort_index()   # time-indexed, so resample() can cut it into windows
    if group.index.has_duplicates:   # the same hour twice means canonicalize_raw_data.py left a duplicate
        raise SystemExit("Duplicate hours found in the canonical file - re-run canonicalize_raw_data.py, "
                         "which is responsible for resolving them.")
    resampled = group["pm25"].resample("3h")   # fixed 3-hour windows, each labelled by its START time
    # mean() ignores missing hours; count() is how many hours in the window have a value (0-3)
    block_mean, valid_hours = resampled.mean(), resampled.count()
    out = pd.DataFrame({"block_start_utc": block_mean.index,
                        "pm25_3h_mean": block_mean.values,
                        "valid_hours": valid_hours.values})
    # too few valid hours: the block gets no mean (missing values are never filled in)
    out.loc[out["valid_hours"] < MIN_VALID_HOURS, "pm25_3h_mean"] = float("nan")
    return out
```

---

## Target construction

The raw data is first aggregated into **3-hour PM2.5 averages**.

For each completed 3-hour input block, the model predicts the PM2.5 average for the 3-hour block **six hours later**.

For example:

```text
Input block             Target block
09:00–11:59     →       15:00–17:59
```

A target is only created when the required hourly measurements are available according to the predefined completeness rule.

The target variable is therefore:

**`target_pm25` = 3-hour mean PM2.5 concentration in the target block.**

Aggregation blocks are fixed 3-hour windows on the UTC clock, so a target two blocks ahead is always exactly six hours later, even on daylight-saving change days. Times in the examples are therefore UTC. The time_block feature is derived from station-local time: the local hour at the start of the block divided by 3, rounded down, so 0 means local 00:00–02:59 and 7 means 21:00–23:59. A block lines up exactly with its time_block only where local time is a whole multiple of 3 hours from UTC (Ireland and the UK in winter, Athens in summer). Elsewhere, such as Paris in winter, the UTC block 09:00–11:59 is 10:00–12:59 local and takes the label of the bin containing its start.

The time_block feature is 0-7 arranged thus:

| `time_block` | Time period |
|---:|---|
| 0 | 00:00–02:59 |
| 1 | 03:00–05:59 |
| 2 | 06:00–08:59 |
| 3 | 09:00–11:59 |
| 4 | 12:00–14:59 |
| 5 | 15:00–17:59 |
| 6 | 18:00–20:59 |
| 7 | 21:00–23:59 |

The target and the time features are created in `feature_extraction.py` (`add_lag_rolling_and_target()` and `add_temporal_features()`, shown in Section 7).

---

## Lag features

The model also uses previous 3-hour PM2.5 averages as input features.

For example, for the 09:00–11:59 input block:

```text
Previous blocks                              Input        Target

00:00–02:59   03:00–05:59   06:00–08:59   09:00–11:59   15:00–17:59
     ↓             ↓             ↓             ↓            ↓
   lag_3         lag_2         lag_1      pm25_current     target
```

The lag features therefore describe recent historical PM2.5 concentrations available before the target period.
The forecast is issued after the input block has completed, so its mean (pm25_current) is a known input; lag_1 is the block before it.

No measurements from the target block or any later period are used to create the lag features.

Implemented in `add_lag_rolling_and_target()` in `feature_extraction.py` (code in Section 7): lag k is the block shifted back by k rows, which is exactly k blocks because the grid has no gaps.

---

## Rolling features

Rolling features summarise recent historical PM2.5 measurements.

The initial features will include:

* a recent rolling mean;
* a recent rolling standard deviation.

Only observations available before the target period are used.

The rolling-window length will be treated as a configuration parameter and selected using the training/development data. The final 2025 test data will not be used to choose the window length.

Implemented in `add_lag_rolling_and_target()` in `feature_extraction.py` (code in Section 7). The default window is 8 blocks (24 hours, including the input block).

---

## Feature preprocessing

Where models require transformations such as:

- standardisation;
- categorical encoding;
- numerical imputation;

the transformation parameters will be fitted using training data only.

The same fitted transformation will then be applied to development and test data.

This prevents the preprocessing stage itself from leaking information from the evaluation datasets.

---

## Model-ready representations

The same underlying processed data will support multiple models.

### Linear Regression

Uses the engineered tabular feature representation.

### XGBoost

Uses the engineered tabular feature representation and allows nonlinear feature interactions.

### PyTorch GRU/LSTM

This is an optional addition if time allows after XGBoost is working and evaluated.

Uses historical sequences of observations, with the exact sequence length selected during M2/M3.

The model-specific representation is therefore produced **after** the common canonical data pipeline rather than requiring separate, manually maintained datasets.

---

# Dataset Size and Suitability

The course requires at least 10,000 learning samples.

This project is expected to exceed that requirement substantially.

The proposed dataset currently consists of 17 monitoring stations across Ireland and Europe. 5 Irish stations are expected to provide hourly PM2.5 observations from 2022–2025, with a 6th Irish station expected to provide data from December 2022 onwards. 6 non-Irish training stations and 5 held-out test stations are expected to provide hourly observations from 2018–2025. 

Before accounting for missing observations and data-quality restrictions, this represents approximately 970,000 hourly station-time observations. Aggregating the hourly measurements into 3-hour means gives a theoretical maximum of approximately 324,000 3-hour station-time observations. 

The final number of supervised learning samples will be lower after applying completeness requirements, quality checks, feature-history requirements and the 6-hour forecasting horizon. This should provide sufficient temporal and geographical diversity for investigating both forecasting performance and generalisation to unseen cities, while still remaining manageable for CPU-based classical ML and a modest PyTorch sequence model.

---

# Legal and Data-Use Considerations

The project uses environmental monitoring data rather than personal information.

The planned measurements do not require names, email addresses, personal identifiers or other obvious personally identifiable information.

The main legal considerations are therefore:

- checking the licence/terms associated with each official source;
- respecting the source's conditions for reuse;
- retaining source attribution;
- preserving source metadata;
- not redistributing data where the source terms prohibit it;
- securing GCS access appropriately.

Because the project uses official environmental monitoring data, personal-data risks are expected to be substantially lower than for datasets containing individuals. The source licence and reuse terms have been checked before final publication or redistribution.

---

# Overall Architecture

The project follows the lecture's general AI data architecture:

```text
 EPA / EEA / UK-AIR
        ↓
  Raw data sources
        ↓
  GCS / Data Lake
        ↓
  Batch ingestion
        ↓
Canonical processing
        ↓
     Parquet
        ↓
┌──────────────────────┐
│ Analytical layer     │
│ pandas / Parquet     │
│ BigQuery (optional)  │
└──────────────────────┘
        ↓
 Feature / sequence construction
        ↓
┌────────────┬───────────┬──────────────────┐
│   Linear   │  XGBoost  │ PyTorch GRU/LSTM │
│ Regression │           │    (optional)    │
└────────────┴───────────┴──────────────────┘
        ↓
      Model
        ↓
 Docker / FastAPI
        ↓
    Cloud Run
```

The architecture deliberately uses different technologies for different workloads rather than treating one database or storage system as the solution to every problem.

---

# Key Design Principles

### 1. Data before model

The project treats data collection, quality, storage, transformation and lineage as fundamental parts of the AI system rather than focusing only on model selection.

### 2. Storage follows workload

GCS is used for durable object storage and batch access, Parquet for efficient analytical data access, and BigQuery for SQL-based analysis (if implemented, otherwise pandas may be used for analysis).

### 3. Raw data is preserved

The raw source data is retained so that preprocessing can be changed and reproduced without recollecting the data.

### 4. Data type is separated from storage format

PM2.5 is numerical time-series data; Parquet is a representation/storage format.

### 5. Batch processing is appropriate for training

Historical environmental measurements can be collected and transformed in batches. Real-time/streaming processing is not necessary for the initial training-data pipeline.

### 6. Future data remains unseen

2025 is reserved as the final temporal test set, while 2026 is reserved for demonstrating new-data/model updates.

### 7. Leakage prevention is explicit

Source-independent canonicalisation and deterministic transformations are performed before splitting. Any transformation that learns parameters from the data is fitted using training data only.

The following preprocessing needs to occur before the split:
- parsing timestamps
- unit standardisation
- station mapping
- identifying duplicates
- source-quality filtering
- hourly → 3-hour aggregation

Historical lag and rolling features are then constructed using only observations available before each forecast time, with no future target information included.

### 8. Data lineage matters

The project aims to link:

**raw data → preprocessing → feature dataset → training run → model**

so that the origin of a deployed model can be reconstructed.

### 9. More complex models are not assumed to be better

Persistence, Linear Regression, XGBoost and PyTorch (optional) sequence modelling will be compared empirically.

### 10. Generalisation matters

The project does not evaluate only performance on cities seen during training.

Derry, London and Paris are held out to investigate geographical generalisation.

---

# Main Questions to Resolve in Milestone 2

The following decisions are intentionally left open until the actual data has been analysed:

1. Is the initial rule of at least 2 of 3 valid hours appropriate?
2. Which rolling-window lengths are most appropriate?
3. How much missing data can be tolerated?
4. How should negative readings be treated?
5. How should stations with no 2025 data be handled in evaluation?
6. Should station ID/city be used as model features or retained only as metadata?
7. What historical sequence length should be supplied to the PyTorch model (if we choose to go ahead with PyTorch)?
8. Which XGBoost hyperparameters provide the best development-set performance without overfitting?
9. Does XGBoost outperform Linear Regression sufficiently to justify its additional complexity?
10. Does the PyTorch sequence model provide additional predictive value beyond XGBoost (if we choose to go ahead with PyTorch)?
11. Should time of day use time_block or a continuous local hour?

These decisions will be made using the training/development data and documented as part of the reproducible M2 pipeline rather than being selected retrospectively using the final test set.

---
# Setup Guide (not necessary for peer-marking):

## Running the pipeline on your own Google Cloud bucket

This guide takes you from a fresh computer to a finished run of the pipeline, with the data uploaded to your own Google Cloud Storage bucket. The download scripts are not used: the raw files are downloaded by hand, as described in Section 5.

**Which terminal am I using?** The few commands that differ between terminals are given twice, once for **PowerShell** and once for **bash** (Git Bash on Windows, the macOS Terminal, Linux, or WSL). Use the block that matches your terminal:
- PowerShell: the prompt starts with `PS C:\...>`.
- bash: the prompt ends with `$` (Git Bash also shows `MINGW64`). VS Code's terminal can be either; its tab shows which.

If a command fails with "command not found", you are probably typing the other kind of command. The `git`, `python`, `pip` and `gcloud` commands are the same in both.

### Step 0 — Get the code and the raw data ready

**0.1 Install Python and Git (once).**
- Python 3.10 or newer, from python.org. On Windows, tick "Add python.exe to PATH" in the installer.
- Git, from git-scm.com.
- Check both in a new terminal window: `python --version` and `git --version` (Mac/Linux: `python3 --version`).

**0.2 Get the code.**
```
git clone <repository URL>
cd dh-PM2.5-forecast
```
The folder is named after the repository. It contains the scripts (`src/`), `metadata/station_reference.csv` and `requirements.txt`. It does not contain the raw data: you download that in 0.3. If your copy already has a `raw_data/` folder full of files, skip to 0.6.

**0.3 Download the raw data.** Follow the "Manual raw data download guide" in Section 5 and let the files save to your Downloads folder as normal. When you have finished, you should have:

| Source | What you should have downloaded | What the files look like |
|---|---|---|
| UK-AIR | 8 yearly CSVs (2018–2025) for each of London N. Kensington, London Bloomsbury, Belfast Centre and Derry: 32 files | Site code, underscore, year: `KC1_2018.csv`, `BEL2_2018.csv` |
| EEA | One download for Denmark, France, Greece and Spain, as a zip. Unzip it. | Many `.parquet` files, one per sampling point, covering every station in the selected cities. The station code is in the name: `SPO-GR0003A_06001_100.parquet`, `SP_28079038_9_47.parquet` |
| EPA | For each of the 6 Irish stations, one CSV per six-month period from January 2022 to December 2025 (about 8 per station) | Named only by date range: `010122-020722.csv`. **The name does not say which station the file is for.** |

**0.4 Create the folders.** The scripts find each file by the folder it sits in, so the folder names must match exactly, capital letters included. Inside the repository folder you should end up with this:
```
dh-PM2.5-forecast/
├── src/                                  (the scripts)
├── metadata/station_reference.csv        (already in the repository)
└── raw_data/                             (you create this)
    ├── epa/
    │   ├── cork/
    │   ├── dublin/
    │   │   ├── dublinKilmainham/
    │   │   └── dublinRathmines/
    │   ├── galway/
    │   ├── limerick/
    │   └── waterford/
    ├── uk_air/
    │   ├── belfast/
    │   ├── derry/
    │   └── london/
    │       ├── londonBloomsbury/
    │       └── londonKensington/
    └── eea/
        ├── athens/
        │   ├── athensAristotelous/
        │   └── athensParaskevi/
        ├── copenhagen/
        ├── madrid/
        │   ├── madridCuatroCaminos/
        │   └── madridEscuelasAguirre/
        └── paris/
            ├── parisGennevilliers/
            └── parisSaintDenis/
```
To create all 17 folders at once, run this from the repository folder.

PowerShell:
```
New-Item -ItemType Directory -Force -Path @(
  "raw_data\epa\cork",
  "raw_data\epa\dublin\dublinKilmainham",
  "raw_data\epa\dublin\dublinRathmines",
  "raw_data\epa\galway",
  "raw_data\epa\limerick",
  "raw_data\epa\waterford",
  "raw_data\uk_air\belfast",
  "raw_data\uk_air\derry",
  "raw_data\uk_air\london\londonBloomsbury",
  "raw_data\uk_air\london\londonKensington",
  "raw_data\eea\athens\athensAristotelous",
  "raw_data\eea\athens\athensParaskevi",
  "raw_data\eea\copenhagen",
  "raw_data\eea\madrid\madridCuatroCaminos",
  "raw_data\eea\madrid\madridEscuelasAguirre",
  "raw_data\eea\paris\parisGennevilliers",
  "raw_data\eea\paris\parisSaintDenis"
) | Out-Null
```

bash:
```
mkdir -p raw_data/epa/cork raw_data/epa/dublin/dublinKilmainham raw_data/epa/dublin/dublinRathmines \
  raw_data/epa/galway raw_data/epa/limerick raw_data/epa/waterford \
  raw_data/uk_air/belfast raw_data/uk_air/derry raw_data/uk_air/london/londonBloomsbury \
  raw_data/uk_air/london/londonKensington \
  raw_data/eea/athens/athensAristotelous raw_data/eea/athens/athensParaskevi raw_data/eea/copenhagen \
  raw_data/eea/madrid/madridCuatroCaminos raw_data/eea/madrid/madridEscuelasAguirre \
  raw_data/eea/paris/parisGennevilliers raw_data/eea/paris/parisSaintDenis
```

**0.5 Put each file in its station's folder.**

| Station | Folder inside `raw_data/` | Which files go there |
|---|---|---|
| Dublin Rathmines | `epa/dublin/dublinRathmines` | The EPA CSVs you downloaded for Rathmines |
| Dublin Kilmainham | `epa/dublin/dublinKilmainham` | The EPA CSVs you downloaded for Kilmainham |
| Cork UCC | `epa/cork` | The EPA CSVs you downloaded for University College Cork |
| Limerick People's Park | `epa/limerick` | The EPA CSVs you downloaded for People's Park |
| Waterford Paddy Browne's Road | `epa/waterford` | The EPA CSVs you downloaded for Paddy Browne's Road |
| Galway Briarhill | `epa/galway` | The EPA CSVs you downloaded for Briarhill |
| Belfast Centre | `uk_air/belfast` | Files with `BEL2_` in the name |
| Derry Rosemount | `uk_air/derry` | Files with `DERR_` in the name |
| London Bloomsbury | `uk_air/london/londonBloomsbury` | Files with `CLL2_` in the name |
| London N. Kensington | `uk_air/london/londonKensington` | Files with `KC1_` in the name |
| Athens Agia Paraskevi | `eea/athens/athensParaskevi` | `.parquet` files with `GR0039A` in the name |
| Athens Aristotelous | `eea/athens/athensAristotelous` | `.parquet` files with `GR0003A` in the name |
| Copenhagen | `eea/copenhagen` | `.parquet` files with `DK0034A` in the name (three files; one is a daily file that the pipeline skips but keeps) |
| Madrid Cuatro Caminos | `eea/madrid/madridCuatroCaminos` | `.parquet` files with `28079038` in the name |
| Madrid Escuelas Aguirre | `eea/madrid/madridEscuelasAguirre` | `.parquet` files with `28079008` in the name |
| Paris Gennevilliers | `eea/paris/parisGennevilliers` | `.parquet` files with `FR04002` in the name |
| Paris Saint-Denis | `eea/paris/parisSaintDenis` | `.parquet` files with `FR04058` in the name |

You can drag and drop in File Explorer, or use these commands from the repository folder.

*UK-AIR* (the site code is in each file name, so one command per station moves them all):

PowerShell:
```
$dl = "$HOME\Downloads"
Move-Item "$dl\*KC1_*.csv"  "raw_data\uk_air\london\londonKensington\"
Move-Item "$dl\*CLL2_*.csv" "raw_data\uk_air\london\londonBloomsbury\"
Move-Item "$dl\*BEL2_*.csv" "raw_data\uk_air\belfast\"
Move-Item "$dl\*DERR_*.csv" "raw_data\uk_air\derry\"
```

bash:
```
dl=~/Downloads
mv "$dl"/*KC1_*.csv  raw_data/uk_air/london/londonKensington/
mv "$dl"/*CLL2_*.csv raw_data/uk_air/london/londonBloomsbury/
mv "$dl"/*BEL2_*.csv raw_data/uk_air/belfast/
mv "$dl"/*DERR_*.csv raw_data/uk_air/derry/
```

*EEA* (this copies rather than moves, because the unzipped folder also holds files for stations the project does not use; replace the folder name with the one you unzipped):

PowerShell:
```
$eea = "$dl\<name of the unzipped EEA folder>"
Get-ChildItem $eea -Recurse -Filter "*GR0039A*.parquet"  | Copy-Item -Destination "raw_data\eea\athens\athensParaskevi\"
Get-ChildItem $eea -Recurse -Filter "*GR0003A*.parquet"  | Copy-Item -Destination "raw_data\eea\athens\athensAristotelous\"
Get-ChildItem $eea -Recurse -Filter "*DK0034A*.parquet"  | Copy-Item -Destination "raw_data\eea\copenhagen\"
Get-ChildItem $eea -Recurse -Filter "*28079038*.parquet" | Copy-Item -Destination "raw_data\eea\madrid\madridCuatroCaminos\"
Get-ChildItem $eea -Recurse -Filter "*28079008*.parquet" | Copy-Item -Destination "raw_data\eea\madrid\madridEscuelasAguirre\"
Get-ChildItem $eea -Recurse -Filter "*FR04002*.parquet"  | Copy-Item -Destination "raw_data\eea\paris\parisGennevilliers\"
Get-ChildItem $eea -Recurse -Filter "*FR04058*.parquet"  | Copy-Item -Destination "raw_data\eea\paris\parisSaintDenis\"
```

bash:
```
eea="$dl/<name of the unzipped EEA folder>"
find "$eea" -name "*GR0039A*.parquet"  -exec cp {} raw_data/eea/athens/athensParaskevi/ \;
find "$eea" -name "*GR0003A*.parquet"  -exec cp {} raw_data/eea/athens/athensAristotelous/ \;
find "$eea" -name "*DK0034A*.parquet"  -exec cp {} raw_data/eea/copenhagen/ \;
find "$eea" -name "*28079038*.parquet" -exec cp {} raw_data/eea/madrid/madridCuatroCaminos/ \;
find "$eea" -name "*28079008*.parquet" -exec cp {} raw_data/eea/madrid/madridEscuelasAguirre/ \;
find "$eea" -name "*FR04002*.parquet"  -exec cp {} raw_data/eea/paris/parisGennevilliers/ \;
find "$eea" -name "*FR04058*.parquet"  -exec cp {} raw_data/eea/paris/parisSaintDenis/ \;
```

*EPA* (the file names do not identify the station, so sort these as you go): download one station at a time, and drag its files into that station's folder before you download the next station. If all the CSVs are already mixed together in Downloads, open each one to see which station it is before filing it.

Rules for all files:
- Keep the files exactly as downloaded. Do not rename, edit, merge, or open and re-save them (in Excel, for example): the raw files must stay unchanged.
- Do not leave extra files in `raw_data/`: no merged CSVs, no duplicate copies such as `KC1_2018 (1).csv`, and no folders for stations the project does not use.
- The pipeline catches most misplaced files, but not all:
  - A file in a folder that is not in the structure above stops the run, and the message names the file.
  - A UK-AIR file in the wrong station's folder stops the run, because the station name inside each file is checked.
  - An EEA file in the wrong station's folder stops the run. An EEA file for a station the project does not use is skipped with a warning.
  - EPA files are the exception: they do not contain the station name. A misplaced EPA file is caught only if it overlaps, and disagrees with, another file for the same station (the run then stops with "DIFFERENT PM2.5 values"). Check the EPA folders by hand.

**0.6 Check the result.** From the repository folder:

PowerShell:
```
Get-ChildItem raw_data -Recurse -File | Group-Object DirectoryName | Select-Object Count, Name | Format-Table -AutoSize
(Get-ChildItem raw_data -Recurse -File).Count
```

bash (the first line prints the total, the loop prints the file count of each folder):
```
find raw_data -type f | wc -l
for d in $(find raw_data -type d | sort); do n=$(find "$d" -maxdepth 1 -type f | wc -l); [ "$n" -gt 0 ] && echo "$n  $d"; done
```
You should see 17 folders. In the project's own dataset:
- Each UK-AIR folder holds 8 files.
- The EPA folders hold 8 files each, except Kilmainham and Galway with 7 (Kilmainham has no data after 29 January 2025, and Galway starts in December 2022).
- Each EEA folder holds 1 file, except Copenhagen with 3.
- The total is 87 files (46 EPA + 32 UK-AIR + 9 EEA).

Note the total: the upload in Step 7 prints the same number. Differences are fine if you downloaded a different date range, but every folder must contain files.

### Step 1 — Create a Google Cloud project

- In console.cloud.google.com, create or choose a project and copy its **Project ID**. It looks like `my-project-123456` and is not the display name.
- The project needs billing linked (the free trial works) and you need to be its owner. That covers creating the bucket and turning on versioning.

### Step 2 — Install the Google Cloud CLI and sign in

Install it from cloud.google.com/sdk/docs/install, then open a new terminal window and run these four commands, in this order:
```
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
gcloud services enable storage.googleapis.com
```
The third command creates the credentials the Python scripts use. They are kept outside the repository. Setting the project first lets it attach the right project to them.

### Step 3 — Decide the bucket name (you do not create it yourself)

You do **not** create the bucket in the Cloud Console. `upload_raw_data.py` creates it automatically the first time it runs, in the EU region with object versioning switched on. If the bucket already exists, the script uses it.

Your only job in this step is to decide the name. It must be unique across all of Google Cloud and use only lowercase letters, numbers and hyphens (3 to 63 characters), for example `my-pm25-forecast-bucket`. You give the name to the script in Step 5. It may be the same text as your project ID: project IDs and bucket names are separate things.

### Step 4 — Create the Python environment

From the repository folder (where Step 0 left you):
PowerShell:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again.

bash:
```
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```
On macOS, Linux and WSL the activate script is `source .venv/bin/activate` and the Python command is `python3`. When the environment is active, the prompt starts with `(.venv)`.

### Step 5 — Tell the scripts your project ID and bucket name

The scripts need your project ID (Step 1) and your bucket name (Step 3). They read them from two **environment variables**: named settings that a terminal window holds and hands to every program you start from that window.

- There is no file to create. A `.env` file is not used, and neither script reads one.
- Type the two lines into the **same terminal window** you will run the scripts from in Step 7, pressing Enter after each. Nothing is printed when it works.
- They last until you close that window (in VS Code, each terminal tab is separate). Type them again in any new window.

Replace the values with your own.

PowerShell:
```
$env:GCP_PROJECT_ID = "your-project-id"
$env:GCS_BUCKET_NAME = "your-bucket-name"
```

bash:
```
export GCP_PROJECT_ID="your-project-id"
export GCS_BUCKET_NAME="your-bucket-name"
```
In bash there must be **no spaces around `=`** and no `$env:`. `$env:` is PowerShell-only: in bash it fails with `env:GCP_PROJECT_ID: command not found`.

Check that they are set, in the same window. It should print your value (nothing printed means the variable is not set in this window):
- PowerShell: `echo $env:GCP_PROJECT_ID`
- bash: `echo $GCP_PROJECT_ID`

### Step 6 — Final check before running

- Run `git status` and commit anything pending, otherwise the manifest records the commit as `-dirty` (harmless but untidy).
- Make sure `raw_data/` still holds the file count from Step 0.6.

### Step 7 — Run the pipeline

From the repository folder:
```
python src/upload_raw_data.py
python src/canonicalize_raw_data.py
python src/validate_data_quality.py
python src/aggregate_to_3hour.py
python src/feature_extraction.py
python src/create_splits.py
python src/store_preprocessed_data.py
```
Steps 2–6 of this list run entirely on your own machine and do not need Google Cloud. Run `upload_raw_data.py` before `store_preprocessed_data.py`.

**What a correct run prints**, for the project's own dataset (the file counts and row counts differ if your files differ):
- **upload:**
  - `Created bucket: gs://…`
  - `87 new, 0 updated, 0 unchanged (skipped).`
  - `Manifest updated: 87 total entries.`
  - `Counts match.`
- **canonicalize:**
  - It skips the daily Copenhagen file (`SPO-DK0034A_06001_103.parquet`).
  - `Coverage: 17 / 17 stations`
  - `Saved 943899 canonical hourly rows`
- **validate:**
  - Four lines ending "OK".
  - `5 expected station-year(s) below 75%`: Kilmainham 2025, Bloomsbury 2021 and 2022, Copenhagen 2021 and 2025. They are flagged, not removed.
  - 5,663 negative values.
  - 7 values above 500.
- **aggregate:** `318791 3-hour blocks, 19701 failed the 2/3-hour rule`
- **features:** `285551 remain`, 22 columns
- **splits:**
  - train 123,364, dev 31,192, test 25,921, held_out_test 13,362 and excluded 91,712.
  - `Leakage check passed`
  - Held-out rows per station: FR04002=2,304, FR04058=2,734, GB0566A=2,892, GB0620A=2,880, GB1060A=2,552.
- **store:** it uploads the canonical, 3-hour and feature tables, four split files and the station table, then `Manifest entry appended`. `No rows for split 'future_update' - skipping upload.` is normal, because there is no 2026 data yet.

### Step 8 — Check it worked

```
gcloud storage ls -r gs://YOUR_BUCKET/
gcloud storage cat gs://YOUR_BUCKET/manifests/dataset_versions.jsonl
```
- **Folders:** you should see `raw/`, `processed/` (canonical, three_hour, features, splits), `manifests/` and `metadata/station_reference.csv`.
  - There is no `future_update/` or `models/` folder yet. Cloud Storage does not keep empty folders, so planned folders appear only once files go in.
  - The scripts do not upload `metadata/source_metadata/`. To add it, run `gcloud storage cp --recursive metadata/source_metadata gs://YOUR_BUCKET/metadata/`.
- **Versioning:** in the Cloud Console, open Cloud Storage, then Buckets, then your bucket, then Protection. "Object versioning" should say Enabled.
- **Skip logic:** run `python src/upload_raw_data.py` a second time. It should print `0 new, 0 updated, 87 unchanged (skipped)` and add no manifest lines.
- **Re-running the store step:** this is safe. It appends one more manifest line, and the old copies of the files are kept by versioning.

### If something fails

| Message | Fix |
|---|---|
| `GCP_PROJECT_ID and GCS_BUCKET_NAME are not set` | Set them again in this window (Step 5). They are forgotten when a terminal window closes. |
| `command not found` when you set them, for example `env:GCP_PROJECT_ID: command not found` | You typed PowerShell syntax in a bash terminal, or put spaces around `=`. In bash use `export GCP_PROJECT_ID="..."` with no spaces (Step 5). |
| `DefaultCredentialsError` | Run `gcloud auth application-default login`. |
| `403` or `409` on the first upload call | The bucket name is probably taken by someone else, or your account lacks Storage Admin on the project. Try another name and check the project ID. |
| Warning about a "quota project" | It is harmless. To silence it, run `gcloud auth application-default set-quota-project YOUR_PROJECT_ID`. |
| `STOPPED - these files could not be used` | The message lists each file and the reason, usually a file in a folder that is not in the structure, or a UK-AIR or EEA file in the wrong station's folder (Step 0.5). |
| `STOPPED: no usable data for selected station(s)` | A folder is missing, empty, or named differently from the structure in Step 0.4. |
| `STOPPED: … DIFFERENT PM2.5 values` | Two files disagree for the same station and hour, often an EPA file in the wrong station's folder or the same download twice with edits. |
| PowerShell will not run the activate script | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first. |

### Cost and clean-up

The bucket holds roughly 100 MB, which should cost pennies or less per month. Each extra store run adds old versions. When you have finished, you can delete the bucket in the Cloud Console or just leave it.