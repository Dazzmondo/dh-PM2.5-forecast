# Six-Hour Urban PM2.5 Forecasting Across Ireland and Europe

## Project Overview

This project develops an end-to-end AI/ML system for forecasting **3-hour mean PM2.5 concentration six hours ahead** at urban air-quality monitoring stations.

The project treats the problem primarily as a **time-series and structured-data ML problem**. Historical PM2.5 observations are collected from official environmental monitoring sources, stored in cloud object storage, transformed into a common canonical dataset, analysed and converted to model features, and then used to train and evaluate several models.

The main research question is:

> **Can machine learning accurately forecast PM2.5 concentrations six hours ahead across different Irish and European cities using historical air-quality measurements?**

Two secondary questions are:

1. Can a model trained across several cities generalise to a city not seen during training?
2. Does including geographically and environmentally diverse cities improve or affect generalisation?

The second question is a hypothesis to be tested rather than an assumption that greater diversity will necessarily improve performance.

The initial modelling progression is:

**Persistence baseline → Linear Regression → XGBoost → PyTorch GRU/LSTM**

This allows simple, interpretable, nonlinear tabular, and explicit sequence-based approaches to be compared.

---

# Milestone 1 Rubric

| **Criterion** | **Points** | **Project specification** |
|---|---:|---|
| **1. Raw data storage** | **1.0** | Raw source data will be stored unchanged in **Google Cloud Storage (GCS)** under source-specific `raw/` paths. GCS is appropriate because the raw datasets are historical files that need durable, scalable object storage and repeated batch access rather than transactional updates. |
| **2. Processed data storage & file formats** | **1.0** | Processed canonical measurements and model features will be stored in **Parquet** in GCS. Parquet is appropriate for columnar analytical access to structured time-series data. Raw files remain in their original source formats. |
| **3. Database / object storage decision** | **0.5** | GCS will be the authoritative object/data-lake layer. **BigQuery** will provide an analytical/query layer for SQL exploration and data-quality analysis. A transactional SQL database is not required because the project has no transactional workload. |
| **4. Data versioning** | **0.5** | Source files will be identified by source, period, filename, download timestamp, checksum and pipeline version. Dataset versions will be linked to the raw-data version, preprocessing configuration and code version to provide data lineage. |
| **5. Data access** | **1.0** | Python ingestion scripts will access official EEA, EPA Ireland and UK-AIR sources, download raw files and upload them to GCS. Training/preprocessing code will read versioned Parquet data from GCS, with BigQuery used for analytical queries. Google Cloud authentication will use service-account/application-default credentials rather than embedded credentials. |
| **6. Data split / validation strategy** | **2.0** | A chronological train/dev/test strategy will be used: **2018–2023 training, 2024 development, 2025 final test**. 2026 is reserved for a future-data/model-update demonstration. Derry, London and Paris are geographical holdouts and are excluded from model development. Time-aware CV may be used within 2018–2023. Future information, test data and target-period observations will not influence training or feature construction. |
| **7. Feature description** | **1.0** | Features include historical PM2.5 lags and rolling statistics, temporal variables, station type and geographical information, with an optional nearby-station PM2.5 feature. Each feature is constructed only from information available before the forecast target. |
| **8. Data types and formats** | **0.5** | The dataset contains numerical continuous data (PM2.5, coordinates), categorical data (station type, city, country), integer temporal variables, timestamps, station identifiers and quality flags. Raw sources include Excel, CSV and Parquet; processed data will use Parquet; manifests use JSONL. |
| **9. Reproducibility of data collection** | **1.0** | Collection is implemented in version-controlled Python ingestion scripts. Source, URL/API parameters, country/city/station selection, pollutant, date range, filename, timestamp, checksum and pipeline version are recorded in a manifest. |
| **10. Reproducibility of preprocessing** | **1.5** | Raw data will be parsed, standardised, mapped to canonical stations, checked for duplicates and quality issues, aggregated into 3-hour measurements, converted into six-hour-ahead targets, and transformed into historical lag/rolling/temporal features. The exact configuration, completeness rules and dataset version will be recorded so the processed dataset can be regenerated from the raw data. |

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
│   ├── epa/
│   ├── uk_air/
│   │   ├── belfast/
│   │   ├── derry/
│   │   └── london/
│   └── eea/
│       ├── ireland/
│       ├── france/
│       ├── italy/
│       ├── denmark/
│       └── greece/
│
├── processed/
├── models/
└── manifests/
```

The raw files will be preserved **unchanged**. Processing will create separate derived datasets rather than overwriting the source data.

This follows the data-lake principle from the lectures: retaining the raw data means that if a preprocessing decision is later found to be incorrect, the dataset can be regenerated without recollecting everything.

GCS is appropriate because the raw data is primarily accessed through **batch processing**, is not transactional, and needs durable, scalable object storage.

The raw data sources are:

- **European Environment Agency (EEA)** Air Quality Download Service
- **Environmental Protection Agency Ireland (EPA)** validated air-quality archive
- **UK-AIR** automatic monitoring data

---

# 2. Processed Data Storage and File Formats — 1.0 point

Processed data will be stored separately from the raw data in GCS.

The main processed format will be **Apache Parquet**.

Planned structure:

```text
processed/
├── canonical/
│   └── canonical_measurements.parquet
│
├── features/
│   └── pm25_features.parquet
│
└── splits/
    ├── train/
    ├── dev/
    └── test/
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

CSV will still be retained where it is the original UK-AIR source format.

The distinction between **data type and storage format** is important:

- PM2.5 is numerical time-series data.
- Station type is categorical data.
- A timestamp represents temporal data.
- These data types can be represented using CSV, Parquet, database tables or other formats.

Therefore, the choice of Parquet is a storage/I/O decision rather than a claim that PM2.5 is itself a "Parquet data type."

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

BigQuery will be used where SQL-based exploration and analysis are useful.

A possible dataset is:

```text
pm25_dataset
├── canonical_measurements
├── station_metadata
└── three_hour_measurements
```

BigQuery is useful for:

- aggregation;
- coverage analysis;
- station comparisons;
- data-quality queries;
- exploratory analysis.

The project will **not** treat BigQuery as a replacement for the raw data lake.

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

```text
source
source_url
period
filename
gcs_path
download_timestamp
sha256
pipeline_version
```

A processed dataset will additionally be associated with:

```text
raw_data_version
preprocessing_version
feature_version
git_commit
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

Raw files will not silently be overwritten when a source publishes revised data.

---

# 5. Data Access — 1.0 point

The data pipeline will use Python to access the official sources.

### EEA

The EEA Air Quality Download Service will provide European monitoring time-series data and station metadata.

### EPA Ireland

EPA validated monitoring archives will provide Irish monitoring measurements.

### UK-AIR

UK-AIR automatic monitoring files will provide measurements for the UK stations.

The ingestion code will use Python tools including:

- `requests`
- `pandas`
- `NumPy`
- `PyArrow`
- Google Cloud Storage client libraries
- BigQuery client libraries

The ML pipeline will use:

- `scikit-learn`
- `xgboost`
- `torch`

GCS access will use Google Cloud authentication through service-account/application-default credentials. Credentials will not be embedded in source code.

The project will primarily use **batch processing** for data collection and training-data generation.

This is appropriate because historical data is accumulated and processed in relatively large batches.

The future M4 system can additionally demonstrate new data arriving and triggering a batch update/retraining workflow. Full continuous streaming is not required for the initial training pipeline because the forecasting task does not require online learning at every individual measurement.

---

# 6. Data Split / Validation Strategy — 2.0 points

This is one of the most important design decisions because the project uses **time-series data**.

A random train/test split is inappropriate because it could allow future observations to influence model development.

## Primary chronological split

```text
2018 ─────────────── 2023 | 2024 | 2025 | 2026
        TRAIN             DEV     TEST    UPDATE
```

### Training: 2018–2023

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

---

## Geographic holdout

Three cities are held out:

- Derry
- London
- Paris

These cities are excluded from training and model selection.

This creates a second evaluation dimension:

```text
Temporal generalisation:
known cities → later years

Spatial generalisation:
known cities → completely unseen cities
```

The held-out cities must remain unseen across **all source systems**.

For example, London data from the EEA cannot accidentally enter training merely because UK-AIR London data was excluded.

---

## Cross-validation

Ordinary random k-fold cross-validation across the entire dataset will not be used.

If cross-validation is required, it will be **time-aware and restricted to the training period**.

The 2024 development set and 2025 final test set remain outside this process.

---

## Leakage prevention

The pipeline explicitly follows:

**Split → fit preprocessing on training data → transform other datasets**

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
- nearby-station features use only historical observations;
- target observations cannot enter feature calculations;
- future measurements cannot be used to construct past features.

---

# 7. Feature Description — 1.0 point

The initial feature set is:

| **Variable** | **Definition** | **Type / Notes** |
|---|---|---|
| `pm25_lag_1` | Most recent completed 3-hour mean PM2.5 | Continuous, µg/m³ |
| `pm25_lag_2` | PM2.5 two 3-hour blocks previously | Continuous, µg/m³ |
| `pm25_lag_3` | PM2.5 three 3-hour blocks previously | Continuous, µg/m³ |
| `pm25_rolling_mean` | Recent historical mean PM2.5 | Continuous; historical observations only |
| `pm25_rolling_std` | Recent historical PM2.5 variability | Continuous; historical observations only |
| `time_block` | 3-hour period within the day | Categorical/integer, 0–7 |
| `day_of_week` | Day of week | Categorical/integer |
| `month` | Calendar month | Categorical/integer |
| `season` | Meteorological season | Categorical |
| `station_type` | Monitoring-station classification | Categorical |
| `city` | City containing the station | Categorical/metadata |
| `country` | Country containing the station | Categorical |
| `latitude` | Monitoring-station latitude | Continuous |
| `longitude` | Monitoring-station longitude | Continuous |
| `nearby_pm25` | Historical PM2.5 from a manually selected nearby station | Optional, continuous |
| `target_pm25` | Mean PM2.5 six hours ahead | Continuous target |

The features are deliberately based primarily on **historical PM2.5 and time/station information**.

Weather variables are excluded from the initial system to keep the first version focused on the predictive information contained in the monitoring data itself.

PM10, NO2 and O3 are also excluded from the initial model.

### Important consideration: station/city identity

A station ID may be useful for predicting known stations but could allow a model to memorise location-specific behaviour.

Because unseen-city generalisation is a core research question, station and city identifiers will be treated carefully.

The final M2 analysis will determine whether they should be model features, metadata only, or represented in a way that does not prevent spatial generalisation.

Latitude, longitude and station type are potentially more appropriate for a model intended to generalise to unseen locations.

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
| Station type | Categorical | string/category |
| City/country | Categorical | string/category |
| Station ID | Identifier | string |
| Timestamp | Time-series/time data | timezone-aware datetime |
| Quality flags | Boolean/categorical | boolean |
| Raw EPA data | Structured tabular | Excel |
| Raw UK-AIR data | Structured tabular | CSV |
| Raw EEA data | Structured time-series | Parquet |
| Processed data | Structured time-series | Parquet |
| Collection manifest | Semi-structured metadata | JSONL |
| Model configuration | Semi-structured metadata | JSON/YAML |

This distinction is important: **data type and storage format are not the same thing**.

For example, PM2.5 is numerical time-series data, while Parquet is a storage format used to represent that data efficiently.

---

# 9. Reproducibility of Data Collection — 1.0 point

Data collection will be automated through version-controlled Python ingestion scripts.

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

Validated Irish PM2.5 data:

**2020–2024**

### UK-AIR

PM2.5 data for:

- Belfast
- Derry
- London

**2018–2025**

### EEA

PM2.5 data for:

- Ireland
- France
- Italy
- Denmark
- Greece

**2018–2025**

The exact station set will be finalised during M2 after coverage and quality analysis.

Every collection run records:

- source;
- requested parameters;
- original filename;
- download timestamp;
- GCS destination;
- checksum;
- pipeline version.

This means another student can rerun the collection code using the documented configuration and identify exactly which source files were used.

The ingestion scripts will not manually edit the downloaded source files.

---

# 10. Reproducibility of Preprocessing — 1.5 points

The preprocessing pipeline will be implemented as version-controlled code and will transform the immutable raw data into the canonical and model-ready datasets.

The complete planned sequence is:

```text
Official raw files
        ↓
Parse source-specific formats
        ↓
Select PM2.5
        ↓
Standardise column names
        ↓
Standardise units
        ↓
Standardise timestamps/time zones
        ↓
Map source station IDs → canonical station IDs
        ↓
Detect duplicates/conflicting records
        ↓
Apply source quality rules
        ↓
Calculate station/year coverage
        ↓
Select eligible stations
        ↓
Aggregate hourly → 3-hour measurements
        ↓
Construct six-hour-ahead target
        ↓
Create historical lag features
        ↓
Create historical rolling features
        ↓
Create temporal features
        ↓
Add station/geographical features
        ↓
Optionally add nearby-station feature
        ↓
Create chronological/geographical splits
        ↓
Write versioned Parquet datasets
```

## Station mapping

Different data providers may use different station identifiers.

A canonical station mapping will therefore contain:

```text
canonical_station_id
source
source_station_id
city
country
latitude
longitude
station_type
```

If the same physical station appears in multiple sources, the source-specific IDs will be mapped to a single canonical identifier.

This is important because combining EPA and EEA data without station mapping could create duplicate observations.

---

## Time standardisation

Source timestamps will be converted into a consistent time representation before temporal aggregation.

Time-zone handling will be explicitly recorded because the project uses measurements from multiple countries.

---

## Missing observations

Monitoring stations may have:

- missing hours;
- outages;
- incomplete days;
- station changes;
- provisional observations;
- revised observations.

These will be identified during M2.

The initial station-eligibility target is approximately **75% valid PM2.5 coverage per year**, subject to the actual data-quality analysis.

The exact minimum number of valid hourly observations required to create a 3-hour mean will be specified in the preprocessing configuration before the final dataset is produced.

---

## Aggregation

Hourly measurements will be converted into **3-hour mean PM2.5 concentrations**.

The aggregation rule will explicitly define:

- the three-hour boundaries;
- the minimum number of valid hourly observations required;
- how missing hours are handled;
- how invalid/quality-flagged measurements are treated.

The project will not silently fill missing target observations.

---

## Target construction

The target is:

> **The 3-hour mean PM2.5 concentration six hours ahead.**

For example:

```text
08:00–10:59  →  14:00–16:59
    INPUT          TARGET
```

A target will only be created when the required future measurements satisfy the predefined completeness rule.

---

## Lag features

Lag features are generated exclusively from completed historical 3-hour blocks.

For example:

```text
t-9h       t-6h       t-3h        t
  │          │          │          │
lag_3      lag_2      lag_1      latest

                             ↓
                        prediction
                             ↓
                            t+6h
```

No future target observations can enter these features.

---

## Rolling features

Rolling means and standard deviations will use historical observations only.

The window length will be treated as a configuration parameter and will be selected using the development/training process rather than by inspecting the final test performance.

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

Uses historical sequences of observations, with the exact sequence length selected during M2/M3.

The model-specific representation is therefore produced **after** the common canonical data pipeline rather than requiring separate, manually maintained datasets.

---

# Dataset Size and Suitability

The course requires at least 10,000 learning samples.

This project is expected to exceed that requirement substantially.

There are approximately 23,000 possible 3-hour periods per station over an eight-year period.

With approximately 25–40 eligible stations, the theoretical number of station-time observations is on the order of **580,000–930,000** before accounting for missing data and coverage restrictions.

The actual number of usable training examples will be determined by the M2 quality audit.

The final dataset therefore should provide substantially more than the minimum 10,000 learning samples while still remaining manageable for CPU-based classical ML and a modest PyTorch sequence model.

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

Because the project uses official environmental monitoring data, personal-data risks are expected to be substantially lower than for datasets containing individuals, but the source licence and reuse terms will still be checked before final publication or redistribution.

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
 ┌───────────────┐
 │   BigQuery    │
 │   Analytics   │
 └───────────────┘
        ↓
 Feature / sequence construction
        ↓
 ┌───────────┬──────────┬────────────┐
 │   Linear  │ XGBoost  │  PyTorch   │
 │ Regression│          │  GRU/LSTM  │
 └───────────┴──────────┴────────────┘
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

GCS is used for durable object storage and batch access, Parquet for efficient analytical data access, and BigQuery for SQL-based analysis.

### 3. Raw data is preserved

The raw source data is retained so that preprocessing can be changed and reproduced without recollecting the data.

### 4. Data type is separated from storage format

PM2.5 is numerical time-series data; Parquet is a representation/storage format.

### 5. Batch processing is appropriate for training

Historical environmental measurements can be collected and transformed in batches. Real-time/streaming processing is not necessary for the initial training-data pipeline.

### 6. Future data remains unseen

2025 is reserved as the final temporal test set, while 2026 is reserved for demonstrating new-data/model updates.

### 7. Leakage prevention is explicit

Temporal features, preprocessing and model selection are designed so that future information cannot influence training.

### 8. Data lineage matters

The project aims to link:

**raw data → preprocessing → feature dataset → training run → model**

so that the origin of a deployed model can be reconstructed.

### 9. More complex models are not assumed to be better

Persistence, Linear Regression, XGBoost and PyTorch sequence modelling will be compared empirically.

### 10. Generalisation matters

The project does not evaluate only performance on cities seen during training.

Derry, London and Paris are held out to investigate geographical generalisation.

---

# Main Questions to Resolve in Milestone 2

The following decisions are intentionally left open until the actual data has been analysed:

1. Which specific monitoring stations satisfy the coverage requirements?
2. What minimum hourly completeness should be required for a 3-hour observation?
3. Which rolling-window lengths are most appropriate?
4. How much missing data can be tolerated?
5. How should source quality flags be handled?
6. How should overlapping EPA/EEA observations be reconciled?
7. Should station ID/city be used as model features or retained only as metadata?
8. Which nearby stations provide useful additional information?
9. What historical sequence length should be supplied to the PyTorch model?
10. Which XGBoost hyperparameters provide the best development-set performance without overfitting?
11. Does XGBoost outperform Linear Regression sufficiently to justify its additional complexity?
12. Does the PyTorch sequence model provide additional predictive value beyond XGBoost?

These decisions will be made using the training/development data and documented as part of the reproducible M2 pipeline rather than being selected retrospectively using the final test set.