# Model card: six-hour urban PM2.5 forecast

**Project:** Six-Hour Urban PM2.5 Forecasting Across Ireland and Europe
**Repository:** https://github.com/Dazzmondo/dh-PM2.5-forecast
**Author and contact:** [Insert: name]

> Drafting conventions (delete this block before the file is final): every `[Insert: ...]` is a result or a fact that must come from a
> logged MLflow run, a model folder or a manifest (name the source). Everything else states only facts already verified in the
> repository or in the Milestone 2 evidence. No slot may be left in the final file, and no number may be typed in that is not in the
> exported run tables (`reports/m3/run_summary.csv`) or the final evaluation table (`reports/final/`).

---

## 1. Model details

| Field | Content |
|---|---|
| Model | [Insert: name and version, e.g. XGBoost v2; the folder `models/<model>/v<N>/` and the same path in the bucket] |
| Type | [Insert: gradient-boosted trees (XGBoost) / GRU / ridge regression: the selected model, and why it was selected: section 5 of `docs/M3.md`] |
| Task | Regression: the mean PM2.5 concentration (µg/m³) of the 3-hour block that starts six hours after the current block, for one monitoring station |
| Forecast origin | A fixed 3-hour UTC block that has just finished; the forecast is for the block two blocks later (target = block t+2) |
| Output | One number per station and block (a point forecast; no uncertainty interval is produced) |
| Framework and versions | Python 3.14.2; xgboost 3.4.1; scikit-learn 1.9.1; torch 2.15.0 (GRU); pandas 3.0.6; numpy 2.5.3; pyarrow 25.0.1 (exact versions in `requirements.txt`) |
| Code | commit [Insert: hash of the commit that trained the model]; the training run's MLflow run ID [Insert]; container image [Insert: image ID or registry digest] |
| Trained on | [Insert: date], hardware [Insert: machine type, CPU/GPU] |
| Reference model | Persistence (the current block is the forecast): MAE 3.869 µg/m³, RMSE 7.99 on the development year (Milestone 2) |

## 2. Intended use

- **Purpose.** A course experiment on whether a learned model can forecast urban PM2.5 six hours ahead better than carrying the current
  value forward, and whether such a model works in cities it was not trained on. The intended setting of the project is an urban
  air-quality network that issues forecasts; the model is **not** used for any real warning.
- **Intended users.** Students and reviewers of the course project, and anyone studying the experiment and its code.
- **Appropriate use.** Reproducing and extending the experiment; comparing other models against the same rows (the 31,192 development
  rows) and the same persistence baseline; studying how forecast skill varies by station, season, hour and concentration.
- **Out-of-scope use.**
  - Health, regulatory, public-warning or operational decisions of any kind.
  - Forecast horizons other than six hours, averaging periods other than 3 hours, or pollutants other than PM2.5.
  - Stations or settings unlike the 17 stations in the study (for example, roadside or industrial sites, or countries not represented
    in the training data), without a new evaluation.
  - Use of a forecast without the persistence comparison and the stated errors: the model under-predicts high concentrations (see
    section 8) and gives no uncertainty.

## 3. Training data

| Item | Content |
|---|---|
| Source stations | 17 urban monitoring stations in Ireland, the United Kingdom, Denmark, Spain, Greece and France; 12 are development stations in 9 cities (Cork, Dublin ×2, Galway, Limerick, Waterford, Belfast, Copenhagen, Madrid ×2, Athens ×2) |
| Not in training | 5 held-out stations in 3 cities (Derry, London ×2, Paris ×2): never used to fit, tune or select anything |
| Sources | UK-AIR (UK Department for Environment, Food and Rural Affairs), the European Environment Agency air-quality download service, and the Irish EPA (airquality.ie); licence and ethics details in Milestone 2 section 2 |
| Period | 2018–2023 for training (the Irish stations only 2022–2023, from their first available data); 2024 development; 2025 test; 2026 reserved |
| Size | 123,364 training rows (one row = one station at one 3-hour block); 31,192 development rows |
| Dataset version | dataset run ID 20261019T213934Z (`manifests/dataset_versions.jsonl`), raw-data version `3d01f5e7…b3bd`, pipeline v2; the features file has SHA-256 `b2cd1f6e…c4` |
| Preprocessing | Hourly readings converted to UTC; 3-hour blocks on a fixed UTC grid, a block needs at least 2 of its 3 hours; nothing is imputed or interpolated; rows before 2018 dropped; values above 500 µg/m³ kept only if sustained and corroborated (the 7 found were all set to missing); negative readings kept in storage and clipped at zero only inside a model's own step; a row exists only if the current block, three earlier blocks, the 24-hour rolling mean and standard deviation, and the target are all valid |
| Inputs | [Insert: the feature set used, listed: current block, lags, rolling mean/std, the seasonal lag and/or the raw window, and the time features (sin/cos of the local hour at the block midpoint and of the month)] |
| Not used as inputs | Station identity, city, country, latitude and longitude (tested only as a named ablation); weather and other pollutants (outside the project's scope) |
| Scaling | Fitted inside the model on the training rows; for the Ridge model verified equal to the stored `scalers.json` (section 7 of `docs/M3.md`) |

## 4. Architecture and configuration

[Insert: for the selected model, the exact hyperparameters from its `config.yaml`: for XGBoost the number of rounds (from early stopping),
learning rate, depth, minimum child weight, subsample, column subsample, L1/L2 penalties and gamma; for the GRU the hidden size, layers,
dropout, learning rate, batch size, epochs and loss. Name the search that chose them (random search or TPE, seeds, number of trials) and
the MLflow run.]

## 5. Training and evaluation procedure

- **Selection folds.** Three expanding folds inside the training years (validation years 2021, 2022, 2023); a training row's target
  must end before the validation year starts. The decision metric is the MAE on seen-station rows (stations with earlier training rows),
  averaged with equal weight over the folds. A variant was adopted only if it improved that MAE by at least 1% in at least 2 of the 3
  folds and the 2024 MAE was not worse; otherwise the simpler option was kept.
- **Development year.** Fit on 2018–2023, score on 2024 (31,192 rows; all 12 stations seen).
- **Final evaluation, once.** The frozen model was scored once on the 2025 test split (stations seen in training) and on the held-out
  cities (never seen). No refit on 2024, so the model has not seen 2024.
- **Seeds and hardware.** Seeds [Insert: 0, 1, 2 or as used]; [Insert: machine type, training time, peak memory].
- **Reproducibility.** Section 7 of `docs/M3.md`: pinned versions, container image, configuration file, dataset and run IDs, and the
  clean re-run result [Insert].

## 6. Metrics

MAE (primary, in µg/m³, because it is in the units of the target and not dominated by a few episodes) and RMSE (secondary, because it
penalises missed episodes more); the mean signed error (bias; negative means under-prediction); and the skill score
1 − MAE_model / MAE_persistence. All are reported overall and per station with the row count, with and without Cork UCC.

## 7. Performance

All numbers come from logged runs (`reports/m3/run_summary.csv`, `reports/final/`).

| Set | Rows | MAE | RMSE | Persistence MAE | Skill | Stations beating persistence |
|---|---|---|---|---|---|---|
| Selection folds (mean of 3, seen-station rows) | [Insert] | [Insert] | [Insert] | [Insert] | [Insert] | n/a |
| Development year 2024 | 31,192 | [Insert] | [Insert] | 3.869 | [Insert] | [Insert] / 12 |
| 2025 test split | [Insert] | [Insert] | [Insert] | [Insert] | [Insert] | [Insert] |
| Held-out cities (never seen) | [Insert] | [Insert] | [Insert] | [Insert] | [Insert] | [Insert] |

[Insert: per-station table for the development year and for 2025 (station, city, rows, MAE, RMSE, bias, persistence MAE, skill), and
the held-out result per city (Derry, London, Paris) with its row count. State Copenhagen (no valid 2025 hours) and Kilmainham (225
valid 2025 rows) in the row counts.]

[Insert: the same table without Cork UCC.]

## 8. Limitations and weaknesses

Each statement below is tied to evidence in `docs/M3.md` (section 9, error analysis); the causes are hypotheses, not findings.

- **Peaks.** [Insert: bias and MAE by concentration band from the error analysis of the selected model: whether high concentrations are
  systematically under-predicted and low ones over-predicted, and by how much.]
- **Where the model loses to persistence.** [Insert: the slices (stations, bands, hours, months) where skill is below zero, from
  `slices_where_model_loses_to_persistence.csv`.]
- **Isolated spikes.** [Insert: the worst 1% of errors, how they differ from other rows, and which stations are over-represented.]
- **Unseen places.** [Insert: the gap between seen and unseen stations (the 2022 fold) and between the development cities and the
  held-out cities; for the place ablation, which features helped or hurt and in which cities.]
- **Data limitations.** One sensor and one data source per station, so errors cannot be separated from instrument effects. The Irish
  data are provisional and unvalidated, start only in 2022 and cover two training years. The hour convention of the Irish files is
  inferred, and the one-hour alternative was not tested (`docs/M3.md`, section 10). Cork UCC has many negative readings (about 19% of
  its 2022–2023 targets) and an unexplained change in 2024 (its level rose by about 70% and the share of negative hours fell from about
  20% to 7.6%). Copenhagen has no valid 2025 hours and Kilmainham few. France is not in the training data, and Great Britain only
  through Belfast.
- **Temporal change.** Skill is lower on 2024 than on the training years the model was fitted on, and 2025 is [Insert: its result]. This
  is described as a performance change; whether it is drift is not established.
- **Design limits.** An 8-block (24-hour) history and no weather inputs; point forecasts only; the final model was not refitted on 2024.
- **Statistical limits.** Three seeds indicate variability and do not prove significance; the 1% adoption threshold is a heuristic.

## 9. Ethical considerations

- The data are public open data about air quality; no personal data are used.
- A forecast that is too low on a polluted day could mislead someone who relies on it; this is why the model is limited to
  course use, why persistence is always shown beside it, and why section 8 states where it fails.
- Skill differs between cities and countries (the evidence is limited by the small number of stations and the short Irish record), so
  results for one place should not be assumed for another.
- Compute used for the experiments: [Insert: the cloud and laptop time and approximate cost of training and benchmarking].

## 10. Caveats and recommendations

- Compare any use of the model with persistence on the same rows, and report the per-station errors with row counts.
- Re-evaluate before using the model for a different station, city, country, season length or sensor type.
- For higher concentrations, treat forecasts as lower bounds until the under-prediction in section 8 has been addressed. [Insert: the
  improvement the evidence supports, for example a loss closer to the metric, a longer history window, or weather inputs (future work).]
- Retraining on the 2026 data is planned for Milestone 4 and will be recorded as a new model version.

## 11. How to reproduce

[Insert: the exact commands, from a fresh checkout, to build the environment (or the container), fetch the processed data, and re-run the
training of this model from its configuration file; the model folder, MLflow run ID and expected development-year MAE; and the result of
the clean-run check.]

## 12. Version history

| Version | Date | Change | Dataset run ID | Development MAE |
|---|---|---|---|---|
| [Insert: v1] | [Insert] | [Insert: first selected model] | 20261019T213934Z | [Insert] |
