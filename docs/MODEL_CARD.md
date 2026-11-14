# Model card: six-hour urban PM2.5 forecast

**Project:** Six-Hour Urban PM2.5 Forecasting Across Ireland and Europe
**Repository:** https://github.com/Dazzmondo/dh-PM2.5-forecast
**Author and contact:** [Insert: name]

> Drafting conventions (delete this block before the file is final): every `[Insert: ...]` is a result or a fact that must come from a
> logged MLflow run, a model folder or a manifest (name the source). Everything else states only facts already verified in the
> repository or in the Milestone 2 evidence. No slot may be left in the final file, and no number may be typed in that is not in the
> exported run tables (`reports/m3/run_summary.csv`) or the final evaluation table (`reports/final/`).
> The numbers in sections 1, 4, 5 and 7 describe the current candidate (XGBoost, starting hyperparameters, absolute-error objective,
> `models/xgboost/v2`); at the freeze they are replaced by those of the tuned, frozen model.

---

## 1. Model details

| Field | Content |
|---|---|
| Model | XGBoost regressor, absolute-error objective; version 1 of the project (the primary version: PM2.5 history and time features only, no place inputs); folder `models/xgboost/v2/` (the folder number counts every saved XGBoost model; `v1` is the earlier squared-error model) and [Insert: the same path in the bucket, `models/xgboost/v2/`, once uploaded] |
| Type | Gradient-boosted decision trees (XGBoost 3.4.1, histogram tree method). Why: it handles non-linear structure and interactions in tabular inputs without scaling, was 9.2% better than ridge regression on the development year (MAE 3.152 against 3.471), and was 1.0% better and roughly 30 times cheaper to fit than the GRU (about 5 s against 150–200 s per fit on the same laptop) (sections 2, 3.2 and 5 of `docs/M3.md`) |
| Task | Regression: the mean PM2.5 concentration (µg/m³) of the 3-hour block that starts six hours after the current block, for one monitoring station |
| Forecast origin | A fixed 3-hour UTC block that has just finished; the forecast is for the block two blocks later (target = block t+2) |
| Output | One number per station and block (a point forecast; no uncertainty interval is produced) |
| Framework and versions | Python 3.14.2; xgboost 3.4.1; scikit-learn 1.9.1; torch 2.15.0 (GRU); pandas 3.0.6; numpy 2.5.3; pyarrow 25.0.1 (exact versions in `requirements.txt`) |
| Code | commit `8a09bc7881610d8ad0f4bb23c23581e462b9be61`; MLflow run `70f05e2e99344d14ba5e0593010ff8c4` (experiment `m3_candidates`, configuration `configs/xgboost_final_candidate.yaml`); trained outside a container; container image [Insert: image ID or registry digest of the run that re-trained it] |
| Trained on | 2026-11-14, on a laptop (Intel i7-8565U, 4 physical cores / 8 threads, 15.8 GB RAM, no GPU); the fits of the three selection folds and of the 2024 model took 20.6 s in total; the model file is 3.5 MB; inference takes 3.1 ms per 1,000 rows |
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
| Inputs | Feature set A, ten columns: the current block's PM2.5 (`pm25_current`), the three earlier blocks (`pm25_lag_1`, `pm25_lag_2`, `pm25_lag_3`), the mean and standard deviation of the last 8 blocks (`pm25_rolling_mean`, `pm25_rolling_std`), and the time features `hour_sin`, `hour_cos` (local hour at the midpoint of the current block) and `month_sin`, `month_cos` (local month). No seasonal-lag or raw-window variant (neither passed the adoption rule) |
| Not used as inputs | Station identity, city, country, latitude and longitude (tested only as a named ablation); weather and other pollutants (outside the project's scope) |
| Scaling | Fitted inside the model on the training rows; for the Ridge model verified equal to the stored `scalers.json` (section 7 of `docs/M3.md`) |

## 4. Architecture and configuration

Hyperparameters of the current candidate (from `models/xgboost/v2/config.yaml`): objective `reg:absoluteerror`, 500 boosting rounds,
learning rate 0.1, maximum depth 6, and XGBoost's defaults for the rest (minimum child weight 1, row and column sampling 1.0, L2 penalty 1,
L1 penalty 0, gamma 0), histogram tree method, seed 0, target and inputs unscaled. These are the **starting values** of section 3.1 of
`docs/M3.md` with the absolute-error objective chosen in E6b; they were not tuned.
[Insert: the tuned configuration of the frozen model: the hyperparameters, the number of rounds from early stopping, and the search that
chose it (round 2, TPE, seeds, number of trials, the MLflow run), with its gain over the starting values.]

## 5. Training and evaluation procedure

- **Selection folds.** Three expanding folds inside the training years (validation years 2021, 2022, 2023); a training row's target
  must end before the validation year starts. The decision metric is the MAE on seen-station rows (stations with earlier training rows),
  averaged with equal weight over the folds. A variant was adopted only if it improved that MAE by at least 1% in at least 2 of the 3
  folds and the 2024 MAE was not worse; otherwise the simpler option was kept.
- **Development year.** Fit on 2018–2023, score on 2024 (31,192 rows; all 12 stations seen).
- **Final evaluation, once.** The frozen model was scored once on the 2025 test split (stations seen in training) and on the held-out
  cities (never seen). No refit on 2024, so the model has not seen 2024.
- **Seeds and hardware.** Seed 0 for the XGBoost candidate (with row and column sampling off the model does not depend on the seed); seeds 0, 1 and 2 for the GRU comparison and the hyperparameter searches. Laptop (section 1); the benchmark and profile on a cloud VM are [Insert].
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
| Selection folds (mean of 3, seen-station rows) | 14,825 / 15,260 / 31,377 | 3.361 | 6.054 | 4.131 | 0.186 | n/a |
| Development year 2024 | 31,192 | 3.152 | 6.263 | 3.869 | 0.185 | 12 / 12 |
| 2025 test split | [Insert] | [Insert] | [Insert] | [Insert] | [Insert] | [Insert] |
| Held-out cities (never seen) | [Insert] | [Insert] | [Insert] | [Insert] | [Insert] | [Insert] |

Development year, per station (`reports/m3/error_analysis/xgboost_v2_dev/by_station.csv`):

| Station | City | Rows | MAE | RMSE | Bias | Persistence MAE | Skill |
|---|---|---|---|---|---|---|---|
| DK0034A | Copenhagen | 2,457 | 2.958 | 4.481 | −0.492 | 3.255 | 0.091 |
| ES0118A | Madrid | 2,564 | 3.513 | 4.953 | −0.551 | 4.268 | 0.177 |
| ES1525A | Madrid | 2,685 | 3.036 | 4.596 | −0.286 | 3.492 | 0.131 |
| GB0567A | Belfast | 2,917 | 2.542 | 4.058 | −0.204 | 2.873 | 0.115 |
| GR0003A | Athens | 2,651 | 4.896 | 9.433 | −1.332 | 6.647 | 0.263 |
| GR0039A | Athens | 2,830 | 2.639 | 5.607 | +0.022 | 3.023 | 0.127 |
| IE0028A | Dublin | 2,158 | 3.095 | 5.733 | −0.611 | 4.023 | 0.231 |
| IE004BP | Cork | 2,652 | 2.707 | 4.498 | −0.074 | 3.393 | 0.202 |
| IE005AP | Dublin | 2,725 | 2.988 | 5.001 | −0.422 | 3.378 | 0.116 |
| IE005CP | Waterford | 2,686 | 3.207 | 7.522 | −0.688 | 3.977 | 0.194 |
| IE007CP | Limerick | 2,534 | 3.524 | 10.563 | −0.896 | 4.878 | 0.278 |
| IE009GY | Galway | 2,333 | 2.775 | 4.823 | −0.248 | 3.374 | 0.178 |

Without Cork UCC (28,540 rows): MAE 3.193, persistence 3.913, skill 0.184.
[Insert: the same two tables for the 2025 test split and the held-out result per city (Derry, London, Paris) with its row count, with and without Cork UCC. State Copenhagen (no valid 2025 hours) and Kilmainham (225 valid 2025 rows) in the row counts.]

## 8. Limitations and weaknesses

Each statement below is tied to evidence in `docs/M3.md` (section 9, error analysis); the causes are hypotheses, not findings.

- **Peaks.** High concentrations are systematically under-predicted. For targets above 25 µg/m³ (3.6% of the development rows) the mean
  signed error is −15.5 µg/m³ and the skill against persistence is only 0.06; those rows give 18.6% of the absolute error. Between 19.3
  and 25 the bias is −5.6; at or below 5 µg/m³ the model over-predicts by 1.9.
- **Where the model loses to persistence.** On the development year no station, band, hour, month or weekday with at least 100 rows
  loses to persistence for this model. The squared-error version of the model does lose in the lowest band (at or below 5 µg/m³: skill
  −0.04), which is one reason for the absolute-error objective; the cost is poorer peaks (above 25 µg/m³: MAE 16.2 against 14.9).
- **Isolated spikes.** The worst 1% of errors (312 rows, an absolute error above 19.6 µg/m³) are 89% under-predictions of surges that
  start from a low current value (mean target 48.3 against 9.1 for all rows); persistence is wrong by about as much in most of them, so
  no method that uses only the station's history catches them. Athens Aristotelous supplies 25% of them with 8.5% of the rows;
  Limerick and Waterford have an RMSE two to three times their MAE.
- **Season and time of day.** The error is largest in the heating season (November to March: 42% of the rows, 54% of the error) and for
  targets in the evening (18:00–23:59 local time: 25% of the rows, 33% of the error).
- **Unseen places.** In the place ablation (leave-one-city-out, 2023, ridge and XGBoost), latitude, longitude and country as inputs
  did not improve the pooled error (−0.6% to +1.9%, none adopted); they improved the five Irish cities a little (1.6–2.4% on average)
  and not the four cities that are alone in their country, and "all three" was much worse for ridge (Belfast +106%). The gap between
  the development cities and the held-out cities is [Insert: from the final evaluation].
- **Data limitations.** One sensor and one data source per station, so errors cannot be separated from instrument effects. The Irish
  data are provisional and unvalidated, start only in 2022 and cover two training years. The hour convention of the Irish files is
  inferred, and the one-hour alternative was not tested (`docs/M3.md`, section 10). Cork UCC has many negative readings (about 19% of
  its 2022–2023 targets) and an unexplained change in 2024 (its level rose by about 70% and the share of negative hours fell from about
  20% to 7.6%). Copenhagen has no valid 2025 hours and Kilmainham few. France is not in the training data, and Great Britain only
  through Belfast.
- **Temporal change.** Skill is lower on 2024 (0.185) than on the training years the model was fitted on (0.25–0.30, in-sample),
  mainly because persistence is a stronger baseline in 2024 (MAE 3.869, the lowest of any year); the model's MAE is similar (3.15 in 2024,
  2.88–3.22 in the training years). The result for 2025 is [Insert: its result]. This is described as a performance change; whether it is
  drift is not established.
- **Design limits.** An 8-block (24-hour) history and no weather inputs; point forecasts only; the final model was not refitted on 2024.
- **Statistical limits.** Three seeds indicate variability and do not prove significance; the 1% adoption threshold is a heuristic.

## 9. Ethical considerations

- The data are public open data about air quality; no personal data are used.
- A forecast that is too low on a polluted day could mislead someone who relies on it; this is why the model is limited to
  course use, why persistence is always shown beside it, and why section 8 states where it fails.
- Skill differs between cities and countries (the evidence is limited by the small number of stations and the short Irish record), so
  results for one place should not be assumed for another.
- Compute used for the experiments: the experiments so far were run on a personal laptop, at no cloud cost, to protect the cloud credits; the benchmark and profile on GCP VMs are [Insert: the VM time and approximate cost of training and benchmarking].

## 10. Caveats and recommendations

- Compare any use of the model with persistence on the same rows, and report the per-station errors with row counts.
- Re-evaluate before using the model for a different station, city, country, season length or sensor type.
- For higher concentrations, treat forecasts as lower bounds until the under-prediction in section 8 has been addressed. The evidence
  points to inputs beyond the station's own PM2.5 history (weather, other pollutants, nearby stations; future work) as the main
  improvement, because the largest failures are surges that nothing in the history foreshadows; a loss or output aimed at the upper tail
  and a blend of the XGBoost and GRU models are further hypotheses, not tested (`docs/M3.md`, section 9.5).
- Retraining on the 2026 data is planned for Milestone 4 and will be recorded as a new model version.

## 11. How to reproduce

From a fresh checkout of the repository (Python 3.14.2):

```bash
python -m venv .venv && source .venv/Scripts/activate      # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                            # exact pinned versions
```

The processed data (dataset run ID 20261019T213934Z) are produced by the pipeline of Milestones 1 and 2 (`docs/M1.md`, `README.md`) or
fetched from the bucket's `processed/` folder; the features file must have SHA-256 `b2cd1f6ecfdb5d313688538dc415c3768aa987e0002f2b559b4d05895695ffc4`
and the block table `f3d9209ded925b93c10c60ae9ea740cd031e3ea56c868abd465e607d8e9d45a3` (the run records both). Then re-train and score:

```bash
export DATASET_RUN_ID=20261019T213934Z
python src/train.py --config configs/xgboost_final_candidate.yaml --save-model
```

Expected: selection-fold MAE 3.3612 and development-year MAE 3.1516 (persistence 3.8688, recomputed and checked by the script), saved as
the next `models/xgboost/v<N>/`. In the container: `docker build -t pm25-m3:cpu --build-arg GIT_COMMIT=$(git rev-parse HEAD) .`, then
`docker run --rm -v "$PWD/processed_data:/app/processed_data:ro" -v "$PWD/mlruns:/app/mlruns" -v "$PWD/reports:/app/reports" pm25-m3:cpu src/train.py --config configs/xgboost_final_candidate.yaml`
(a CPU container reproduced the host metrics of the starting XGBoost configuration exactly).
[Insert: the same commands for the tuned frozen model (its configuration file and expected metric), and the result of the clean-run check.]

## 12. Version history

| Version | Date | Change | Dataset run ID | Development MAE |
|---|---|---|---|---|
| 1 (candidate) | 2026-11-14 | XGBoost, starting hyperparameters, absolute-error objective, no place inputs (`models/xgboost/v2`) | 20261019T213934Z | 3.152 |
| [Insert: 1 (frozen)] | [Insert] | [Insert: the tuned model chosen at the freeze] | 20261019T213934Z | [Insert] |
