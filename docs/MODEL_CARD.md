# Model card: six-hour urban PM2.5 forecast

**Project:** Six-Hour Urban PM2.5 Forecasting Across Ireland and Europe
**Repository:** https://github.com/Dazzmondo/dh-PM2.5-forecast
**Author:** the student who made this Milestone 3 submission (identified by the submission itself)

**Scope of the numbers.** The numbers in sections 1, 4, 5 and 7 describe the tuned model (XGBoost, E7c configuration, absolute-error objective, `models/xgboost/v3`), which was frozen on 2026-11-15.

---

## 1. Model details

| Field | Content |
|---|---|
| Model | XGBoost regressor, absolute-error objective; PM2.5 history and time features only, no place inputs; folder `models/xgboost/v3/` (the folder number counts every saved XGBoost model; `v1` is the earlier squared-error model and `v2` the starting-values model) and the same path, `models/xgboost/v3/`, in the project's Google Cloud Storage bucket (uploaded on 2026-11-24: four files, 22,248,138 bytes, with its record in `manifests/model_versions.jsonl`, which names the dataset run, the raw-data version, the scaler hash, the MLflow run and the commit) |
| Type | Gradient-boosted decision trees (XGBoost 3.4.1, histogram tree method). Why: it handles non-linear structure and interactions in tabular inputs without scaling, was 9.3% better than ridge regression on the development year (MAE 3.147 against 3.471), and was 1.3% better and about three to four times cheaper to fit than the tuned GRU (114 s against 316–492 s for the 2024 fit on the same laptop) (sections 2, 3.2 and 5 of `docs/M3.md`) |
| Task | Regression: the mean PM2.5 concentration (µg/m³) of the 3-hour block that starts six hours after the current block, for one monitoring station |
| Forecast origin | A fixed 3-hour UTC block that has just finished; the forecast is for the block two blocks later (target = block t+2) |
| Output | One number per station and block (a point forecast; no uncertainty interval is produced) |
| Framework and versions | Python 3.14.2; xgboost 3.4.1; scikit-learn 1.9.1; torch 2.15.0 (GRU); pandas 3.0.6; numpy 2.5.3; pyarrow 25.0.1 (exact versions in `requirements.txt`) |
| Code | commit `0fbdf710f7daaeb063f9890afc1706e700b1819f` (the working tree had uncommitted changes); MLflow run `92f101e7bb67465cbfd00d16bfaa31be` (experiment `m3_candidates`, configuration `configs/xgboost_tuned_v1.yaml`); trained outside a container; container image `sha256:1d496aae27a9fd9ef3695ebcdc39114a4f7b3c42332c8752aa4ce77d3a595ed9` (the CPU image that re-trained it from a fresh clone, commit `8852fef`, on a Google Cloud VM) |
| Trained on | 2026-11-15, on a laptop (Intel i7-8565U, 4 physical cores / 8 threads, 15.8 GB RAM, no GPU); the fits of the three selection folds and of the 2024 model took 339 s in total (114 s for the 2024 fit); the model file is 22 MB; inference takes 22.9 ms per 1,000 rows |
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

Hyperparameters of the tuned model (from `models/xgboost/v3/config.yaml`): objective `reg:absoluteerror`, learning rate 0.0080,
maximum depth 6, minimum child weight 14, row subsample 0.794, column sample per tree 0.919, L2 penalty 0.013, L1 penalty 18.5,
gamma 0.40, histogram tree method, seed 1, target and inputs unscaled, early stopping on an inner 12-month window (patience 50, cap
4,000 rounds), which chose 3,682 rounds for the final fit on 2018–2023. They are the best configuration of one of three round-2
searches (E7c: TPE, 3 seeds × 60 trials, absolute-error objective, `docs/M3.md` section 3.6), the one with the lowest selection-fold
MAE (3.3393). The starting values (learning rate 0.1, 500 rounds, XGBoost defaults otherwise) scored 3.3554 on the same folds; the gain
is 0.48% on the selection folds and 0.2% on 2024 (3.1466 against 3.1542 for the starting values with early stopping), above the seed
noise (sd 0.001–0.002) and below the 1% adoption threshold, which applies to variants and not to the search. The F2023 fold stopped at
3,996 of 4,000 rounds, so the cap was nearly binding there.

## 5. Training and evaluation procedure

- **Selection folds.** Three expanding folds inside the training years (validation years 2021, 2022, 2023); a training row's target
  must end before the validation year starts. The decision metric is the MAE on seen-station rows (stations with earlier training rows),
  averaged with equal weight over the folds. A variant was adopted only if it improved that MAE by at least 1% in at least 2 of the 3
  folds and the 2024 MAE was not worse; otherwise the simpler option was kept.
- **Development year.** Fit on 2018–2023, score on 2024 (31,192 rows; all 12 stations seen).
- **Final evaluation, once.** The frozen model was scored once on the 2025 test split (stations seen in training) and on the held-out
  cities (never seen). No refit on 2024, so the model has not seen 2024.
- **Seeds and hardware.** Seed 1 for the tuned XGBoost model (it samples rows and columns, so the seed changes the trees; the seed alone moves the selection-fold MAE by about 0.001–0.002); seeds 0, 1 and 2 for the GRU comparison and the hyperparameter searches. Laptop (section 1); the benchmark, the profile and a reproduction run on a Google Cloud CPU VM (`n1-standard-8`: 4 physical cores, Xeon at 2.0 GHz) are in section 6.3 of `docs/M3.md`; the CPU-against-GPU benchmark, the GPU profile and two reproduction runs on a Google Cloud GPU VM (`n1-standard-8` with one NVIDIA T4) are in the same section.
- **Reproducibility.** Section 7 of `docs/M3.md`: pinned versions, container image, configuration file, dataset and run IDs, and the
  clean re-run result: selection-fold MAE 3.3409 and 2024 MAE 3.1452 from a fresh clone in a container on a VM, against the logged 3.3393 and 3.1466, inside the tolerance (section 7).

## 6. Metrics

MAE (primary, in µg/m³, because it is in the units of the target and not dominated by a few episodes) and RMSE (secondary, because it
penalises missed episodes more); the mean signed error (bias; negative means under-prediction); and the skill score
1 − MAE_model / MAE_persistence. All are reported overall and per station with the row count, with and without Cork UCC.

## 7. Performance

All numbers come from logged runs (`reports/m3/run_summary.csv`, `reports/final/`).

| Set | Rows | MAE | RMSE | Persistence MAE | Skill | Stations beating persistence |
|---|---|---|---|---|---|---|
| Selection folds (mean of 3, seen-station rows) | 14,825 / 15,260 / 31,377 | 3.339 | 6.070 | 4.131 | 0.192 | n/a |
| Development year 2024 | 31,192 | 3.147 | 6.263 | 3.869 | 0.187 | 12 / 12 |
| 2025 test split | 25,944 | 3.038 | 5.462 | 3.699 | 0.179 | 11 / 11 |
| Held-out cities (never seen) | 13,362 | 3.069 | 5.057 | 3.480 | 0.118 | 5 / 5 |

Development year, per station (`reports/m3/error_analysis/xgboost_v3_dev/by_station.csv`):

| Station | City | Rows | MAE | RMSE | Bias | Persistence MAE | Skill |
|---|---|---|---|---|---|---|---|
| DK0034A | Copenhagen | 2,457 | 2.948 | 4.474 | −0.484 | 3.255 | 0.094 |
| ES0118A | Madrid | 2,564 | 3.525 | 4.974 | −0.591 | 4.268 | 0.174 |
| ES1525A | Madrid | 2,685 | 3.040 | 4.602 | −0.299 | 3.492 | 0.130 |
| GB0567A | Belfast | 2,917 | 2.544 | 4.039 | −0.184 | 2.873 | 0.114 |
| GR0003A | Athens | 2,651 | 4.892 | 9.503 | −1.423 | 6.647 | 0.264 |
| GR0039A | Athens | 2,830 | 2.627 | 5.590 | −0.058 | 3.023 | 0.131 |
| IE0028A | Dublin | 2,158 | 3.091 | 5.727 | −0.615 | 4.023 | 0.232 |
| IE004BP | Cork | 2,652 | 2.692 | 4.514 | −0.100 | 3.393 | 0.207 |
| IE005AP | Dublin | 2,725 | 2.974 | 4.968 | −0.404 | 3.378 | 0.120 |
| IE005CP | Waterford | 2,686 | 3.196 | 7.524 | −0.694 | 3.977 | 0.196 |
| IE007CP | Limerick | 2,534 | 3.532 | 10.522 | −0.902 | 4.878 | 0.276 |
| IE009GY | Galway | 2,333 | 2.759 | 4.805 | −0.267 | 3.374 | 0.182 |

Without Cork UCC (28,540 rows): MAE 3.189, persistence 3.913, skill 0.185.
2025 test split, per station (`reports/m3/error_analysis/xgboost_v3_final/by_station.csv`; Copenhagen has no valid 2025 hours and so no rows, and Kilmainham, `IE005AP`, has 225):

| Station | City | Rows | MAE | RMSE | Bias | Persistence MAE | Skill |
|---|---|---|---|---|---|---|---|
| ES0118A | Madrid | 2,439 | 3.406 | 4.933 | −0.414 | 4.117 | 0.173 |
| ES1525A | Madrid | 2,289 | 3.044 | 4.846 | −0.159 | 3.436 | 0.114 |
| GB0567A | Belfast | 2,793 | 2.630 | 4.091 | −0.290 | 2.944 | 0.107 |
| GR0003A | Athens | 2,295 | 3.714 | 6.265 | −0.438 | 4.671 | 0.205 |
| GR0039A | Athens | 2,776 | 2.331 | 3.567 | 0.130 | 2.763 | 0.156 |
| IE0028A | Dublin | 2,446 | 3.289 | 5.949 | −0.752 | 4.150 | 0.208 |
| IE004BP | Cork | 2,599 | 3.076 | 6.325 | −0.389 | 3.892 | 0.210 |
| IE005AP | Dublin | 225 | 4.505 | 7.471 | −1.281 | 5.697 | 0.209 |
| IE005CP | Waterford | 2,574 | 3.738 | 7.614 | −1.007 | 4.634 | 0.193 |
| IE007CP | Limerick | 2,784 | 2.873 | 5.997 | −0.535 | 3.721 | 0.228 |
| IE009GY | Galway | 2,724 | 2.398 | 3.656 | −0.077 | 2.790 | 0.141 |

Without Cork UCC (23,345 test rows): MAE 3.034, persistence 3.677, skill 0.175. Held-out cities, per station (they contain no Cork station):

| Station | City | Rows | MAE | RMSE | Bias | Persistence MAE | Skill |
|---|---|---|---|---|---|---|---|
| FR04002 | Paris | 2,304 | 3.456 | 5.997 | −0.734 | 3.788 | 0.088 |
| FR04058 | Paris | 2,734 | 3.689 | 5.471 | −1.007 | 4.255 | 0.133 |
| GB0566A | London | 2,892 | 2.620 | 4.282 | −0.433 | 2.738 | 0.043 |
| GB0620A | London | 2,880 | 2.855 | 4.779 | −0.497 | 2.927 | 0.025 |
| GB1060A | Derry | 2,552 | 2.804 | 4.776 | −0.368 | 3.836 | 0.269 |

Per held-out city: Derry 2,552 rows, MAE 2.804, skill 0.269; London 5,772 rows, MAE 2.737, skill 0.034; Paris 5,038 rows, MAE 3.582, skill 0.114.

## 8. Limitations and weaknesses

Each statement below is tied to evidence in `docs/M3.md` (section 9, error analysis); the causes are hypotheses, not findings.

- **Peaks.** High concentrations are systematically under-predicted. For targets above 25 µg/m³ (3.6% of the development rows) the mean
  signed error is −15.8 µg/m³ and the skill against persistence is only 0.05; those rows give 18.8% of the absolute error. Tuning did not change this (−15.5 before tuning). Between 19.3
  and 25 the bias is −5.6; at or below 5 µg/m³ the model over-predicts by 1.9.
- **Where the model loses to persistence.** On the development year no station, band, hour, month or weekday with at least 100 rows
  loses to persistence for this model. The squared-error version of the model does lose in the lowest band (at or below 5 µg/m³: skill
  −0.04), which is one reason for the absolute-error objective; the cost is poorer peaks (above 25 µg/m³: MAE 16.2 against 14.9).
- **Isolated spikes.** The worst 1% of errors (312 rows, an absolute error above 19.6 µg/m³) are 91% under-predictions of surges that
  start from a low current value (mean target 48.3 against 9.1 for all rows); persistence is wrong by about as much in most of them, so
  no method that uses only the station's history catches them. Athens Aristotelous supplies 25% of them with 8.5% of the rows;
  Limerick and Waterford have an RMSE two to three times their MAE.
- **Season and time of day.** The error is largest in the heating season (November to March: 42% of the rows, 54% of the error) and for
  targets in the evening (18:00–23:59 local time: 25% of the rows, 33% of the error).
- **Unseen places.** In the place ablation (leave-one-city-out, 2023, ridge and XGBoost), latitude, longitude and country as inputs
  did not improve the pooled error (−0.6% to +1.9%, none adopted); they improved the five Irish cities a little (1.6–2.4% on average)
  and not the four cities that are alone in their country, and "all three" was much worse for ridge (Belfast +106%). The gap between
  the seen stations and the held-out cities on the same year is small in error (MAE 3.038 against 3.069) and larger in skill (0.179 against 0.118), because the held-out cities' persistence error is lower (3.480; London 2.832, skill 0.034).
- **Data limitations.** One sensor and one data source per station, so errors cannot be separated from instrument effects. The Irish
  data are provisional and unvalidated, start only in 2022 and cover two training years. The hour convention of the Irish files is
  inferred, and the one-hour alternative was not tested (`docs/M3.md`, section 10). Cork UCC has many negative readings (about 19% of
  its 2022–2023 targets) and an unexplained change in 2024 (its level rose by about 70% and the share of negative hours fell from about
  20% to 7.6%). Copenhagen has no valid 2025 hours and Kilmainham few. France is not in the training data, and Great Britain only
  through Belfast.
- **Temporal change.** Skill is lower on 2024 (0.185) than on the training years the model was fitted on (0.25–0.30, in-sample),
  mainly because persistence is a stronger baseline in 2024 (MAE 3.869, the lowest of any year); the model's MAE is similar (3.15 in 2024,
  2.88–3.22 in the training years). The 2025 test split gives an MAE of 3.038 and a skill of 0.179 (persistence 3.699, lower than in 2024), so the absolute error did not rise and the skill is close to 2024's. This is described as a performance change; whether it is
  drift is not established.
- **Design limits.** An 8-block (24-hour) history and no weather inputs; point forecasts only; the final model was not refitted on 2024.
- **Statistical limits.** Three seeds indicate variability and do not prove significance; the 1% adoption threshold is a heuristic.

## 9. Ethical considerations

- The data are public open data about air quality; no personal data are used.
- A forecast that is too low on a polluted day could mislead someone who relies on it; this is why the model is limited to
  course use, why persistence is always shown beside it, and why section 8 states where it fails.
- Skill differs between cities and countries (the evidence is limited by the small number of stations and the short Irish record), so
  results for one place should not be assumed for another.
- Compute used for the experiments: the experiments were run on a personal laptop, at no cloud cost, to protect the cloud credits, and the compute comparison also ran on two Google Cloud virtual machines; the CPU VM session on Google Cloud lasted 1 hour 14 minutes, about 0.60 at an assumed list price of 0.49 per hour (an estimate in US dollars, not the billed charge); the GPU VM session lasted about 1 hour 2 minutes, about 0.93 at an assumed 0.9 per hour (an estimate), after about four hours of refused requests before a T4 was available; training and prediction of this model need only a CPU (the GPU saved 15 seconds per fit and gave no gain in accuracy); [Insert: the billed charges of both sessions].

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
python src/train.py --config configs/xgboost_tuned_v1.yaml --save-model
```

Expected: selection-fold MAE 3.3393 and development-year MAE 3.1466 (persistence 3.8688, recomputed and checked by the script), saved as
the next `models/xgboost/v<N>/`. In the container: `docker build -t pm25-m3:cpu --build-arg GIT_COMMIT=$(git rev-parse HEAD) .`, then
`docker run --rm -v "$PWD/processed_data:/app/processed_data:ro" -v "$PWD/mlruns:/app/mlruns" -v "$PWD/reports:/app/reports" pm25-m3:cpu src/train.py --config configs/xgboost_tuned_v1.yaml`
(a CPU container reproduced the host metrics of the starting XGBoost configuration exactly. For the tuned configuration, which samples rows and columns, the CPU container on the Windows laptop (Linux under WSL2, 8 threads, the same XGBoost 3.4.1) gave selection-fold MAE 3.3409 and development-year MAE 3.1452 against 3.3393 and 3.1466 on the host: not identical, but within the seed-to-seed spread of 0.001–0.004 measured for such configurations, and the early-stopping rounds differ (2,540, 2,243, 4,000 and 3,996 against 2,241, 2,852, 3,996 and 3,682). A reproduction is therefore accepted when the selection-fold MAE is within 0.005 and the 2024 MAE within 0.003 of the host values).
Clean-run check: a fresh clone of the repository (commit `8852fef`), built into the CPU image on a Google Cloud VM and run with `configs/reproduce_xgboost_tuned_cpu.yaml`, gave selection-fold MAE 3.3409 and development-year MAE 3.1452. These are the same as the CPU container on the laptop (including the early-stopping rounds 2,540, 2,243, 4,000 and 3,996), and within the tolerance of 0.005 and 0.003 of the host values 3.3393 and 3.1466.

## 12. Version history

| Version | Date | Change | Dataset run ID | Development MAE |
|---|---|---|---|---|
| 1 (starting values) | 2026-11-14 | XGBoost, starting hyperparameters, absolute-error objective, no place inputs (`models/xgboost/v2`) | 20261019T213934Z | 3.152 |
| 2 (tuned, frozen 2026-11-15) | 2026-11-15 | The same model with the E7c configuration, early stopping on an inner window (`models/xgboost/v3`); frozen for the final evaluation | 20261019T213934Z | 3.147 |
