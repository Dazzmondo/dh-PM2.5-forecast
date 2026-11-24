# Hyperparameter history: Milestone 3

**Project:** Six-Hour Urban PM2.5 Forecasting Across Ireland and Europe
**Repository:** https://github.com/Dazzmondo/dh-PM2.5-forecast

This document records every hyperparameter value, range and change used in Milestone 3, with the justification and the evidence for each.
The values and ranges were developed in consultation with an AI assistant. The results are in `docs/M3.md` (sections 3 and 5); the
advantages and drawbacks of the final values are discussed in section 3.8 of that report.

## 1. Terms

| Term | Meaning | Examples in this project |
|---|---|---|
| **Hyperparameter** | A setting fixed before training, by a person or by a search that a person designs. It is not learned by the fitting step | XGBoost learning rate, depth, minimum child weight, subsample, colsample, lambda, alpha, gamma and the cap on boosting rounds; Ridge alpha; GRU hidden size, layers, dropout, learning rate and batch size; the early-stopping patience and inner-window length |
| **Parameter** | A value learned from the training data by the fitting step | XGBoost: the split features, thresholds and leaf values of every tree. Ridge: the coefficients and the intercept. GRU: all weights and biases. Preprocessing: the mean and standard deviation of each scaler |
| **Chosen by the training procedure** | A hyperparameter in kind whose value is fixed by a rule inside training | The number of boosting rounds picked by early stopping (3,682 in the final fit); the epoch at which the GRU stops |
| **Design choice** | A decision about the data, inputs or evaluation rather than the model's settings | The 8-block window, lags 1–3, target form, scaling choice, cyclical time features, the 1% adoption rule, the splits |

The tools call every logged setting a "param" (MLflow's `params`, the `params:` key in `configs/`, Optuna's "parameter importance"): there
it means hyperparameter. No learned parameter is ever set by hand.

## 2. XGBoost

| Step | Hyperparameter | From | To | Justification | Evidence |
|---|---|---|---|---|---|
| Starting values (`docs/M3.md` 3.1) | learning rate; rounds | library default 0.3 and 100 | 0.1 and 500 | 0.1 is the usual gentler-than-default shrinkage; 500 rounds is a deliberately generous budget, so that early stopping can find the number the data supports | A reference point, not derived from the data |
| | depth, minimum child weight, subsample, colsample, lambda, alpha, gamma | | 6, 1, 1.0, 1.0, 1, 0, 0 | The library defaults. Depth 6 allows interactions between the current block, its lags, the time of day and the season without an unrestricted tree | As above |
| E6a early stopping | rounds | 500, fixed | chosen per fit by early stopping | Removes the number of rounds as a hand-set value | Best rounds 144, 164, 410, 270; better than 500 fixed rounds by 2.2%, 1.9%, 1.4% (selection folds) and 0.9% (2024), so adopted under the 1% rule |
| | patience; inner window | | 50 rounds; the last 12 months of each fold's training rows | The inner window means the stopping decision never sees the validation fold; 50 is a conventional patience | Not tested separately |
| E6b objective | objective | squared error | absolute error | The metric is MAE, and absolute error fits the conditional median, which is what MAE rewards | Better by 4.4%, 3.6%, 5.0% (folds) and 4.3% (2024) |
| | pseudo-Huber: starting score, slope | library defaults (0.5, 1) | 10, 5 | The default starting score did not converge (MAE 451: its steps are bounded and the start was far from the data); 10 is close to the mean target (9.07) and 5 is of the order of the typical absolute error | Pseudo-Huber was 3.1% and 3.5% better than squared error but not better than absolute error |
| E7 and E7b: search ranges | learning rate | | 0.01 to 0.3, log | A wide range on both sides of the starting value, so the search can move either way; log scale because the effect of a rate, a penalty or a shrinkage is multiplicative | See the results below |
| | max depth | | 3 to 10 | As above | |
| | min child weight | | 1 to 20 | As above | |
| | subsample, colsample | | 0.5 to 1.0 each | As above; below 0.5 too little data per tree | |
| | lambda, alpha | | 0.001 to 10 each, log | As above | |
| | gamma | | 0 to 5 | As above | |
| | rounds cap | | 2,000 | A budget limit; early stopping picks the actual number | Two studies reached it (2,000 and 1,980), which is why E7c raises it |
| | search design | | Random search and TPE, seeds 0–2, 50 trials each (300 trials per experiment); E7 squared error, E7b absolute error | Equal budgets, so that the extra cost of Bayesian search can be judged; E7b repeats E7 once the absolute-error objective had won E6b | E7 gained 0.45% and E7b 0.29–0.36% on the selection folds; random and TPE differ by 0.001 (E7) and 0.002 (E7b) |
| E7c: narrowed ranges | learning rate | 0.01 to 0.3 | 0.005 to 0.05, log | E7 and E7b winners lay at 0.012–0.042; the lower end is extended because hypothesis T1 predicts slower learning with more rounds | |
| | max depth | 3 to 10 | 5 to 9 | Winners were 6 to 9, mostly 7 | |
| | min child weight | 1 to 20 | 5 to 30 | Winners were 1 to 19; hypothesis T3 predicts a larger value helps | |
| | subsample | 0.5 to 1.0 | 0.5 to 0.9 | Winners were 0.64 to 0.92 | |
| | colsample | 0.5 to 1.0 | 0.6 to 1.0 | Winners were 0.62 to 0.97 | |
| | lambda | 0.001 to 10 | 0.001 to 10 | Unchanged: the winners showed no pattern | |
| | alpha | 0.001 to 10 | 0.1 to 20 | Winners were 0.01 to 9.5; hypothesis T4 predicts a larger L1 helps, so the upper end is raised | |
| | gamma | 0 to 5 | 0 to 3 | Winners were 0.33 to 4.1 | |
| | rounds cap | 2,000 | 4,000 | Two E7b studies reached the old cap | The final fold stopped at 3,996 of 4,000 rounds |
| | search design | | TPE only, seeds 0–2, 60 trials each (180 trials) | The sampler comparison was already made in E7 and E7b | |
| E7c outcome | all eight | starting values | learning rate 0.00799, depth 6, minimum child weight 14, subsample 0.794, colsample 0.919, lambda 0.0129, alpha 18.5, gamma 0.400 | The study with the lowest selection-fold objective (3.3393); 2024 was not used to choose | Three studies agree on a low learning rate, row subsampling and a larger minimum child weight, and disagree on lambda, alpha and gamma, with objectives within 0.003 |
| Refit | seed | 0 | 1 | The seed of the selected study, so its scores are reproduced | 3.3393 and 3.1466 reproduced exactly |

Configurations scored by the searches: 300 (E7) + 300 (E7b) + 180 (E7c) = 780, each setting all eight hyperparameters together; no
hyperparameter was varied alone. The noise-floor check (15 evaluations of three configurations with seeds 0–4) introduced no new values.

## 3. Ridge

| Step | Hyperparameter | From | To | Justification | Evidence |
|---|---|---|---|---|---|
| Starting | alpha | | 1.0, on standardised inputs | The library default; with 17 inputs and 123,364 rows the penalty is small, giving a clean linear reference | |
| E7d sweep | alpha | | 13 values from 0.001 to 10,000 | A range wide enough on both sides to show where the penalty starts to matter; a tie with the starting value keeps the starting value | Flat up to about 100; alpha 1 kept |

## 4. GRU

| Step | Hyperparameter | From | To | Justification | Evidence |
|---|---|---|---|---|---|
| Starting | hidden size, layers, dropout, learning rate, batch size | | 32, 1, 0, 0.003, 256 | A small network for 123,364 rows of 8 steps, so that capacity is not obviously larger than the data supports; 0.003 is in the usual range for Adam | |
| | optimiser, schedule, stopping | | Adam; learning rate halved after 2 epochs without improvement; early stopping with patience 7; at most 50 epochs; weight decay 0 | Conventional settings | |
| E9a grid | hidden size; learning rate; loss | | 32 or 64; 0.001, 0.003 or 0.01; squared error or L1; dropout fixed at 0; one fold, one seed | A small grid inside a 12-configuration cap; L1 is included because absolute error won for XGBoost; dropout 0 because a one-layer network of this size has little to regularise | Best: 32 units, 0.001, L1 (L1 beat squared error in every pair) |
| E9b search | hidden size | | 16, 32, 64, 128 | Wider than E9a to test whether capacity is the limit (T9) | |
| | layers | | 1 or 2 | As above | |
| | dropout | | 0 to 0.3 | To test regularisation | |
| | learning rate | | 0.0003 to 0.01, log | Around and below the starting value | The best trials sit at the lower end |
| | batch size | | 128, 256, 512, 1,024 | To test the effect on error and training time | |
| | search design | | TPE, 40 trials (the first 10 random), one sampler seed, one model seed; loss fixed at L1; weight decay not searched | A budget that fits a laptop; L1 won every pair in E9a; the learning curves show no over-fitting | Search took 9.7 hours |
| E9b outcome | all five | 32, 1, 0, 0.003, 256 | 32 units, 2 layers, dropout 0.0841, learning rate 0.000787, batch 256 | The trial with the lowest selection-fold objective (3.3650; three-seed mean 3.371) | The six best trials are within 0.003 and cannot be separated |

At the freeze (2026-11-15) the GRU batch size was kept at 256. A larger batch (512 or 1,024) would cut the time per epoch (on the laptop CPU, from 6.0 s at 256 to 3.7 s at 1,024 in the profile of the tuned GRU), but XGBoost is the stronger and cheaper model, so a small saving on the GRU does not change the project's conclusion; and the saving is not clear-cut (fewer updates per epoch can mean more epochs, batch 1,024 was about 0.5% worse in the search, and the tuned learning rate belongs to batch 256).

## 5. What is learned, not set (parameters)

XGBoost: the trees (3,682 in the final fit). Ridge: 17 coefficients and an intercept for the standardised inputs. GRU: the weights of a
32-unit, 2-layer network and its output head. Scalers: a mean and a standard deviation per input, fitted on the training rows of each fold.

## 6. Design choices that interact with hyperparameters

The 8-block window and lags 1–3 (checked against 16 and 24 blocks in E4d: no gain), feature set A, the raw target, no standardisation
for trees, cyclical time features, no seasonal feature, no place inputs, the 2-of-3-hours block rule and the 1% adoption rule.
