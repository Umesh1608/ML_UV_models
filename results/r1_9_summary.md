# R1-9 Step 0 — Chromophore-Solvent Leakage Audit

_v3 cleaned Joung+Beard primary dataset, 5-fold stratified CV_

## Headline numbers

- v3 unique solutes: **7846**, unique solvents: **726**, records: **18415**
- mean leakage rate (test records whose solute also appears in train *only* slice): **66.8%**
- mean leakage rate (test records whose solute also appears in train *or* val slice): **69.3%**

## Per-fold breakdown

| fold | n test | unique solutes in test | leaked vs train | leaked vs train+val |
|---|---|---|---|---|
| 0 | 3683 | 2609 | 2440 (66.2%) | 2534 (68.8%) |
| 1 | 3683 | 2564 | 2458 (66.7%) | 2554 (69.3%) |
| 2 | 3683 | 2567 | 2469 (67.0%) | 2546 (69.1%) |
| 3 | 3683 | 2547 | 2490 (67.6%) | 2575 (69.9%) |
| 4 | 3683 | 2539 | 2451 (66.5%) | 2545 (69.1%) |

## Per-model leakage effect (RMSE)

| model | leak rate | leaked RMSE | novel RMSE | ΔRMSE (novel − leaked) |
|---|---|---|---|---|
| bigru_tuned | 69.3% | 20.96 $\pm$ 1.45 | 57.16 $\pm$ 3.03 | +36.21 |
| chemberta | 69.3% | 45.13 $\pm$ 4.56 | 70.17 $\pm$ 3.68 | +25.04 |
| chemprop | 69.3% | 19.65 $\pm$ 4.10 | 51.90 $\pm$ 4.73 | +32.26 |
| rf_tuned | 66.8% | 16.36 $\pm$ 0.88 | 49.53 $\pm$ 2.90 | +33.16 |
| xgboost | 66.8% | 21.33 $\pm$ 0.29 | 50.52 $\pm$ 2.79 | +29.19 |

## Decision-rule bucketing (per-model)

Per user's spec: ΔRMSE = (novel − leaked); large → ≥ 2 nm; bounded → 0.5–2 nm; small → < 0.5 nm.

| model | ΔRMSE | bucket |
|---|---|---|
| bigru_tuned | +36.21 | **LARGE** |
| chemberta | +25.04 | **LARGE** |
| chemprop | +32.26 | **LARGE** |
| rf_tuned | +33.16 | **LARGE** |
| xgboost | +29.19 | **LARGE** |