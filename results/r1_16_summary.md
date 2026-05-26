# R1-16 Step 0: ChemBERTa 600 nm Prediction-Ceiling Diagnostic

## (H1) Training-distribution imbalance (fold 0)
- Train records: 11,785
- Train records with lambda_max > 600 nm: 980 (8.32%)
- Train records with lambda_max > 700 nm: 272 (2.31%)
- Train records with lambda_max > 800 nm: 47
- Train median: 396.0 nm; p95: 647.0 nm; p99: 755.0 nm

## (H2) ChemBERTa prediction distribution
- Pred min/median/max: 185.0 / 396.4 / 556.4 nm
- Frac pred > 600 nm: 0.00%
- Frac pred > 700 nm: 0.00%
- Test records with y_true > 600 nm: **1555**
- Median ChemBERTa prediction when y > 600: **549.1 nm**
- Frac of those predictions in [550, 650]: **36.9%**
- Frac underprediction (pred < y - 50): **98.5%**
- Median underprediction (y - pred): **125.7 nm**

## (H3) SMILES length vs lambda_max bin
- Test set median combined-SMILES length: 61.0 chars
- p95: 135.0 chars; max: 645 chars
- Frac of test records truncated at MAX_LEN=256: **0.23%**

| y_true bin (nm) | n | mean SMILES len | frac truncated |
|---|---:|---:|---:|
| 150-400 | 9490 | 60.8 | 0.2% |
| 400-500 | 5037 | 70.4 | 0.2% |
| 500-600 | 2318 | 84.8 | 0.5% |
| 600-700 | 1139 | 90.1 | 0.3% |
| 700-1100 | 430 | 101.6 | 0.2% |
