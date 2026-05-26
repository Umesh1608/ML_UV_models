# R2-5 Step 0: Polarity-Grouped Prediction Variability

## Tier coverage (v3 records)
- nonpolar: 3,010 records (16.3%)
- moderate: 6,619 records (35.9%)
- polar_aprotic: 3,625 records (19.7%)
- polar_protic: 3,412 records (18.5%)
- other: 1,749 records (9.5%)
- TOTAL: 18,415 records

## Test-fold RMSE by (model, tier)

| Tier | RF (tuned) | Chemprop | BiGRU (tuned) | ChemBERTa |
|---|---|---|---|---|
| nonpolar | 27.7 (n=3010) | 28.4 (n=3010) | 30.5 (n=3010) | 50.5 (n=3010) |
| moderate | 34.5 (n=6619) | 37.9 (n=6619) | 39.6 (n=6619) | 53.6 (n=6619) |
| polar aprotic | 29.7 (n=3625) | 32.0 (n=3625) | 32.6 (n=3625) | 42.5 (n=3625) |
| polar protic | 30.2 (n=3412) | 29.7 (n=3412) | 32.7 (n=3412) | 49.7 (n=3412) |
| other | 32.4 (n=1749) | 31.8 (n=1749) | 44.7 (n=1749) | 84.4 (n=1749) |
| ALL | 31.5 (n=18415) | 33.3 (n=18415) | 36.2 (n=18415) | 54.2 (n=18415) |