# R1-14 Step 0: Solvent Ablation on Multi-Solvent Subset Only

- Dataset: v2 (`UV_canonical_full_dataset.csv`), 18755 records after dropna
- Folds aggregated: 5 / 5
- Fraction of test records whose chromophore is multi-solvent in train: mean = 60.3%

## Aggregate Results (5-fold mean ± std)

| Model | Solvent | Full RMSE | Multi-solv subset RMSE | Single-solv/novel subset RMSE | Δ Full | Δ Multi | Δ Single |
|---|---|---|---|---|---|---|---|
| RF (tuned) | +Solv | 31.336 ± 1.821 | 15.494 ± 0.795 | 45.853 ± 2.995 | 2.791 (8.18%) | 7.224 (31.8%) | 0.416 (0.9%) |
| RF (tuned) | Solute-only | 34.127 ± 1.442 | 22.718 ± 0.684 | 46.269 ± 2.925 |  |  |  |
| Chemprop | +Solv | 31.692 ± 3.181 | 17.547 ± 2.288 | 45.277 ± 4.951 | 2.09 (6.19%) | 5.913 (25.2%) | -0.275 (-0.61%) |
| Chemprop | Solute-only | 33.782 ± 2.732 | 23.46 ± 0.654 | 45.002 ± 5.187 |  |  |  |
| BiGRU | +Solv | 36.453 ± 1.124 | 20.97 ± 1.382 | 51.701 ± 1.799 | 2.577 (6.6%) | 5.65 (21.22%) | 0.769 (1.47%) |
| BiGRU | Solute-only | 39.03 ± 1.374 | 26.62 ± 1.052 | 52.47 ± 2.427 |  |  |  |

Δ = (Solute-only RMSE) − (+Solvent RMSE) on that subset; positive Δ means solvent encoding helps.