# R2-1 Step 0: Experimental lambda_max Uncertainty

## (A) Inter-Source (Joung vs Beard) Disagreement
- Paired (canon, solvent) observations in both sources: **21**
- Median |Joung - Beard|: **49.00 nm**
- Mean   |Joung - Beard|: 35.81 nm
- 75th pct: 65.00 nm
- 90th pct: 73.00 nm
- 95th pct: **76.00 nm**
- Maximum:  77.00 nm
- Fraction within  5 nm: 38.1%
- Fraction within 10 nm: 38.1%
- Fraction within 20 nm: 42.9%
- Inter-source RMSD: **47.52 nm**

## (B) Within-Source Duplicate Spread (max - min)
### Joung
- Total records: 1,479
- Unique (canon, solvent) pairs: 1,479
- Pairs with multiplicity >1: 0 (0 records)

### Beard
- Total records: 17,276
- Unique (canon, solvent) pairs: 16,999
- Pairs with multiplicity >1: 273 (550 records)
- Median spread: **0.00 nm**
- 75th pct: 0.00 nm
- 95th pct: **21.00 nm**
- Fraction spread <=  5 nm: 89.7%
- Fraction spread <= 10 nm: 92.3%
- Fraction spread <= 20 nm: 94.1%
- Fraction spread >  50 nm: 1.1%

### v2_combined
- Total records: 18,755
- Unique (canon, solvent) pairs: 18,457
- Pairs with multiplicity >1: 294 (592 records)
- Median spread: **0.00 nm**
- 75th pct: 0.01 nm
- 95th pct: **38.35 nm**
- Fraction spread <=  5 nm: 86.1%
- Fraction spread <= 10 nm: 88.4%
- Fraction spread <= 20 nm: 90.5%
- Fraction spread >  50 nm: 4.1%

## (C) Comparison to Model Errors (v3 cleaned dataset, Table~1)
- RF_RMSE_nm: 31.5 nm
- Chemprop_default_RMSE_nm: 33.15 nm
- Chemprop_tuned_RMSE_nm: 28.83 nm
- BiGRU_tuned_RMSE_nm: 36.2 nm
- ChemBERTa_RMSE_nm: 54.09 nm

**Headroom analysis** (if inter-source RMSD is interpretable):
- RF: total RMSE = 31.5 nm, inter-source noise floor = 47.5 nm, residual (model-attributable) = 0.0 nm (0% of total)
- Chemprop: total RMSE = 33.1 nm, inter-source noise floor = 47.5 nm, residual (model-attributable) = 0.0 nm (0% of total)
- ChemBERTa: total RMSE = 54.1 nm, inter-source noise floor = 47.5 nm, residual (model-attributable) = 25.8 nm (48% of total)