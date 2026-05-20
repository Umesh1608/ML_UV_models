# R1-8 Step 0 — Duplicate-Handling Analysis Summary

_Protocol: Greenman 2022 / Song 2025 (5.0 nm threshold)_

## Headline numbers

| stage | rows |
|---|---|
| raw CSV | 21,699 |
| after dropna (current training input) | 18,755 |
| after hypothetical keep-first-occurrence dedup | 18,456 |
| **after Greenman/Song protocol** | **18,415** |

**Loss vs current pipeline:** 340 rows (1.81%)  
**Loss vs first-occurrence dedup:** 41 rows (0.22%)

## Group statistics

- total (canon solute, canon solvent) groups: **18,456**
- singletons (n=1): 18,162
- duplicate groups (n≥2): 294
- duplicate groups averaged (spread ≤ 5 nm): 253 (covering 508 records)
- duplicate groups dropped (spread > 5 nm): 41 (covering 84 records)

## Spread percentiles (duplicate groups only)

| percentile | spread (nm) |
|---|---|
| p50 | 0.0 |
| p75 | 0.0 |
| p90 | 15.4 |
| p95 | 38.3 |
| p99 | 76.1 |
| max | 100.0 |
| mean | 5.3 |

## Wavelength-stratified breakdown

| wavelength_bin_nm | records_now | groups_total | groups_singleton | groups_duplicate | groups_averaged_le5nm | groups_dropped_gt5nm | records_in_avg_groups | records_in_drop_groups | records_after_protocol | pct_lost |
|---|---|---|---|---|---|---|---|---|---|---|
| <300 | 964 | 959 | 955 | 4 | 1 | 3 | 2 | 7 | 956 | 0.83 |
| 300-400 | 8807 | 8673 | 8540 | 133 | 103 | 30 | 206 | 61 | 8643 | 1.86 |
| 400-500 | 5097 | 4991 | 4887 | 104 | 97 | 7 | 196 | 14 | 4984 | 2.22 |
| 500-600 | 2314 | 2278 | 2242 | 36 | 35 | 1 | 70 | 2 | 2277 | 1.6 |
| >600 | 1572 | 1555 | 1538 | 17 | 17 | 0 | 34 | 0 | 1555 | 1.08 |

## Top-10 solvent breakdown

| solvent_canon | records | unique_solutes | duplicate_groups | dropped_gt5nm_groups | drop_rate_within_dup |
|---|---|---|---|---|---|
| ClCCl | 2854 | 2804 | 50 | 14 | 0.28 |
| CC#N | 1781 | 1742 | 39 | 6 | 0.154 |
| ClC(Cl)Cl | 1466 | 1432 | 34 | 1 | 0.029 |
| Cc1ccccc1 | 1449 | 1412 | 37 | 5 | 0.135 |
| C1CCOC1 | 1364 | 1342 | 22 | 0 | 0.0 |
| CO | 1322 | 1300 | 20 | 3 | 0.15 |
| CCO | 1034 | 1022 | 11 | 2 | 0.182 |
| CS(C)=O | 884 | 874 | 10 | 2 | 0.2 |
| C1CCCCC1 | 658 | 647 | 11 | 2 | 0.182 |
| CN(C)C=O | 622 | 617 | 5 | 0 | 0.0 |
