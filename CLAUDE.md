# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Reframed as benchmark paper** (March 2026): Systematic benchmark of ML approaches (RF, XGBoost, BiGRU) for UV λ_max prediction from 1D molecular representations. Key findings: (1) RF+Morgan FP beats BiGRU on every benchmark, (2) solvent concatenation helps both model families, (3) 1D approaches have fundamental ceiling vs 3D GNNs. Secondary contribution: solvent concatenation strategy.

**New LaTeX file**: `benchmark_paper.tex` (benchmark framing). Original `ml_chemistry_template.tex` preserved.

**Target journals**: J. Cheminformatics (Springer), Digital Discovery (RSC), Mol. Informatics (Wiley).

---

## FULL PROJECT PLAN

### What's Been Achieved

#### Phase 1 — LaTeX Quick Fixes ✅ COMPLETE
All grammar, affiliation, and package fixes applied to `ml_chemistry_template.tex`.

#### Phase 2a — Baseline Experiments on Joung+Beard (18,755 samples)

**v1 results (KFold 80/20, early stopping on test — has data leakage for DL models):**

| Model | Folds | RMSE | MAE | R² | r | Status |
|-------|-------|------|-----|-----|---|--------|
| BiGRU + Solvent | 5/5 | 33.48 ± 0.77 | 18.46 ± 0.11 | 0.9017 ± 0.0054 | 0.9497 ± 0.0029 | ✅ Aggregate done |
| BiGRU no solvent | 5/5 | ~37.85 ± 1.19 | ~20.81 ± 0.56 | ~0.8745 | ~0.9355 | ✅ Folds done, needs `--summary` |
| BiLSTM + Solvent | 2/5 | fold0: 34.41, fold1: 40.71 | — | — | — | ❌ Folds 2-4 missing |
| CNN-BiGRU + Solvent | 1/5 | fold0: 33.43 | — | — | — | ❌ Folds 1-4 missing |
| RF (MSE) | 5/5 | 30.85 ± 0.71 | 14.24 ± 0.27 | 0.9166 ± 0.0040 | 0.9577 ± 0.0019 | ✅ Aggregate done |
| RF (MAE) | 5/5 | 30.92 ± 0.74 | 14.35 ± 0.14 | 0.9162 ± 0.0035 | 0.9576 ± 0.0016 | ✅ Aggregate done |
| XGBoost (MSE) | 5/5 | 32.71 ± 1.14 | 19.51 ± 0.43 | 0.9062 ± 0.0052 | 0.9529 ± 0.0025 | ✅ Aggregate done |
| XGBoost (MAE) | 5/5 | 40.12 ± 2.27 | 21.41 ± 0.77 | 0.8587 ± 0.0148 | 0.9295 ± 0.0074 | ✅ Aggregate done |

**v2 results (StratifiedKFold 64/16/20 train/val/test, proper early stopping — preferred for paper):**

| Model | Folds | RMSE | MAE | R² | r | Status |
|-------|-------|------|-----|-----|---|--------|
| BiGRU + Solvent v2 | 5/5 | 36.45 ± 1.12 | 20.70 ± 0.51 | 0.8836 ± 0.0058 | 0.9402 ± 0.0031 | ✅ Aggregate done |
| BiGRU no solvent v2 | 0/5 | — | — | — | — | 🔄 Running (started 2026-03-01) |
| RF v2 (MSE) | 5/5 | 32.18 ± 1.74 | 15.42 ± 0.47 | 0.9091 ± 0.0095 | 0.9538 ± 0.0049 | ✅ Aggregate done |
| RF v2 no solvent | 5/5 | 34.93 ± 1.40 | 16.77 ± 0.37 | 0.8930 ± 0.0082 | 0.9453 ± 0.0042 | ✅ Aggregate done |
| RF v2 (MAE) | 3/5 | fold0: 33.27, fold1: 31.86, fold2: 30.17 | — | — | — | ❌ Folds 3-4 missing |
| XGBoost v2 (MSE) | 5/5 | 33.70 ± 1.61 | 20.05 ± 0.48 | 0.9003 ± 0.0094 | 0.9496 ± 0.0049 | ✅ Aggregate done |
| XGBoost v2 (MAE) | 5/5 | 40.75 ± 1.58 | 21.99 ± 0.47 | 0.8544 ± 0.0105 | 0.9272 ± 0.0055 | ✅ Aggregate done |

#### Phase 2b — Cross-Dataset Benchmarking (verified from result files)

Single train/val/test splits matching each paper's published protocol. Val used for early stopping (no data leakage).

| Dataset | Split | Published | Our RMSE | Our MAE | Our R² | Status |
|---------|-------|-----------|----------|---------|--------|--------|
| Deep4Chem | random 80/10/10 | GCNN RMSE 26.6 (Joung 2020) | 27.07 | 17.22 | 0.9339 | ✅ Done (very close to published!) |
| Jung 2024 | random 72/18/10 | GBFS RMSE 32.2 (Jung 2024) | 36.31 | 21.54 | 0.8868 | ✅ Done |
| Jung 2024 | scaffold 80/10/10 | GBFS RMSE 32.2 (Jung 2024) | 67.56 | 46.77 | 0.5993 | ✅ Done (scaffold is much harder for 1D models) |
| nablaColors | precomputed scaffold | UniMol+ MAE 15.97 (3D GNN, 27.7M params) | 56.38 | 39.48 | 0.6958 | ✅ Done (3D models have huge advantage) |
| nablaColors SELFIES | precomputed scaffold | — | 56.57 | 40.42 | 0.6938 | ✅ Done |

#### Phase 2c — Classification Validation (verified from result files)

| Task | Status | Result |
|------|--------|--------|
| Mamede classification (from v1 pooled) | ✅ Done | Sens=0.9904, Spec=0.5446, F1=0.9817, AUC=0.9215 (N=3751) |

---

### What Still Needs To Be Done (Benchmark Reframing)

#### Step 1 — BiGRU no-solvent v2 (RUNNING)
🔄 `python3 run_baselines.py --model bigru_nosolvent --v2` — needed for solvent ablation table.
~15-25 hrs GPU. After completion: fill in Table 2 in `benchmark_paper.tex`.

#### Step 2 — Polish `benchmark_paper.tex`
- Fill in BiGRU no-solvent v2 numbers in Table 2
- Verify all citations compile against `Proposal.bib`
- Add any missing BibTeX entries
- Write supplementary material (per-fold breakdown)
- Proofread and final formatting

#### Step 3 (Optional) — Statistical Significance Tests
- Paired t-tests across 5 folds: RF vs BiGRU, solvent vs no-solvent
- Reviewers will likely ask for this

#### Dropped from Plan
- Transfer learning (high risk, uncertain payoff)
- BiLSTM/CNN-BiGRU v2 (incomplete, don't add to benchmark story)
- ChemFluor external validation (no published benchmark)

---

## GPU Setup

**Hardware**: RTX 4090 Laptop GPU, TF 2.20, CUDA 12.7

```bash
# libdevice fix (one-time):
mkdir -p ~/.local/cuda_compat/nvvm/libdevice
ln -sf ~/.local/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc ~/.local/cuda_compat/nvvm/libdevice/libdevice.10.bc
```

**Before DL training**: kill competing processes, check `nvidia-smi`. `verify_gpu_available()` runs automatically.

**Expected epoch times** (batch_size=32, seq_len=649):
- v2 (~12K train, 375 steps/epoch): **~70-85 sec/epoch**, ~3-5 hours per fold
- 18% GPU utilization is NORMAL for RNNs (sequential timestep computation)
- Tqdm callback prints detailed metrics every 10 epochs automatically

## Training Config

batch_size=32, lr=0.001, mixed_float16, epochs=250, patience=25, RMSprop, loss=MAE

## Overleaf + Git

```
git pull origin master → edit locally → git add/commit/push → changes in Overleaf
```
Files: `ml_chemistry_template.tex`, `Proposal.bib` (77K tokens — read with offset/limit), `references_zo.bib`

## File Architecture

- **`paper1_new_cl/`** — Python package
  - `models.py` — compute_metrics, vectorize_smiles, create_dl_model, train_dl_model, verify_gpu_available, get_tqdm_callback
  - `splits.py` — scaffold_split, create_solvent_bins, create_stratified_folds
- **`run_baselines.py`** — 9 models (incl. rf_nosolvent), 5-fold CV, v1/v2 (CLI: `--model`, `--fold`, `--v2`, `--summary`)
- **`run_cross_dataset.py`** — Cross-dataset benchmarks (CLI: `--dataset`, `--split`, `--summary`)
- **`run_selfies_experiment.py`** — SELFIES vs SMILES on nablaColors
- **`eval_classification.py`** — Mamede photosafety classification (CLI: `--model`, `--threshold`)
- **`eval_wetlab.py`** — Wetlab experimental validation (16 molecules × 2 solvents)
- **`download_datasets.py`** — Download external datasets (CLI: `--dataset`)
- **`convert_reaxys_web_export.py`** — Reaxys TSV → raw CSV
- **`postprocess_reaxys.py`** — Raw Reaxys → clean regression + classification datasets (CLI: `--solvent`)
- **`benchmark_paper.tex`** — NEW benchmark-framed paper (preferred for submission)
- **`ml_chemistry_template.tex`** — Original DL-focused paper (preserved, Overleaf-synced)
- **`results/`** — All outputs. **`data/`** — Datasets. **`previous_code/`** — Original work.

## Datasets

- **Primary**: `previous_code/UV_canonical_full_dataset.csv` — 18,755 rows (smiles, lambda_max, canon, solvents)
- **Deep4Chem**: `data/deep4chem_processed.csv` — ~20K (Joung 2020, Figshare)
- **Jung 2024**: `data/jung2024_processed.csv` — ~26K (GitHub)
- **nablaColors**: `data/nablacolors_processed.csv` — 24,567 (Zenodo), splits in `nablacolors_splits.npz`
- **ChemFluor**: `data/chemfluor_processed.csv` — 4,232 entries
- **Reaxys**: `data/raw/reaxys_uv_raw.csv` — 114,699 records (from 24/75 exports, needs re-run after all 75)
- **Reaxys exports**: `data/raw/reaxys_exports/` — 48/75 TSV files collected

## Commands

```bash
pip install -e ".[dev]"      # Install
pytest                        # Tests
ruff check . && ruff format . # Lint
```
