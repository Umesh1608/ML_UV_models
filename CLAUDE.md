# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Benchmark paper with "complementary strengths" narrative** (March 2026): Systematic benchmark of ML approaches (RF, XGBoost, BiGRU, ChemBERTa) for UV λ_max prediction. Key findings: (1) RF+Morgan FP wins on cross-validated benchmarks, (2) BiGRU generalizes better to novel out-of-distribution molecules (wetlab MAE 28.6 vs 38.5), (3) solvent concatenation helps both families, (4) ChemBERTa pretrained Transformer baseline adds modern context. The paper argues model selection depends on the task: RF for interpolation/screening, BiGRU for novel compound exploration.

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
| BiGRU no solvent v2 | 5/5 | 39.03 ± 1.37 | 22.53 ± 0.32 | 0.870 ± 0.010 | 0.930 ± 0.010 | ✅ Aggregate done |
| **RF TUNED** (B=1000,mf=0.3) | 5/5 | **31.34 ± 1.82** | **15.16 ± 0.44** | 0.914 ± 0.010 | 0.956 ± 0.005 | ✅ Aggregate done |
| RF TUNED no solvent | 5/5 | 34.13 ± 1.44 | 16.45 ± 0.33 | 0.900 ± 0.010 | 0.950 ± 0.004 | ✅ Aggregate done |
| RF v2 (MAE) | 3/5 | fold0: 33.27, fold1: 31.86, fold2: 30.17 | — | — | — | ❌ Folds 3-4 missing |
| XGBoost v2 (MSE) | 5/5 | 33.70 ± 1.61 | 20.05 ± 0.48 | 0.9003 ± 0.0094 | 0.9496 ± 0.0049 | ✅ Aggregate done |
| XGBoost v2 (MAE) | 5/5 | 40.75 ± 1.58 | 21.99 ± 0.47 | 0.8544 ± 0.0105 | 0.9272 ± 0.0055 | ✅ Aggregate done |
| ChemBERTa v2 | 3/5 | ~48.56 (folds 0-2) | ~23.49 | ~0.792 | ~0.905 | ⏸️ Fold 3 paused ep67, fold 4 not started |

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

### What Still Needs To Be Done

#### Current Phase — Transformer Baseline + Paper Improvements (March 2026)
- ✅ Created `run_chemberta.py` — ChemBERTa fine-tuning script (PyTorch)
- ✅ Fixed thermal shutdown: MAX_LEN 512→256 (4x attention savings), BATCH_SIZE 32→8 + grad_accum=4, num_workers=0, gpu_cooldown between folds
- ✅ Fixed figure alignment: subfigure [b]→[t], figure [t]→[htbp], standardized figsize
- ✅ Added "UV = local features" argument (Woodward-Fieser, Kasha's rule) in 5 paper locations
- ✅ Strengthened Chen et al. (2023) connection: RNNs capture local features, Transformers capture global
- ✅ Added 3 new BibTeX entries (Kang 2020, Beard 2022, Jiang 2023 TranGRU)
- ✅ Updated contributions list with local-feature finding
- ✅ Added TikZ diagram (fig:local_vs_global) — local vs global feature alignment
- ✅ ChemBERTa EPOCHS increased 50→100, PATIENCE 10→15 (first run didn't converge)
- ✅ Crash-resilient checkpointing added to `run_chemberta.py` (full checkpoint save/resume, periodic saves every 10 epochs, SIGINT/SIGTERM signal handler, inline result saving)
- ✅ `eval_wetlab.py` updated to handle both old (weights-only) and new (full dict) checkpoint formats
- ✅ ChemBERTa fold 0: RMSE=49.31, MAE=23.88, R²=0.780, r=0.899
- ✅ ChemBERTa fold 1: RMSE=48.66, MAE=23.54, R²=0.794, r=0.908
- ✅ ChemBERTa fold 2: RMSE=47.70, MAE=23.05, R²=0.801, r=0.909
- ✅ ChemBERTa fold 3: RMSE=53.04, MAE=25.68, R²=0.756, r=0.889
- ✅ ChemBERTa fold 4: RMSE=53.24, MAE=25.39, R²=0.756, r=0.889
- ✅ ChemBERTa 5-fold aggregate: RMSE=50.39±2.30, MAE=24.31±1.04, R²=0.777, r=0.899
- ✅ ChemBERTa cross-dataset: Deep4Chem RMSE=35.90, Jung 2024 RMSE=38.27
- ✅ ChemBERTa wetlab: MAE=26.3 [20.1--33.0], RMSE=32.2 [25.2--38.8] (BEST on novel compounds!)
- ✅ All 6 ChemBERTa placeholders filled in benchmark_paper.tex
- ✅ nablaColors removed from paper
- ✅ Text fixes applied (tuning asymmetry, Reaxys, XGBoost MAE, bootstrap CIs, data availability)

#### RESUME INSTRUCTIONS (start here next session)
```bash
# ChemBERTa COMPLETE — all folds, cross-dataset, wetlab done
# Next: proofread + journal formatting
```

#### Step 1 — Final Polish
- Proofread entire paper
- Format for target journal (J. Cheminformatics, Digital Discovery, or Mol. Informatics)
- Ensure all figures render correctly on Overleaf

#### Completed
- ✅ Paper expanded: ~38 citations (was 13), 792+ lines, ~7200 words
- ✅ All citations verified against Proposal.bib (27 new bib entries added)
- ✅ Statistical significance tests: all paired t-tests significant (p < 0.02)
- ✅ Supplementary material: per-fold tables, significance table, RF tuning analysis
- ✅ Figures 3-4 regenerated with v2 data (was showing old v1 RMSE)
- ✅ Related Work expanded (6 subsections, comprehensive literature coverage)
- ✅ BiGRU no-solvent v2: 5/5 folds done, RMSE 39.03±1.37 (solvent improvement = 7%)
- ✅ Table 2 + abstract placeholders filled in benchmark_paper.tex
- ✅ RF hyperparameter tuning (432 configs → B=1000, max_features=0.3)
- ✅ Tuned RF 5-fold CV: RMSE 31.34±1.82 (was 32.18)
- ✅ Tuned RF no-solvent 5-fold CV: RMSE 34.13±1.44
- ✅ Wetlab predictions with tuned RF models
- ✅ "Complementary strengths" narrative revision
- ✅ RF interpretability figures regenerated with tuned model

#### Planned — RNN Architecture & HP Study (after ChemBERTa + polish)
- Architecture comparison: BiGRU vs BiLSTM vs CNN-BiGRU (all implemented, `bilstm_v2` and `cnn_bigru_v2` registered in run_baselines.py)
- BiGRU HP search: Optuna TPE, 30 trials on fold 0
- Search space: units {64,128,256}, layers {1,2,3}, embed {32,50,100}, dropout {0.1-0.4}, lr [1e-4,3e-3], batch {32,64,128}
- New script: `tune_bigru.py` (analogous to `tune_rf.py`)
- Paper: new subsection "RNN Architecture and Hyperparameter Study" after Solvent Ablation
- 5 new BibTeX: Chung 2014, Goh 2017, Grisoni 2020, Bergstra 2012, Akiba 2019
- Addresses tuning asymmetry (RF had 432-config grid, BiGRU gets 30-trial Optuna)

#### Dropped from Plan
- Transfer learning (high risk, uncertain payoff)
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

**BiGRU**: batch_size=32, lr=0.001, mixed_float16, epochs=250, patience=25, RMSprop, loss=MAE

**ChemBERTa** (thermal-safe config): MAX_LEN=256, batch_size=8, grad_accum=4 (effective=32), lr=5e-5, AdamW, fp16, epochs=100, patience=15, num_workers=0, gpu_cooldown=10s between folds. RTX 4090 Laptop idles at 71°C — the old config (batch=32, MAX_LEN=512, 12 worker processes) caused thermal shutdown. First run (50 epochs, patience=10) didn't converge (RMSE=116.79). **Crash-resilient**: full checkpoint format (model+optimizer+scheduler+scaler+epoch+history), periodic saves every 10 epochs to `_latest.pt`, SIGINT/SIGTERM handler saves checkpoint before exit, results saved inside `train_chemberta()` not just caller. Resume auto-detects checkpoint format. ~100-110s/epoch on GPU, ~3h per fold.

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
- **`eval_wetlab.py`** — Wetlab experimental validation (16 molecules × 2 solvents, RF + BiGRU + ChemBERTa)
- **`run_chemberta.py`** — ChemBERTa fine-tuning (PyTorch, HuggingFace) for primary + cross-dataset
- **`download_datasets.py`** — Download external datasets (CLI: `--dataset`)
- **`convert_reaxys_web_export.py`** — Reaxys TSV → raw CSV
- **`postprocess_reaxys.py`** — Raw Reaxys → clean regression + classification datasets (CLI: `--solvent`)
- **`benchmark_paper.tex`** — NEW benchmark-framed paper (preferred for submission)
- **`notes_on_paper.tex`** — Revision notes, reviewer Q&A, submission checklist, fallback git hashes
- **`ml_chemistry_template.tex`** — Original DL-focused paper (preserved, Overleaf-synced)
- **`tune_rf.py`** — RF hyperparameter grid search (432 configs)
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
