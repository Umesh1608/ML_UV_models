# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Benchmark paper with "complementary strengths" narrative** (March 2026): Systematic benchmark of ML approaches (RF, XGBoost, BiGRU, ChemBERTa) for UV λ_max prediction. Key findings: (1) RF+Morgan FP wins on cross-validated benchmarks (RMSE 31.34), (2) Deep learning (ChemBERTa MAE 26.3, BiGRU 28.6) generalizes better to novel OOD molecules vs RF (38.5), (3) solvent concatenation helps both families (7-8% RMSE reduction), (4) ChemBERTa worst on CV (50.39) but best on wetlab — pretrained representations compensate for architectural mismatch on novel compounds. Model selection depends on task: RF for interpolation/screening, DL for novel compound exploration.

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
| ChemBERTa v2 | 5/5 | 50.39 ± 2.30 | 24.31 ± 1.04 | 0.777 ± 0.019 | 0.899 ± 0.009 | ✅ Aggregate done |

#### Phase 2b — Cross-Dataset Benchmarking (verified from result files)

Single train/val/test splits matching each paper's published protocol. Val used for early stopping (no data leakage).

| Dataset | Split | Published | Our RMSE | Our MAE | Our R² | Status |
|---------|-------|-----------|----------|---------|--------|--------|
| Deep4Chem | random 80/10/10 | GCNN RMSE 26.6 (Joung 2020) | RF 22.70, BiGRU 27.07, CB 35.90 | — | — | ✅ All 3 models done |
| Jung 2024 | random 72/18/10 | GBFS RMSE 32.2 (Jung 2024) | RF 29.82, BiGRU 36.31, CB 38.27 | — | — | ✅ All 3 models done |
| Jung 2024 | scaffold 80/10/10 | GBFS RMSE 32.2 (Jung 2024) | 67.56 | 46.77 | 0.5993 | ✅ Done (scaffold is much harder for 1D models) |
| nablaColors | precomputed scaffold | UniMol+ MAE 15.97 (3D GNN, 27.7M params) | 56.38 | 39.48 | 0.6958 | ✅ Done (3D models have huge advantage) |
| nablaColors SELFIES | precomputed scaffold | — | 56.57 | 40.42 | 0.6938 | ✅ Done |

#### Phase 2c — Classification Validation (verified from result files)

| Task | Status | Result |
|------|--------|--------|
| Mamede classification (from v1 pooled) | ✅ Done | Sens=0.9904, Spec=0.5446, F1=0.9817, AUC=0.9215 (N=3751) |

---

### What Still Needs To Be Done — Publication Roadmap

**Target**: J. Cheminformatics (IF ~8, Springer, open access) or Digital Discovery (RSC)

#### Phase I Remaining Tasks

**A3. BiGRU HPO — #1 reviewer concern** ⏸ PAUSED (25/30 trials complete)
- Optuna TPE, 30 trials on fold 0 val set, SQLite at `results/bigru_hpo.db`
- Search space: units {64,128,256}, layers {1,2,3}, embed {32,50,100}, dropout {0.1-0.4}, lr [1e-4,3e-3], batch {32,64,128}
- **Best config (Trial 25): val_loss=33.33 — 8.6% better than default (36.45)**
  - n_units=256, n_layers=3, embed_dim=50, batch_size=128, dropout=0.105, lr=0.00111
- Key findings: 3L/256u > 2L/128u (default), 1 layer fails, embed=50 optimal, low dropout (~0.10) best for large arch
- Resume HPO: `python3 tune_bigru.py --n-trials 30` (picks up at trial 27, ~10-15h GPU for 5 remaining)
- After HPO: `python3 tune_bigru.py --full-cv` (~25h GPU, trains best config on all 5 folds)

**Remaining Phase I steps (in order):**
1. Finish A3 HPO (5 remaining trials): `python3 tune_bigru.py --n-trials 30`
2. Run full-cv with best config: `python3 tune_bigru.py --full-cv`
3. Re-evaluate wetlab: `python3 eval_wetlab.py`
4. Update paper: Table 2, Table S5, TikZ, discussion
5. A7: BiGRU direct classification on Mamede (~3-5h GPU)
   - Use tuned architecture (256u/3l) with task="classification" (sigmoid + BCE)
   - Train on Mamede/Reaxys ~74K binary labels (POS = λ_max ∈ [290,700] AND MEC ≥ 1000)
   - Compare vs: Mamede RF (Sens=0.90, Spec=0.88) and our RF (Sens=0.876, Spec=0.882)
   - Update Table 4, discussion, conclusion; publish model weights
6. A5: GitHub repo + Zenodo DOI + model weights
7. Final compile + proofread

#### What's DONE ✅

| Task | Commit |
|------|--------|
| A0.1 Factual errors (ChemBERTa params, multirow, TikZ) | 215736e |
| A0.2 Citations (Chen et al. journal, Joung GCNN, scope caveats) | 215736e |
| A0.3 Soften 6 overclaimed statements + confounders paragraph | 215736e |
| A2 Supplementary (XGBoost+ChemBERTa per-fold, significance) | aa8800d |
| A4 4-model parity plot (shared axes) | 3057f34 |
| A6 Full proofread (12+ issues fixed) | aa8800d |
| B3 Wetlab per-molecule figure (horizontal bars) | 0f8043d |
| B4 ChemBERTa learning curves (supplementary) | 3f7b8cf |
| Figure 7 fix (vertical layout) | 189c44e |
| Paper title updated to "When Do Simple Models Win?" | latest |

#### Phase II — Multi-Property Expansion (`benchmark_paper_v2.tex`)

**Script**: `run_multi_property.py` (CLI: `--property {emission,log_mec,lipophilicity} --model {rf,bigru,chemberta} --fold {0-4}`)

**RF results (complete or in progress):**

| Property | Dataset | Rows | RMSE | R² | Status |
|----------|---------|------|------|----|--------|
| Emission (λ_em) | `data/chemfluor_processed.csv` | 4,232 | 25.28 ± 2.81 | 0.936 | ✅ 5/5 |
| Lipophilicity (LogP) | `data/lipophilicity_processed.csv` | 4,200 | 0.83 ± 0.02 | 0.528 | ✅ 5/5 |
| log₁₀(MEC) | `data/mamede_log_mec_processed.csv` | 81,638 | ~0.47 | ~0.45 | ⏸ 3/5 (resume: `--property log_mec --model rf`) |

**Remaining:** Finish log_mec RF (folds 3-4), then BiGRU + ChemBERTa on emission & lipophilicity (~32h GPU)

#### Priority Execution Order
```
Phase I: A3 ⏸ (5 trials left) → full-cv → wetlab → paper updates → A7 → A5 → final proofread
Phase II (after Phase I): log_mec RF ⏸ → BiGRU (GPU) → ChemBERTa (GPU) → paper_v2
```

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

**BiGRU (default)**: batch_size=32, lr=0.001, mixed_float16, epochs=250, patience=25, RMSprop, loss=MAE
**BiGRU (tuned, from HPO Trial 25)**: n_units=256, n_layers=3, embed_dim=50, batch_size=128, dropout=0.105, lr=0.00111, val_loss=33.33 (8.6% better than default 36.45)

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
- **`tune_bigru.py`** — BiGRU Optuna HPO (30 trials, SQLite storage, stop/resume safe)
- **`run_multi_property.py`** — Phase II multi-property experiments (emission, log_mec, lipophilicity)
- **`ROADMAP.txt`** — Full execution guide with commands and stop/resume safety
- **`results/`** — All outputs. **`data/`** — Datasets. **`previous_code/`** — Original work.

## Datasets

- **Primary**: `previous_code/UV_canonical_full_dataset.csv` — 18,755 rows (smiles, lambda_max, canon, solvents)
- **Deep4Chem**: `data/deep4chem_processed.csv` — ~20K (Joung 2020, Figshare)
- **Jung 2024**: `data/jung2024_processed.csv` — ~26K (GitHub)
- **nablaColors**: `data/nablacolors_processed.csv` — 24,567 (Zenodo), splits in `nablacolors_splits.npz`
- **ChemFluor**: `data/chemfluor_processed.csv` — 4,232 entries (emission wavelength)
- **Mamede log₁₀(MEC)**: `data/mamede_log_mec_processed.csv` — 81,638 rows (MEC ∈ [1,500000], 88% methanol)
- **Lipophilicity**: `data/lipophilicity_processed.csv` — 4,200 rows (LogP, MoleculeNet, no solvent)
- **Mamede/Reaxys classification**: ~74K compounds, binary photosafety labels
- **Reaxys**: `data/raw/reaxys_uv_raw.csv` — 114,699 records (from 24/75 exports)
- **Reaxys exports**: `data/raw/reaxys_exports/` — 48/75 TSV files collected

## Commands

```bash
pip install -e ".[dev]"      # Install
pytest                        # Tests
ruff check . && ruff format . # Lint
```
