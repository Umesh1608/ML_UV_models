# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Benchmark paper with "complementary strengths" narrative** (March 2026): Systematic benchmark of ML approaches (RF, XGBoost, Chemprop D-MPNN, BiGRU, ChemBERTa) for UV λ_max prediction. Key findings: (1) RF+Morgan FP wins on cross-validated benchmarks (RMSE 31.34), (2) Deep learning (ChemBERTa MAE 26.3, BiGRU 28.6) generalizes better to novel OOD molecules vs RF (38.5), (3) solvent concatenation helps both families (7-8% RMSE reduction), (4) ChemBERTa worst on CV (50.39) but best on wetlab — pretrained representations compensate for architectural mismatch on novel compounds. Model selection depends on task: RF for interpolation/screening, DL for novel compound exploration.

**Target journal**: **J. Chem. Inf. Model. (JCIM)** — ACS, IF ~6.5, achemso format (`benchmark_paper_jcim.tex`).
Also have `benchmark_paper.tex` (generic article format, Overleaf-synced).

**Current state (May 2026): JCIM revision in progress.** Manuscript ID ci-2026-009433 received "Major Revisions" decision on 20-May-2026, deadline 17-Jun-2026. Resubmission deliverables live as **new files** (`benchmark_paper_jcim_revised.tex`, `benchmark_paper_jcim_revised_marked.tex`, `supporting_information_revised.tex`, `response_to_reviewers.tex`); originals are kept untouched. **Read `revision_status.md` and `reviewer_comments.md` before continuing the revision work** — they capture which reviewer concerns are completed, which is next, and the conventions for marked-copy edits and the response document.

---

## Phase I: Primary Benchmark ✅ COMPLETE (ready for JCIM submission)

5 model families benchmarked on Joung+Beard dataset (18,755 UV λ_max samples).

### v2 Cross-Validated Results (StratifiedKFold 64/16/20 — used in paper)

| Model | RMSE (nm) | MAE | R² | Status |
|-------|-----------|-----|-----|--------|
| RF TUNED (B=1000,mf=0.3) | **31.34 ± 1.82** | **15.16 ± 0.44** | 0.914 | ✅ |
| Chemprop D-MPNN | 31.69 ± 3.18 | 16.84 ± 1.74 | 0.911 | ✅ |
| XGBoost MSE | 33.70 ± 1.61 | 20.05 ± 0.48 | 0.900 | ✅ |
| BiGRU TUNED (3L/256u) | 34.71 ± 1.40 | 18.09 ± 0.59 | 0.894 | ✅ |
| BiGRU+Solvent (default) | 36.45 ± 1.12 | 20.70 ± 0.51 | 0.884 | ✅ |
| ChemBERTa | 50.39 ± 2.30 | 24.31 ± 1.04 | 0.777 | ✅ |

### Cross-Dataset Benchmarking ✅

| Dataset | Split | Our Best RMSE | Status |
|---------|-------|---------------|--------|
| Deep4Chem | random 80/10/10 | RF 22.70 | ✅ |
| Jung 2024 | random 72/18/10 | RF 29.82 | ✅ |
| Jung 2024 | scaffold 80/10/10 | RF 67.56 | ✅ |
| nablaColors | precomputed scaffold | RF 56.38 | ✅ |

### Wetlab Validation ✅ (16 molecules × 2 solvents)

| Model | MAE | RMSE |
|-------|-----|------|
| ChemBERTa | **26.3** | 32.2 |
| BiGRU (default) | 28.6 | 33.9 |
| RF | 38.5 | 43.7 |

### Deliverables ✅
Cover letter, TOC graphic, reviewer suggestions, all figures in PNG, BiGRU classification, citation verification, SI per-fold tables.
**Remaining**: commit+push, Overleaf sync, submit to JCIM (ACS Paragon Plus).

---

## Phase II: Multi-Property Expansion ⏸ PAUSED

Tests locality-of-property hypothesis: local-bias models (RF, Chemprop) should dominate on LOCAL properties while global-attention models (ChemBERTa) compete on GLOBAL properties.

**Script**: `run_multi_property.py` (CLI: `--property {emission,log_mec,lipophilicity,solubility} --model {rf,xgboost,chemprop,bigru,chemberta} --fold {0-4}`)

### Completed Results

**Emission (λ_em) — LOCAL, with solvent, 4,232 rows (`data/chemfluor_processed.csv`):**

| Model | RMSE | MAE | R² | Folds |
|-------|------|-----|-----|-------|
| Chemprop | **24.03 ± 1.95** | 14.77 ± 0.70 | 0.943 | 5/5 ✅ |
| XGBoost | 24.57 ± 1.63 | 13.77 ± 0.51 | 0.940 | 5/5 ✅ |
| RF | 25.28 ± 2.81 | 13.06 ± 1.11 | 0.936 | 5/5 ✅ |
| BiGRU | 31.87 ± 2.03 | 19.40 ± 0.80 | 0.900 | 5/5 ✅ |
| ChemBERTa | — | — | — | 0/5 ❌ |

**Lipophilicity (LogP) — GLOBAL, no solvent, 4,200 rows (`data/lipophilicity_processed.csv`):**

| Model | RMSE | MAE | R² | Folds |
|-------|------|-----|-----|-------|
| Chemprop | **0.61 ± 0.03** | 0.43 ± 0.02 | 0.745 | 5/5 ✅ |
| XGBoost | 0.79 ± 0.02 | 0.60 ± 0.02 | 0.568 | 5/5 ✅ |
| RF | 0.83 ± 0.02 | 0.63 ± 0.01 | 0.528 | 5/5 ✅ |
| BiGRU | 0.84 ± 0.02 | 0.62 ± 0.01 | 0.517 | 5/5 ✅ |
| ChemBERTa | — | — | — | 0/5 ❌ |

**Solubility (logS) — GLOBAL, no solvent (`data/aqsoldb_processed.csv`):**

| Model | RMSE | MAE | R² | Folds |
|-------|------|-----|-----|-------|
| Chemprop | **1.08 ± 0.03** | 0.73 ± 0.02 | 0.794 | 5/5 ✅ |
| XGBoost | 1.33 ± 0.03 | 0.97 ± 0.02 | 0.686 | 5/5 ✅ |
| RF | 1.34 ± 0.02 | 0.95 ± 0.02 | 0.679 | 5/5 ✅ |
| BiGRU | — | — | — | 1/5 (fold 0 running) |
| ChemBERTa | — | — | — | 0/5 ❌ |

**log₁₀(MEC) — LOCAL, with solvent, 81,638 rows (`data/mamede_log_mec_processed.csv`):**

| Model | RMSE | MAE | R² | Folds |
|-------|------|-----|-----|-------|
| RF | **0.47 ± 0.00** | 0.26 ± 0.00 | 0.461 | 5/5 ✅ |
| XGBoost | 0.50 ± 0.01 | 0.30 ± 0.00 | 0.387 | 5/5 ✅ |

*(log_mec too large for BiGRU/ChemBERTa — RF/XGBoost only)*

### Key Finding So Far
RF-to-Chemprop gap is 2-3× larger on GLOBAL properties (lipophilicity, solubility) than LOCAL (emission). Supports locality hypothesis in relative terms. Chemprop dominates everywhere. Still need ChemBERTa results.

### Remaining for Phase II
1. Solubility BiGRU folds 1-4
2. Emission ChemBERTa folds 0-4
3. Lipophilicity ChemBERTa folds 0-4
4. Solubility ChemBERTa folds 0-4
5. Run `--summary` aggregation after all jobs complete
6. Potential: extinction coefficient from Deep4Chem raw (8,280 rows with log₁₀(ε))

### OOM/Crash Prevention (learned the hard way)
- RF `n_jobs=4` for datasets >20K rows (WSL crashed with `n_jobs=-1` on 81K × 32 cores)
- XGBoost uses `device="cuda"` for datasets >20K rows
- BiGRU has checkpoint/resume: `_latest.keras` + `_state.json` saved every 5 epochs
- ChemBERTa has full checkpoint format (model+optimizer+scheduler+scaler+epoch+history)

---

## Verified External Facts (from PDF verification — DO NOT change without re-checking source)

- **nablaColors** (Potapov 2026): 26,369 pairs, best model = **UniProp** (NOT UniMol+), RMSE=27.2, MAE=15.97, R²=0.929
- **ChemBERTa**: `seyonec/ChemBERTa-zinc-base-v1` pretrained on **ZINC** (NOT 77M PubChem)
- **Liu et al. 2023**: MTBG = **BiGRU + GraphSAGE hybrid**, we use only BiGRU component
- **Mamede 2021**: Scientific Reports (NOT Chem Res Toxicol), author Florbela (NOT Filipe)
- **UV-adVISor 2021**: Urbina, Batra, Ekins (NOT Beard et al.), volume 93
- **Lupo Pasini 2023**: Mehta, Yoo, Irle (NOT Li, Blaiszik), page 546

---

## GPU Setup

**Hardware**: RTX 4090 Laptop GPU, TF 2.20, CUDA 12.7

```bash
# libdevice fix (one-time):
mkdir -p ~/.local/cuda_compat/nvvm/libdevice
ln -sf ~/.local/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc ~/.local/cuda_compat/nvvm/libdevice/libdevice.10.bc
```

**Before DL training**: kill competing processes, check `nvidia-smi`. `verify_gpu_available()` runs automatically.

**Expected epoch times**:
- Primary dataset (~12K train, seq_len=649): ~70-85 sec/epoch per BiGRU fold
- Phase II datasets (~2.7K train, seq_len~275): ~11-39 sec/epoch per BiGRU fold
- Tqdm callback prints detailed metrics every 10 epochs automatically

## Training Config

**BiGRU (default)**: batch_size=32, lr=0.001, mixed_float16, epochs=250, patience=25, RMSprop, loss=MAE
**BiGRU (tuned, from HPO Trial 25)**: n_units=256, n_layers=3, embed_dim=50, batch_size=128, dropout=0.105, lr=0.00111

**ChemBERTa** (thermal-safe config): MAX_LEN=256, batch_size=8, grad_accum=4 (effective=32), lr=5e-5, AdamW, fp16, epochs=100, patience=15, num_workers=0, gpu_cooldown=10s between folds. ~100-110s/epoch on GPU, ~3h per fold.

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
- **`run_chemprop.py`** — Chemprop D-MPNN training (Lightning, CLI: `--fold {0-4}`, `--summary`, `--dataset deep4chem`)
- **`run_multi_property.py`** — Phase II multi-property experiments (emission, log_mec, lipophilicity, solubility)
- **`download_datasets.py`** — Download external datasets (CLI: `--dataset`)
- **`convert_reaxys_web_export.py`** — Reaxys TSV → raw CSV
- **`postprocess_reaxys.py`** — Raw Reaxys → clean regression + classification datasets (CLI: `--solvent`)
- **`benchmark_paper.tex`** — Benchmark paper (generic article format, Overleaf-synced, includes inline appendix)
- **`benchmark_paper_jcim.tex`** — JCIM submission version (achemso, journal=jcisd8, main text only)
- **`supporting_information.tex`** — JCIM Supporting Information (standalone achemso suppinfo)
- **`notes_on_paper.tex`** — Revision notes, reviewer Q&A, submission checklist
- **`ml_chemistry_template.tex`** — Original DL-focused paper (preserved, Overleaf-synced)
- **`tune_rf.py`** — RF hyperparameter grid search (432 configs)
- **`tune_bigru.py`** — BiGRU Optuna HPO (25/30 trials done, SQLite storage, stop/resume safe)
- **`analyze_bigru_saliency.py`** — BiGRU gradient saliency → atom-level 2D heatmaps
- **`analyze_rf_interpretability.py`** — RF feature importance analysis (Morgan FP bit → substructure)
- **`run_phase2_gpu.sh`** — Batch script for Phase II GPU jobs (sequential)
- **`run_phase2_cpu.sh`** — Batch script for Phase II CPU jobs
- **`ROADMAP.txt`** — Full execution guide with commands and stop/resume safety
- **`results/`** — All outputs. **`data/`** — Datasets. **`previous_code/`** — Original work.

## Datasets

- **Primary**: `previous_code/UV_canonical_full_dataset.csv` — 18,755 rows (smiles, lambda_max, canon, solvents)
- **Deep4Chem**: `data/deep4chem_processed.csv` — ~20K (Joung 2020, Figshare). Raw has 14 columns incl. extinction coeff.
- **Jung 2024**: `data/jung2024_processed.csv` — ~26K (GitHub)
- **nablaColors**: `data/nablacolors_processed.csv` — 24,567 (Zenodo), splits in `nablacolors_splits.npz`
- **ChemFluor**: `data/chemfluor_processed.csv` — 4,232 entries (emission wavelength)
- **Mamede log₁₀(MEC)**: `data/mamede_log_mec_processed.csv` — 81,638 rows (MEC ∈ [1,500000], 88% methanol)
- **Lipophilicity**: `data/lipophilicity_processed.csv` — 4,200 rows (LogP, MoleculeNet, no solvent)
- **AqSolDB**: `data/aqsoldb_processed.csv` — aqueous solubility (logS, no solvent)
- **Mamede/Reaxys classification**: ~74K compounds, binary photosafety labels
- **Reaxys**: `data/raw/reaxys_uv_raw.csv` — 114,699 records (from 24/75 exports)

## Commands

```bash
pip install -e ".[dev]"      # Install
pytest                        # Tests
ruff check . && ruff format . # Lint
```
