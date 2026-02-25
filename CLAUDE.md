# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository supports a research paper on **ML-based UV absorption (λ_max) prediction** using SMILES molecular representations. The key contribution is a **solvent concatenation strategy** — encoding solute+solvent as a single SMILES sequence with delimiters — which achieves a 16.5% RMSE improvement over solute-only models. The model architecture is a **BiGRU** (bidirectional GRU) operating on tokenized SMILES.

The paper includes experimental validation with real UV-Vis spectra on compounds like avobenzone, ferulic acid, oxybenzone, octisalate, and homosalate.

**Target journals** (in order of fit): Journal of Chemical Information and Modeling (ACS), Journal of Cheminformatics (Springer), Molecular Informatics (Wiley), Digital Discovery (RSC).

## Overleaf + Git Workflow

The paper LaTeX source lives on Overleaf with Git sync enabled. The workflow is:

```
git pull origin master   →   edit locally   →   git add/commit/push   →   changes appear in Overleaf
```

Key LaTeX files (from Overleaf):
- `ml_chemistry_template.tex` — Main paper source
- `Proposal.bib`, `references_zo.bib` — Bibliography files
- Image assets: `p_1.png`, `UV.jpg`, various molecule/architecture diagrams

## Datasets

- **Joung 2020** and **Beard 2019** databases, merged into ~38,000+ UV absorption records
- Each record: solute SMILES, solvent, measured λ_max (nm)
- **Primary dataset file**: `previous_code/UV_canonical_full_dataset.csv` — columns: `smiles`, `lambda_max`, `canon`, `solvents` (18,755 rows after dropna)

## Current Session State (Resume Here)

### Next Steps (priority order)
1. Finish bigru_solvent folds 2–4
2. Run remaining 5 models × 5 folds
3. `python3 run_baselines.py --summary` → comparison table + plots
4. Export remaining 51 Reaxys batches → converter → postprocessor
5. Transfer learning: pre-train on Reaxys, fine-tune on Joung+Beard (Phase 2b)
6. `python3 eval_external.py` on ChemFluor
7. `python3 eval_classification.py` on Mamede
8. Phase 3: write results into paper
9. Phase 4: polish

### COMPLETED (Phase 1 — Quick LaTeX fixes)
All applied to `ml_chemistry_template.tex`:
- ✅ F1: Fixed "Solubility prediction" paragraph → rewritten about UV absorption/solvent effects (line ~509)
- ✅ F2: Fixed author affiliation `$^1$` → `$^2$` for Midwest Bioprocessing Center (line 42)
- ✅ F3a: Removed stray backslash after "(LSTM)" (line 94)
- ✅ F3b: Fixed grammar "is depended on" → "depends on", double period, smart quotes (line 108)
- ✅ F4: Fixed informal tone "and others we tried as well" → "as well as alternative architectures evaluated" (line 263)
- ✅ F5: Removed unused `\usepackage{algorithm}` and `\usepackage{algpseudocode}` (lines 11-12)

### IN PROGRESS (Phase 2 — Baseline Experiments with 5-Fold CV)

**Script**: `run_baselines.py` — runs 6 models with 5-fold cross-validation (KFold, shuffle=True, seed=7)

**GPU setup**: RTX 4090 Laptop GPU. Requires:
```bash
# libdevice fix for TF/XLA on this machine:
mkdir -p ~/.local/cuda_compat/nvvm/libdevice
ln -sf ~/.local/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc ~/.local/cuda_compat/nvvm/libdevice/libdevice.10.bc
# The run_baselines.py script sets XLA_FLAGS automatically
```

**Current settings**: batch_size=80, lr=0.001, mixed_float16, epochs=250, patience=25

**Progress**:
- bigru_solvent: fold 0 ✅ (RMSE 33.2), fold 1 ✅ (RMSE 32.8), folds 2–4 pending
- Other 5 models (bigru_nosolvent, bilstm, cnn_bigru, rf, xgboost): not started

**To resume** (folds are fully independent and resumable; completed folds auto-skip):
```bash
python3 run_baselines.py --model bigru_solvent --fold 2   # next fold
python3 run_baselines.py --model bigru_solvent             # remaining folds (auto-skips done ones)
python3 run_baselines.py                                    # all 6 models, 5-fold CV
python3 run_baselines.py --summary                          # regenerate table + plots
```

**External validation**: `eval_external.py` now uses ensemble of 5 fold models (mean prediction + uncertainty).

### NOT STARTED (Phase 2b — Transfer Learning)

**Concept**: Pre-train the best model (BiGRU+Solvent) on the larger ~74k Reaxys/Mamede dataset, then fine-tune on the curated 18.7k Joung+Beard dataset. Compare against training from scratch (Phase 2 baseline).

**Prerequisites** (must be done first):
1. All 75 Reaxys exports collected and converted (currently 24/75 done)
2. `python3 postprocess_reaxys.py --solvent all` to get clean regression dataset
3. Phase 2 complete (to establish the baseline to beat)

**Implementation plan** (new script `run_transfer.py`):
1. Load Reaxys dataset via `data/mamede_regression_dataset.csv` (output of postprocessor)
2. Build charset as **superset** of Reaxys + Joung+Beard vocabularies (same embed_dim=50)
3. Pre-train BiGRU+Solvent on Reaxys data (same architecture: 2×BiGRU-128, Dense-128, dropout 0.2)
   - Optimizer: RMSprop, lr=0.001, loss=MAE
   - Early stopping patience=25, max 250 epochs
4. Fine-tune on Joung+Beard 18.7k dataset with **5-fold CV** (same folds as Phase 2 for fair comparison)
   - Option A: Lower lr (e.g. 1e-4) on all layers
   - Option B: Freeze embedding + first BiGRU, train rest at normal lr
   - Try both, report best
5. Compare RMSE/MAE/R² against Phase 2 baseline

**Key constraint**: Tokenization must be compatible — the charset from pre-training must include all characters in the fine-tuning dataset. Use the union charset for both stages.

### Reaxys Data Pipeline

**Status**: 24/75 TSV exports converted → `data/raw/reaxys_uv_raw.csv` (114,699 records, 24,000 XRNs). 51 more exports needed (daily Reaxys limit ≈ 25).

After all 75 exports are collected: re-run converter → postprocess → regression + classification datasets.

```bash
python3 convert_reaxys_web_export.py          # TSV → raw CSV
python3 postprocess_reaxys.py --solvent all   # raw → clean datasets
```

### NOT STARTED (Phase 3 — Paper Writing Updates)
After experiments finish, these changes go into `ml_chemistry_template.tex`:
- Add comparison results table (all 6 models, RMSE/MAE/R²/r)
- Add error analysis subsection with figures (parity plot, error distribution, error by wavelength, error by solvent)
- Expand Related Work with UV-specific ML papers (Kneiding et al., Ju et al. 2021, etc.)
- Add limitations paragraph (no 3D info, solvent representation limited to SMILES)
- Fill in Data and Code Availability section
- Add new BibTeX entries to `Proposal.bib`

### NOT STARTED (Phase 4 — Final Polish)
- Proofread entire paper
- Verify all cross-references compile
- Format for target journal

## Baseline Experiments

The `run_baselines.py` script in the repo root runs all comparison models with 5-fold CV:

```bash
# Run all 6 baselines, 5-fold CV (GPU recommended, ~29 hrs total)
python3 run_baselines.py

# Run one model, all 5 folds
python3 run_baselines.py --model bigru_solvent

# Run one model, one fold (for testing/resuming)
python3 run_baselines.py --model bigru_solvent --fold 0

# Regenerate summary table + plots from saved results
python3 run_baselines.py --summary

# Results go to results/ directory:
#   results/cv_fold_indices.npz                    — fold splits for reproducibility
#   results/{model_key}_config.json                — global config (once per model)
#   results/{model_key}_fold{0-4}_metrics.json     — per-fold metrics
#   results/{model_key}_fold{0-4}_model.keras      — per-fold Keras models
#   results/{model_key}_fold{0-4}_model.joblib     — per-fold RF/XGB models
#   results/{model_key}_cv_aggregate.json           — mean +/- std metrics
#   results/{model_key}_cv_pooled.npz               — all predictions pooled
#   results/baseline_comparison.csv                 — summary table (mean +/- std)
#   results/parity_plot.png/pdf                     — from pooled predictions
#   results/error_distribution.png/pdf
#   results/error_by_wavelength.png/pdf
#   results/error_by_solvent.png/pdf
```

## Commands

```bash
# Install package with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file/function
pytest tests/test_example.py
pytest tests/test_example.py::test_function_name

# Lint and format
ruff check .
ruff format .
```

## Architecture

- **`paper1_new_cl/`** — Main Python package source
- **`tests/`** — Test directory (pytest)
- **`pyproject.toml`** — Build config, dependencies, tool settings
- **`run_baselines.py`** — Baseline experiment script (6 models, 5-fold CV)
- **`convert_reaxys_web_export.py`** — Converts Reaxys TSV web exports → `data/raw/reaxys_uv_raw.csv`
- **`eval_external.py`** — External validation on ChemFluor (ensemble of 5 fold models)
- **`eval_classification.py`** — Classification evaluation on Mamede dataset
- **`results/`** — Experiment outputs (created by run_baselines.py)
- **`data/`** — Processed datasets and raw data files
- **`previous_code/`** — Original code and datasets from prior work
