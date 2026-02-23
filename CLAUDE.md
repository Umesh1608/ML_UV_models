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

### COMPLETED (Phase 1 — Quick LaTeX fixes)
All applied to `ml_chemistry_template.tex`:
- ✅ F1: Fixed "Solubility prediction" paragraph → rewritten about UV absorption/solvent effects (line ~509)
- ✅ F2: Fixed author affiliation `$^1$` → `$^2$` for Midwest Bioprocessing Center (line 42)
- ✅ F3a: Removed stray backslash after "(LSTM)" (line 94)
- ✅ F3b: Fixed grammar "is depended on" → "depends on", double period, smart quotes (line 108)
- ✅ F4: Fixed informal tone "and others we tried as well" → "as well as alternative architectures evaluated" (line 263)
- ✅ F5: Removed unused `\usepackage{algorithm}` and `\usepackage{algpseudocode}` (lines 11-12)

### IN PROGRESS (Phase 2 — Baseline Experiments)

**Script**: `run_baselines.py` — runs 6 models on same data split (90/10, seed=7)

**GPU setup**: RTX 4090 Laptop GPU. Requires:
```bash
# libdevice fix for TF/XLA on this machine:
mkdir -p ~/.local/cuda_compat/nvvm/libdevice
ln -sf ~/.local/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc ~/.local/cuda_compat/nvvm/libdevice/libdevice.10.bc
# The run_baselines.py script sets XLA_FLAGS automatically
```

**Current settings**: batch_size=256, lr=0.005, mixed_float16, epochs=250, patience=25

**Completed models (2 of 6)**:
| Model | RMSE | MAE | R² | r |
|-------|------|-----|-----|---|
| BiGRU + Solvent | 36.21 | 22.83 | 0.8788 | 0.9375 |
| BiGRU (no solvent) | 38.26 | 23.67 | 0.8646 | 0.9301 |

Training histories saved to `results/BiGRU_w_Solvent_history.json` and `results/BiGRU_no_solvent_history.json`.

**Remaining models to run**:
- BiLSTM + Solvent (was ~epoch 32 when stopped, ~35s/epoch)
- CNN-BiGRU + Solvent (not started)
- Random Forest + Morgan FP (not started, ~5 min)
- XGBoost + Morgan FP (not started, ~5 min)

**To resume**: Just run `python3 run_baselines.py` — it will re-run all 6 models from scratch (no checkpointing). Each DL model takes ~50-60 min on GPU. Total: ~3-4 hours.

**Note on RMSE values**: The batch_size=256 + lr=0.005 configuration produces slightly higher RMSE than the original paper's batch_size=32 + lr=0.001 (36.21 vs ~20.34 RMSE). Consider reverting to batch_size=32 if you need results closer to the paper's reported values (but training will take ~3x longer per model, i.e. ~2.5 min/epoch vs ~35s/epoch).

### NOT STARTED (Phase 3 — Paper Writing Updates)
After experiments finish, these changes go into `ml_chemistry_template.tex`:
- Add comparison results table (all 6 models, RMSE/MAE/R²/r)
- Add error analysis subsection with figures (parity plot, error distribution, error by wavelength, error by solvent)
- Expand Related Work with UV-specific ML papers (Kneiding et al., Ju et al. 2021, etc.)
- Add limitations paragraph (test set as validation, random split, no 3D info, single seed)
- Fill in Data and Code Availability section
- Add new BibTeX entries to `Proposal.bib`

### NOT STARTED (Phase 4 — Final Polish)
- Proofread entire paper
- Verify all cross-references compile
- Format for target journal

## Baseline Experiments

The `run_baselines.py` script in the repo root runs all comparison models:

```bash
# Run all 6 baselines (GPU recommended, ~3-4 hours total)
python3 run_baselines.py

# Results go to results/ directory:
#   results/baseline_comparison.csv  — final metrics table
#   results/predictions.npz          — all predictions for error analysis
#   results/*_history.json           — training curves
#   results/parity_plot.png/pdf      — predicted vs actual scatter
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
- **`run_baselines.py`** — Baseline experiment script (6 models)
- **`results/`** — Experiment outputs (created by run_baselines.py)
- **`previous_code/`** — Original code and datasets from prior work
