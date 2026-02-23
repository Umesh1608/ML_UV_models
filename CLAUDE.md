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

## Baseline Experiments

A `baseline_experiments/` directory (from a provided tarball) contains scripts for running comparison models:

```bash
# Step 1: Prepare merged dataset
python prepare_data.py --joung joung2020.csv --beard beard2019.csv

# Step 2: Fingerprint baselines (RF, XGBoost, SVR, Ridge) — ~10 min, no GPU
python run_fp_baselines.py --data data/combined_uv_data.csv --include-solvent

# Step 3: Deep learning baselines (GRU, LSTM, BiGRU, BiLSTM) — ~1-2 hrs with GPU
python run_dl_baselines.py --data data/combined_uv_data.csv --epochs 150 --gpu

# Step 4: Generate comparison table + plots
python generate_results.py
```

All baselines share the same **scaffold-based split** (Bemis-Murcko) saved to `split_indices.json` for fair comparison. Solvent name→SMILES mappings are in `utils/data_loader.py` (~40 solvents; may need extending).

## Paper Improvement Plan (Publication Readiness)

### Phase 1: Experiments
- Run baseline comparisons (RF, XGBoost, Ridge, LSTM, BiLSTM) on same scaffold split
- Report RMSE, MAE, R², and Pearson correlation for all models
- Create parity plots (predicted vs. experimental λ_max)
- Error analysis: distribution, error vs. wavelength range, error by solvent type

### Phase 2: Writing Overhaul
- Cut irrelevant sections 1.1–1.4 (HOMO-LUMO, logP, SAS, NP Score) — move to supplementary
- Remove all commented-out LaTeX
- Fix "Integration of Solubility" → "Integration of Solvent" title error (Section 4.5)
- Add "Training Details" subsection (optimizer, LR, batch size, early stopping, hardware)
- Expand Related Work with UV absorption prediction papers
- Add comparison table as centerpiece of Results
- Fix duplicate figure labels (`\label{fig:myfig}` reused) and cross-references
- Add Data and Code Availability section

### Phase 3: Polish
- Add supplementary material
- Format for target journal
- Remove the active learning diagram (Figure 7) — never discussed in the paper

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
