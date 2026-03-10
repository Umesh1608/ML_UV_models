# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Benchmark paper with "complementary strengths" narrative** (March 2026): Systematic benchmark of ML approaches (RF, XGBoost, BiGRU, ChemBERTa) for UV λ_max prediction. Key findings: (1) RF+Morgan FP wins on cross-validated benchmarks (RMSE 31.34), (2) Deep learning (ChemBERTa MAE 26.3, BiGRU 28.6) generalizes better to novel OOD molecules vs RF (38.5), (3) solvent concatenation helps both families (7-8% RMSE reduction), (4) ChemBERTa worst on CV (50.39) but best on wetlab — pretrained representations compensate for architectural mismatch on novel compounds. Model selection depends on task: RF for interpolation/screening, DL for novel compound exploration.

**New LaTeX file**: `benchmark_paper.tex` (benchmark framing). Original `ml_chemistry_template.tex` preserved.

**Target journal**: **J. Chem. Inf. Model. (JCIM)** — ACS, IF ~6.5, achemso format (`benchmark_paper_jcim.tex`).
Also have `benchmark_paper.tex` (generic article format, Overleaf-synced).

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

**Target**: **J. Chem. Inf. Model. (JCIM)** — ACS, achemso format

#### Phase I Remaining Tasks — Execution Order

**A3 full-cv**: 🔄 RUNNING — folds 0-3 complete, fold 4 training (epoch ~100, val_loss=19.62)
- Folds 0-3 RMSE: 35.42, 35.51, 32.02, 35.91 → mean 34.72 ± 1.57 (~5% better than default 36.45)
- Fold 4 still improving

**After fold 4 completes (in order):**
1. **Collect 5-fold aggregate** — read metrics, compute mean±std, save `bigru_tuned_cv_aggregate.json`
2. **Re-evaluate wetlab** with tuned BiGRU — `python3 eval_wetlab.py` (already has `predict_bigru_tuned()`)
3. **Regenerate saliency** with tuned models — `python3 analyze_bigru_saliency.py --tuned`
4. **Update paper numbers** in BOTH `benchmark_paper.tex` AND `benchmark_paper_jcim.tex`:
   - Table 2: add/update BiGRU (tuned) row with full 5-fold results
   - Table S6 (tuning asymmetry): replace fold-0-only with full CV numbers
   - TikZ diagrams: update RMSE values if needed
   - Abstract, discussion, conclusion: update BiGRU RMSE claims
   - Saliency paragraph: update if tuned models change the story
5. **Update SI file** — `supporting_information.tex` with final tuned BiGRU numbers
6. **A7: BiGRU direct classification on Mamede** (~3-5h GPU)
   - Tuned architecture (256u/3l) with task="classification" (sigmoid + BCE)
   - Train on Mamede/Reaxys ~74K binary labels (POS = λ_max ∈ [290,700] AND MEC ≥ 1000)
   - Compare vs: Mamede RF (Sens=0.90, Spec=0.88) and our RF (Sens=0.876, Spec=0.882)
   - Update Table 4, discussion, conclusion in both tex files
7. **Convert figure formats** — .pdf figures → .eps or high-res .png (ACS accepts TIF/JPG/PNG/EPS)
8. **Create TOC graphic** — 3.25" × 1.75" summary graphic (recommended for Articles)
9. **A5: GitHub repo finalization + Zenodo DOI + model weights**
10. **Write cover letter** (separate document, required for ACS submission)
11. **Prepare 5 reviewer suggestions** with academic email addresses
12. **Final proofread + compile** both tex files
13. **Submit to JCIM** via ACS Paragon Plus (need ORCID iDs for all authors)

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
| A8 BiGRU saliency interpretability (gradient → atom heatmaps) | done |
| A3 HPO stopped at 25/30 trials (sufficient), best config saved | done |
| Architecture diagrams: 3-panel RF/BiGRU/ChemBERTa overview (fig:model_overview) | done |
| Architecture diagrams: 2-panel default vs optimized BiGRU (fig:bigru_architectures) | done |
| HPO methodology paragraph in Section 3.3 (Optuna search description) | done |
| Supplementary tuning asymmetry table: added tuned BiGRU row (33.33, -8.6%) | done |
| Appendix float placement: all `[h]`/`[htbp]` → `[H]`, `\clearpage` before bib | done |
| 3D sensitivity claim reframed (1D models competitive except nablaColors) | done |
| GitHub repo files: README.md, requirements.txt, .gitignore updated | done |
| Bib entry: akiba2019optuna (Optuna KDD 2019) | done |
| GCNN citation fix: joung2020experimental → joung2021deep (all locations) | 935d12d |
| SMILES2Vec citation moved to saliency intro (removed from triangulation) | 6320547 |
| Saliency figure redesigned: green/red grouped, prominent error labels | 7d5a8c2 |
| **JCIM paper created**: `benchmark_paper_jcim.tex` (achemso, journal=jcisd8) | f8c43e5 |
| eval_wetlab.py updated: added predict_bigru_tuned() for tuned models | a45130d |
| Bib entry: joung2021deep (JACS Au 2021, GCNN RMSE=26.6 source) | 935d12d |
| JCIM abstract condensed: 8 → 4 sentences, emphasizes optimized BiGRU + broader guidance | edb5fd4 |
| Data curation expanded: 4-step procedure in Section 3.1 (both tex files) | edb5fd4 |
| SI file created: `supporting_information.tex` (standalone achemso suppinfo) | edb5fd4 |
| Conflict of interest declaration added to Acknowledgements | b26c9e7 |
| QSAR best practices cited: tropsha2006best + cherkasov2014qsar in evaluation protocol | b26c9e7 |
| Abstract broadened: locality-of-property principle for model selection beyond UV | ea808f4 |
| Conclusion reframed: practical guidance for molecular property prediction generally | ea808f4 |

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
Phase I: [fold 4 finishes] → collect aggregate → wetlab re-eval → saliency --tuned
  → paper number updates (both tex + SI) → A7 classification (GPU)
  → convert figures → TOC graphic → A5 repo/Zenodo → cover letter
  → reviewer suggestions → final proofread → submit JCIM
Phase II (after submission): log_mec RF ⏸ → BiGRU (GPU) → ChemBERTa (GPU) → paper_v2
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
- **`benchmark_paper.tex`** — Benchmark paper (generic article format, Overleaf-synced, includes inline appendix)
- **`benchmark_paper_jcim.tex`** — JCIM submission version (achemso, journal=jcisd8, main text only)
- **`supporting_information.tex`** — JCIM Supporting Information (standalone achemso suppinfo: per-fold tables, significance, learning curves, tuning asymmetry, chartype saliency)
- **`notes_on_paper.tex`** — Revision notes, reviewer Q&A, submission checklist, fallback git hashes
- **`ml_chemistry_template.tex`** — Original DL-focused paper (preserved, Overleaf-synced)
- **`tune_rf.py`** — RF hyperparameter grid search (432 configs)
- **`tune_bigru.py`** — BiGRU Optuna HPO (25/30 trials done, SQLite storage, stop/resume safe)
- **`analyze_bigru_saliency.py`** — BiGRU gradient saliency → atom-level 2D heatmaps (InputxGradient)
- **`analyze_rf_interpretability.py`** — RF feature importance analysis (Morgan FP bit → substructure)
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
