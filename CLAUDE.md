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

#### Phase A0 — Fix Claims, Citations, Factual Errors (IN PROGRESS)

**A0.1 Factual errors** (MUST FIX):
- `benchmark_paper.tex:297`: Says "patience 10, max 50 epochs" → actual is patience=15, max=100
- `benchmark_paper.tex` preamble: Missing `\usepackage{multirow}` → COMPILE ERROR
- `benchmark_paper.tex:412`: TikZ BiGRU box says "(best on novel cpds)" → ChemBERTa is actually best on wetlab

**A0.2 Citation fixes**:
- `Proposal.bib`: `chenMolecularLanguage2023` has WRONG JOURNAL — says "JCIM" but actual is "Briefings in Functional Genomics" (vol 22, issue 4, pp 392-400, DOI 10.1093/bfgp/elad012)
- Chen et al. paper is about molecular GENERATION, not property prediction — add scope caveat at 6 citation locations
- `benchmark_paper.tex:148`: "Joung et al. applied a GCNN" → misleading for a dataset paper, rephrase to "reported a GCNN baseline"
- `Proposal.bib`: 30 duplicate citation keys (281 entries, 244 unique) — low priority

**A0.3 Soften overclaimed local-feature narrative** (6 locations):
- Contributions list (line 98): "Empirical evidence" → "Evidence consistent with"
- Related Work (line 140): Add "generation tasks" qualifier for Chen et al.
- TikZ caption (line 418): "Results confirm" → "Cross-validated results show"
- Discussion (line 422): "are better suited" → "achieve better in-distribution performance"
- Gap discussion (line 425): "demonstrates" → "is consistent with the hypothesis"
- Conclusion (line 757): "corroborate" → "are consistent with"
- Add confounders/tuning-asymmetry paragraph to Limitations section (~line 748)

**A0.4 Leave space for BiGRU tuning updates**:
- Table 2, Table 5, TikZ diagram, discussion, contributions, abstract may need updating after Optuna HPO
- If tuned BiGRU beats ChemBERTa on wetlab → strengthen local-feature claim
- If tuned BiGRU ≈ ChemBERTa → maintain "complementary" narrative

#### Phase A — Critical Fixes (MUST do before submission)

**A1. LaTeX compilation fixes** — MERGED INTO A0.1 above

**A2. Supplementary completeness** (~1 hour)
- Add ChemBERTa per-fold supplementary table (analogous to existing RF and BiGRU tables)
- Add XGBoost per-fold supplementary table
- Update significance table (Table S3) to include ChemBERTa vs RF, ChemBERTa vs BiGRU comparisons
- Add ChemBERTa learning curves figure to supplementary

**A3. BiGRU hyperparameter tuning — #1 reviewer concern** (~2 days GPU)
- RF had 432-config grid search; BiGRU uses default architecture from Liu et al. — this WILL be flagged
- Optuna TPE, 30 trials on fold 0 val set
- Search space: units {64,128,256}, layers {1,2,3}, embed {32,50,100}, dropout {0.1-0.4}, lr [1e-4,3e-3], batch {32,64,128}
- Script: `tune_bigru.py` (analogous to `tune_rf.py`)
- Run best config on all 5 folds → compare tuned vs default BiGRU
- Paper: update Table 2, add row to tuning asymmetry table (S5), discuss in text
- Expected outcome: modest improvement (2-5%), confirming RF advantage is real
- 5 new BibTeX: Chung 2014, Goh 2017, Grisoni 2020, Bergstra 2012, Akiba 2019

**A4. Multi-model parity/error figure** (~2 hours)
- Current error analysis (Figs 3-4) shows only BiGRU — reviewer will want all models
- Create 2×2 parity plot (RF, XGBoost, BiGRU, ChemBERTa) from pooled CV predictions
- Shows RF's tighter clustering vs BiGRU's better tail behavior vs ChemBERTa's scatter

**A5. Code + data repository** (~2 hours)
- Clean GitHub repo with README, requirements.txt, run instructions
- Zenodo DOI for reproducibility archive (model weights, processed datasets)
- Update Data Availability section with actual URLs

**A6. Full proofread** (~2 hours)
- Check all numbers match result files
- Consistent decimal places, units, terminology
- Grammar pass, especially around new ChemBERTa text
- Verify figure captions are accurate and self-contained
- Check references render correctly (especially BibTeX edge cases)

#### Phase B — High-Impact Additions (strongly recommended)

**B1. RNN architecture comparison** (~3 days GPU)
- BiGRU vs BiLSTM vs CNN-BiGRU, all 5-fold CV v2 (already implemented: `bilstm_v2`, `cnn_bigru_v2`)
- Answers "why BiGRU specifically?" — common reviewer question
- Paper: new table or expand Table 2 with architecture variants
- Expected: BiGRU ≈ BiLSTM, CNN-BiGRU possibly slightly better (adds conv features)

**B2. Scaffold split on primary dataset** (~1 day GPU + analysis)
- Currently only random stratified split; scaffold split is the gold standard for generalization
- Run RF + BiGRU + ChemBERTa on Murcko scaffold split (80/10/10)
- Strengthens the interpolation/extrapolation narrative with in-distribution evidence
- J. Cheminformatics reviewers strongly prefer scaffold splits
- Paper: add Table or discussion paragraph in Model Comparison section

**B3. Wetlab per-molecule comparison figure** (~1 hour)
- Grouped bar chart: experimental vs RF vs BiGRU vs ChemBERTa for each of 16 molecules
- Currently only a text table (Table 5 shows aggregates); a figure is much more informative
- Highlights where models agree/disagree and which molecules are hardest

**B4. ChemBERTa convergence analysis** (~1 hour)
- Plot training/val loss curves for all 5 folds (data in history JSON files)
- Discuss whether MAX_LEN=256 truncation matters (check % of SMILES truncated)
- Note that ChemBERTa had no separate HP tuning either (same as BiGRU baseline treatment)

#### Phase C — Differentiating Additions (for top-tier journal)

**C1. RF + BiGRU ensemble** (~4 hours)
- Simple average or learned weighting of RF and BiGRU predictions
- The complementary strengths narrative predicts this should work well
- Evaluate on CV, cross-dataset, and wetlab
- Could become the recommended practical approach in the conclusion

**C2. Applicability domain analysis** (~4 hours)
- Compute Tanimoto similarity of each test molecule to its nearest training neighbor
- Plot MAE vs distance-to-training-set for RF and BiGRU
- Should show RF degrades faster with distance (fingerprint-based) while BiGRU degrades gracefully
- Quantifies the interpolation/extrapolation narrative with concrete evidence
- Very common in J. Cheminformatics papers

**C3. SHAP values for RF** (~2 hours)
- TreeSHAP is much faster and more principled than Gini importance
- Provides per-prediction explanations, not just global importance
- Could show waterfall plots for wetlab molecules explaining why predictions differ

**C4. Journal-specific formatting** (~2 hours)
- J. Cheminformatics: use their LaTeX template (jcheminf class)
- Or Digital Discovery: RSC template
- Convert supplementary to separate file if required
- Ensure open-access compliance (CC-BY license)

#### Phase D — Multi-Property Expansion (NEW: `benchmark_paper_v2.tex`)

**Rationale**: Strengthen local-feature claim by showing same RF > Transformer pattern on multiple properties. Create new paper file to keep current version intact.

**D1. Fluorescence emission wavelength (λ_em)** — LOCAL property
- Dataset: ChemFluor (`data/raw/chemfluor.xlsx`) — 4,386 rows, already in project
- Target: `Emission/nm` (296–1045 nm, mean 537 ± 91)
- Solvents: 63 unique names → map to SMILES
- Models: RF + BiGRU + ChemBERTa, 5-fold CV (~26h GPU)
- Expected: RF > BiGRU > ChemBERTa (same pattern as UV)

**D2. Molar extinction coefficient (log₁₀ MEC)** — LOCAL property
- Dataset: Mamede (`data/mamede_regression_dataset.csv`) — 84,262 rows (after filtering)
- Target: `log10(mec)` (0–5.7, mean 4.14 ± 0.64)
- Models: RF only (DL on 84K too expensive for current scope)
- Expected: RF dominates

**D3. Lipophilicity / LogP (CONTROL)** — INTERMEDIATE/GLOBAL property
- Dataset: MoleculeNet Lipophilicity — ~4,200 rows (download from DeepChem S3)
- Target: `exp` (LogP), no solvent
- Models: RF + BiGRU + ChemBERTa, 5-fold CV (~14h GPU)
- Expected: Pattern equalizes or reverses (RF advantage disappears)

**Implementation:**
- New script: `run_multi_property.py` (parameterized by property + model)
- Preprocessed data: `data/chemfluor_emission_processed.csv`, `data/mamede_log_mec_processed.csv`, `data/lipophilicity_processed.csv`
- Results: `results/emission/`, `results/log_mec/`, `results/lipophilicity/`
- Paper: New section "Multi-Property Generalization" (~100-120 lines) + table + figure
- Total GPU: ~32h (recommended scope)

#### Dropped from Plan
- Transfer learning (high risk, uncertain payoff)
- nablaColors / UniProp / 3D comparison (removed from paper)
- ChemBERTa attention visualization (interesting but not critical)

#### Priority Execution Order
```
Phase A (DONE except A3+A5): A0 ✅ → A2 ✅ → A4 ✅ → A6 ✅ → A3 🔄 → A5
Phase B (DONE: B3 ✅, B4 ✅): B2 optional
Phase C (optional): C1 → C2 → C3 → C4
Phase D (after Phase A complete): D1 → D2 → D3 → paper integration
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
