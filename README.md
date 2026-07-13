# ML_UV_models

**When Do Simple Models Win? Machine Learning Architectures for UV Absorption Prediction**

Companion code for the J. Chem. Inf. Model. manuscript (ci-2026-009433). A controlled comparison of five ML model families — Random Forest, XGBoost, Chemprop D-MPNN, BiGRU, ChemBERTa — for predicting UV–Vis absorption wavelength ($\lambda_{\max}$) across 18,415 solute–solvent pairs (Joung+Beard, Greenman–Song deduplicated).

- **Manuscript**: JCIM submission, ci-2026-009433 (under revision)
- **Pretrained checkpoints + cleaned dataset**: Zenodo DOI [10.5281/zenodo.20600225](https://doi.org/10.5281/zenodo.20600225)
- **License**: MIT

---

## TL;DR for reviewers

**Reproduce the headline RMSE numbers without retraining anything:**

```bash
pip install -e ".[dev]"
# Download the Zenodo bundle, unzip, then:
python -c "
import numpy as np
from sklearn.metrics import mean_squared_error
for model in ['rf_tuned', 'chemprop', 'bigru_tuned', 'chemberta', 'xgboost']:
    rmses = []
    for f in range(5):
        y    = np.load(f'predictions/{model}_v3_fold{f}_y_test.npy')
        pred = np.load(f'predictions/{model}_v3_fold{f}_predictions.npy')
        rmses.append(np.sqrt(mean_squared_error(y, pred)))
    print(f'{model:20s} {np.mean(rmses):6.2f} ± {np.std(rmses, ddof=1):.2f} nm')
"
```

Expected output:

```
rf_tuned                31.50 ± 1.47 nm
chemprop                33.15 ± 3.27 nm
bigru_tuned             36.20 ± 1.20 nm
chemberta               54.09 ± 3.57 nm
xgboost                 33.92 ± 1.26 nm
```

These are the values in Table 2 of the manuscript.

---

## Repository layout

```
ML_UV_models/
├── README.md                     this file
├── pyproject.toml                Python package definition
├── requirements.txt              pinned dependencies
├── LICENSE                       MIT license
├── jcim_secondrevision_submission/  LaTeX sources + compiled PDFs for the current
│                                 (second) revision. Build with `latexmk -pdf <file>.tex`.
├── jcim_firstrevision_submission/   first-revision submission, preserved for reference
├── legacy/                       earlier drafts, LaTeX templates, and standalone assets
├── paper1_new_cl/                Python package (shared utilities, models, splits)
├── data/                         processed datasets (excluding raw)
├── previous_code/                original v2 dataset + earlier work
├── results/                      training outputs (per-fold metrics, predictions,
│                                 configs, training logs). Big binary artefacts
│                                 (.ckpt, .keras, .joblib, .npy) are gitignored;
│                                 fetch from the Zenodo bundle.
├── checkpoints/                  small checkpoint metadata
├── wetlab_uv/                    16-molecule wetlab validation set
├── tests/                        sanity tests
└── scripts/                      runnable entry points (see below)
```

### `scripts/` — where to find each script

All scripts are run **from the project root**, e.g. `python scripts/training/run_baselines.py`. Each script supports `--help` for its CLI.

| Folder | What's in it | When to use it |
|---|---|---|
| `scripts/training/` | Train each model family from scratch | Retrain a model, reproduce per-fold checkpoints |
| `scripts/tuning/` | Optuna / grid hyperparameter searches | Reproduce or extend the HPO study (Chemprop, BiGRU) |
| `scripts/evaluation/` | Evaluate without training | Recompute metrics on wetlab, external, or classification splits |
| `scripts/analysis/` | One-off analyses for the revision | Reproduce the chromophore-leakage audit, non-local subset breakdown, ChemBERTa ceiling diagnostic, etc. |
| `scripts/data_prep/` | Download and clean source datasets | Rebuild any of the processed CSVs in `data/` from raw sources |
| `scripts/figures/` | Generate paper figures | Regenerate parity plots, TOC graphic, dataset-overview panels |
| `scripts/wetlab_aux/` | Auxiliary hybrid/transfer experiments | Older exploratory branches; not part of the JCIM manuscript headline results |

---

## Setup

```bash
git clone https://github.com/Umesh1608/ML_UV_models.git
cd ML_UV_models
pip install -e ".[dev]"
```

**Compute requirements**:

| Task | Hardware | Wall-clock |
|---|---|---|
| Reproduce metrics from saved predictions (Zenodo bundle) | CPU laptop | seconds |
| Train RF / XGBoost from scratch (5 folds) | CPU | ~20 min |
| Train Chemprop / BiGRU (5 folds) | NVIDIA GPU, 24 GB | ~2.5 h each |
| Train ChemBERTa (5 folds) | NVIDIA GPU, 24 GB | ~15 h |
| Optuna HPO for Chemprop or BiGRU | NVIDIA GPU | ~6 h |

Tested on RTX 4090 (local) and A40 (RunPod). CUDA 12.7, PyTorch 2.x, TensorFlow 2.20, Chemprop 2.2.x.

---

## Reproducing the manuscript

### 1. Primary 5-fold benchmark (Table 2 of the manuscript)

```bash
# Random Forest (tuned hyperparameters from a 432-config grid search)
python scripts/training/run_baselines.py --model rf_tuned --v3

# XGBoost
python scripts/training/run_baselines.py --model xgboost --v3

# BiGRU (tuned: 3-layer 256-unit; configuration from Optuna trial 25)
python scripts/training/run_baselines.py --model bigru_tuned --v3

# Chemprop D-MPNN (default hyperparameters from the v2 release)
for f in 0 1 2 3 4; do python scripts/training/run_chemprop.py --fold $f --v3; done

# ChemBERTa fine-tuning (PyTorch; one fold at a time)
for f in 0 1 2 3 4; do python scripts/training/run_chemberta.py --fold $f --v3; done

# Aggregate everything into a results table
python scripts/training/run_baselines.py --summary --v3
```

### 2. Chemprop hyperparameter probe

```bash
python scripts/tuning/tune_chemprop.py --n-trials 25
# then retrain Chemprop with the best config on all five folds:
python scripts/training/run_chemprop.py --tuned --fold 0   # repeat for folds 1-4
```

The tuned Chemprop achieves RMSE 28.83 ± 1.32 nm versus the default 33.15 ± 3.27 nm, primarily by eliminating the fold-4 early-stopping outlier (39.21 → 28.46 nm).

### 3. Cross-dataset benchmarks

```bash
python scripts/training/run_cross_dataset.py --dataset deep4chem
python scripts/training/run_cross_dataset.py --dataset jung2024
python scripts/training/run_cross_dataset.py --summary
```

### 4. Wetlab validation (16 commercial UV-absorbers)

```bash
python scripts/evaluation/eval_wetlab.py
```

### 5. Photosafety classification (Mamede protocol)

```bash
python scripts/training/run_mamede.py
python scripts/evaluation/eval_classification.py
```

---

## Revision-specific analyses

Every analysis surfaced during peer review can be reproduced with one command.

| Concern | Script | Output |
|---|---|---|
| Non-local subset analysis (D-A, D-A-D, long-wavelength, BODIPY) | `scripts/analysis/analyze_nonlocal_subsets.py` | `results/nonlocal_subsets_v3/` |
| Deep4Chem overlap with Joung+Beard | `scripts/analysis/analyze_deep4chem_overlap.py` | `results/r1_7_*` |
| Greenman–Song duplicate-handling audit | `scripts/analysis/analyze_duplicate_handling.py` | `results/r1_8_*` |
| Chromophore-overlap leakage in CV splits | `scripts/analysis/analyze_chromophore_leakage.py` | `results/r1_9_*` |
| Wetlab Tanimoto + scaffold audit | `scripts/analysis/analyze_wetlab_similarity.py` | `results/r1_13_*` |
| Solvent-encoding gain on multi-solvent subset | `scripts/analysis/analyze_multisolvent_subset.py` | `results/r1_14_*` |
| ChemBERTa 600 nm prediction ceiling | `scripts/analysis/analyze_chemberta_ceiling.py` | `results/r1_16_*` |
| Experimental $\lambda_{\max}$ uncertainty | `scripts/analysis/analyze_experimental_uncertainty.py` | `results/r2_1_*` |
| Per-solute solvatochromism span | `scripts/analysis/analyze_solvent_shifts.py` | `results/r2_4_*` |
| Polarity-grouped prediction variability | `scripts/analysis/analyze_solvent_polarity.py` | `results/r2_5_*` |
| RF feature importance + Morgan-bit interpretability | `scripts/analysis/analyze_rf_interpretability.py` | `results/rf_*` |
| BiGRU gradient saliency | `scripts/analysis/analyze_bigru_saliency.py` | `results/bigru_*saliency*` |

---

## Trained model checkpoints

Per-fold checkpoints for all five model families (4.7 GB RF joblibs + 50 MB Chemprop ckpts + 115 MB BiGRU keras + ChemBERTa pt + 7 MB XGBoost joblibs) are deposited at:

**Zenodo DOI: [10.5281/zenodo.20600225](https://doi.org/10.5281/zenodo.20600225)**

The Zenodo bundle also includes the cleaned v3 dataset, the per-fold CV index file, the per-fold predictions, and the per-fold metrics, so you can reproduce the manuscript's Table 2 numbers (and Table S9 v2/v3 comparison, and the chromophore-leakage decomposition) without training anything.

---

## Datasets

| Dataset | Records | Location |
|---|---|---|
| Joung+Beard primary (v3 cleaned, Greenman–Song dedup) | 18,415 | `previous_code/UV_canonical_v3_dedup.csv` |
| Deep4Chem external | 17,683 | `data/deep4chem_processed.csv` |
| Jung et al. 2024 external | ~26,000 | `data/jung2024_processed.csv` |
| Mamede photosafety (classification) | 74,783 | reconstructed via `scripts/data_prep/postprocess_reaxys.py` from Mamede's published SI |
| Wetlab measurements (this work) | 16 molecules × 2 solvents | `wetlab_uv/` |

Raw downloads are scripted in `scripts/data_prep/download_datasets.py`.

---

## Citation

If you use this code or the deposited checkpoints, please cite both the manuscript and the Zenodo record:

```bibtex
@article{arampath2026uv,
  title   = {When Do Simple Models Win? Machine Learning Architectures for UV Absorption Prediction},
  author  = {Arampath, Umesh and Pero, Bryant and Stewart, David and Demirjian, David},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026},
  note    = {Under revision (ci-2026-009433)}
}

@dataset{arampath2026zenodo,
  title     = {ML\_UV\_models: v3 artifact bundle for ``When Do Simple Models Win?''},
  author    = {Arampath, Umesh and Pero, Bryant and Stewart, David and Demirjian, David},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20600225},
  year      = {2026}
}
```

---

## License

MIT.

## Contact

- Corresponding author: Umesh Arampath, `uarampath@mwbioprocessing.com`
- Issues / questions: please open an issue on this repository.
