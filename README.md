# When Do Simple Models Win? Benchmarking Machine Learning for UV Absorption Prediction

Systematic benchmark of ML approaches (RF, XGBoost, BiGRU, ChemBERTa) for predicting UV-Vis absorption maxima (lambda_max) from molecular structure.

## Key Findings

- **RF + Morgan fingerprints** achieves the best cross-validated performance (RMSE 31.34 nm)
- **Deep learning** (ChemBERTa, BiGRU) generalizes better to novel out-of-distribution molecules
- **Solvent concatenation** improves both model families by 7-8% RMSE
- Model selection depends on task: RF for interpolation/screening, DL for novel compound exploration

## Setup

```bash
pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

**GPU**: NVIDIA GPU with CUDA 12+ required for deep learning models. Tested on RTX 4090.

## Reproducing Results

### Primary benchmark (Joung+Beard, 18,755 compounds)

```bash
# Random Forest (tuned, ~3 min/fold)
python run_baselines.py --model rf_tuned --v2

# BiGRU with solvent (~5 hr/fold on GPU)
python run_baselines.py --model bigru_solvent --v2

# BiGRU hyperparameter optimization (Optuna, ~50 hr total)
python tune_bigru.py --n-trials 25
python tune_bigru.py --full-cv

# ChemBERTa (~3 hr/fold on GPU)
python run_chemberta.py

# XGBoost
python run_baselines.py --model xgboost --v2

# Aggregate results
python run_baselines.py --summary --v2
```

### Cross-dataset evaluation

```bash
python run_cross_dataset.py --dataset deep4chem --split random
python run_cross_dataset.py --dataset jung2024 --split random
python run_cross_dataset.py --dataset nablacolors --split precomputed
```

### Wetlab experimental validation

```bash
python eval_wetlab.py
```

### Classification (Mamede photosafety)

```bash
# RF classifier (trained on ~74K Reaxys compounds)
python eval_classification.py

# BiGRU direct classification (tuned 3L/256u architecture)
python train_bigru_classification.py
```

### Interpretability analysis

```bash
python analyze_rf_interpretability.py
python analyze_bigru_saliency.py
```

## Project Structure

```
paper1_new_cl/          Python package (models, splits, metrics)
run_baselines.py        Main 5-fold CV benchmark (RF, XGBoost, BiGRU)
run_chemberta.py        ChemBERTa fine-tuning (PyTorch/HuggingFace)
tune_rf.py              RF hyperparameter grid search (432 configs)
tune_bigru.py           BiGRU Bayesian HPO (Optuna TPE, stop/resume safe)
run_cross_dataset.py    Cross-dataset benchmarks
eval_wetlab.py          Wetlab experimental validation
eval_classification.py  Mamede photosafety classification
analyze_rf_interpretability.py     RF feature importance analysis
analyze_bigru_saliency.py          BiGRU gradient saliency analysis
data/                   Processed datasets
results/                Model outputs, figures, metrics
benchmark_paper.tex     Manuscript (LaTeX)
```

## Datasets

| Dataset | Samples | Source |
|---------|---------|--------|
| Joung+Beard (primary) | 18,755 | [Figshare](https://doi.org/10.6084/m9.figshare.12045567) |
| Deep4Chem | ~20,000 | Joung 2020 |
| Jung 2024 | ~26,000 | GitHub |
| nablaColors | 24,567 | Zenodo |

## Citation

Paper under review at J. Chem. Inf. Model. (JCIM).

## License

MIT
