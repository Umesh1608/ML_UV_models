# When Do Simple Models Win? Machine Learning Architectures for UV Absorption Prediction and Insights for General Molecular Property Prediction

A controlled comparison of five models spanning four architecture families — fingerprint-based ensembles (Random Forest, XGBoost), a directed message-passing graph neural network (Chemprop D-MPNN), a bidirectional gated recurrent network (BiGRU), and a pretrained Transformer (ChemBERTa) — for predicting UV-Vis absorption wavelength (lambda_max) across 18,755 solute-solvent pairs. The answer depends on the task: RF is optimal for screening within known chemical space, while deep learning is superior for exploring novel scaffolds. Encoding solvent identity improves all models by 6-8% RMSE without domain-specific descriptor engineering.

## Key Findings

- **RF + Morgan fingerprints** achieves the best cross-validated RMSE (31.34 nm) on 18,755 chromophore-solvent pairs
- **Chemprop D-MPNN** statistically ties RF (31.69 nm, p=0.77) with a learned molecular graph representation
- **Deep learning generalizes better**: on 16 novel wetlab molecules, ChemBERTa (MAE 26.3 nm) and BiGRU (28.6 nm) outperform RF (38.5 nm)
- **Solvent concatenation** improves all model families by 7-8% RMSE
- **Model choice depends on task**: RF/Chemprop for interpolation and virtual screening; DL for novel compound exploration

## Setup

```bash
pip install -e ".[dev]"
```

**GPU**: NVIDIA GPU with CUDA 12+ required for deep learning models. Tested on RTX 4090.

## Reproducing Results

### Primary benchmark (Joung+Beard, 18,755 compounds)

```bash
# Random Forest (tuned, ~3 min/fold)
python run_baselines.py --model rf_tuned --v2

# XGBoost
python run_baselines.py --model xgboost --v2

# Chemprop D-MPNN (~5 min/fold on GPU)
python run_chemprop.py --fold 0  # repeat for folds 1-4
python run_chemprop.py --summary

# BiGRU with solvent (~3-5 hr/fold on GPU)
python run_baselines.py --model bigru_solvent --v2

# BiGRU hyperparameter optimization (Optuna, ~50 hr total)
python tune_bigru.py --n-trials 25
python tune_bigru.py --full-cv

# ChemBERTa (~3 hr/fold on GPU)
python run_chemberta.py

# Aggregate results
python run_baselines.py --summary --v2
```

### Cross-dataset evaluation

```bash
python run_cross_dataset.py --dataset deep4chem --split random
python run_cross_dataset.py --dataset jung2024 --split random
python run_cross_dataset.py --dataset jung2024 --split scaffold
python run_cross_dataset.py --dataset nablacolors --split precomputed
```

### Wetlab experimental validation

```bash
python eval_wetlab.py  # 16 molecules x 2 solvents, RF + BiGRU + ChemBERTa
```

### Classification (Mamede photosafety)

```bash
python eval_classification.py
python train_bigru_classification.py
```

### Multi-property benchmark (Phase II)

Extends the benchmark to additional molecular properties to test the locality-of-property hypothesis.

```bash
# Single fold
python run_multi_property.py --property emission --model rf --fold 0

# All folds for a model/property
python run_multi_property.py --property lipophilicity --model chemprop

# Available properties: emission, lipophilicity, solubility, log_mec
# Available models: rf, xgboost, chemprop, bigru, chemberta

# Aggregate results
python run_multi_property.py --summary
```

### Interpretability analysis

```bash
python analyze_rf_interpretability.py   # Morgan FP bit -> substructure mapping
python analyze_bigru_saliency.py        # Gradient saliency -> atom-level heatmaps
```

## Results Summary

### Cross-validated performance (5-fold, 64/16/20 train/val/test split)

| Model | RMSE (nm) | MAE (nm) | R² |
|-------|-----------|----------|-----|
| RF (tuned) | 31.34 ± 1.82 | 15.16 ± 0.44 | 0.914 |
| Chemprop D-MPNN | 31.69 ± 3.18 | 16.84 ± 1.74 | 0.911 |
| XGBoost | 33.70 ± 1.61 | 20.05 ± 0.48 | 0.900 |
| BiGRU (tuned) | 34.71 ± 1.40 | 18.09 ± 0.59 | 0.894 |
| ChemBERTa | 50.39 ± 2.30 | 24.31 ± 1.04 | 0.777 |

### Wetlab validation (16 novel molecules x 2 solvents)

| Model | MAE (nm) | RMSE (nm) |
|-------|----------|-----------|
| ChemBERTa | **26.3** | 32.2 |
| BiGRU | 28.6 | 33.9 |
| RF | 38.5 | 43.7 |

## Project Structure

```
paper1_new_cl/                  Python package (models, splits, metrics)
  models.py                     Model architectures, training, evaluation
  splits.py                     Scaffold splits, stratified folds

run_baselines.py                Main 5-fold CV benchmark (RF, XGBoost, BiGRU variants)
run_chemprop.py                 Chemprop D-MPNN training (PyTorch Lightning)
run_chemberta.py                ChemBERTa fine-tuning (PyTorch/HuggingFace)
run_multi_property.py           Multi-property benchmark (emission, solubility, etc.)
run_cross_dataset.py            Cross-dataset transfer benchmarks
eval_wetlab.py                  Wetlab experimental validation
eval_classification.py          Mamede photosafety classification
tune_rf.py                      RF hyperparameter grid search (432 configs)
tune_bigru.py                   BiGRU Bayesian HPO (Optuna, stop/resume safe)
analyze_rf_interpretability.py  RF feature importance (Morgan FP bit -> substructure)
analyze_bigru_saliency.py       BiGRU gradient saliency (atom-level heatmaps)
download_datasets.py            Download and preprocess external datasets

data/                           Processed datasets
results/                        Model outputs, metrics JSONs, figures
benchmark_paper_jcim.tex        JCIM submission manuscript (achemso)
benchmark_paper.tex             Generic article version (Overleaf-synced)
supporting_information.tex      Supporting Information (per-fold tables, etc.)
```

## Datasets

| Dataset | Samples | Properties | Source |
|---------|---------|------------|--------|
| Joung+Beard (primary) | 18,755 | λ_max, solvent | [Figshare](https://doi.org/10.6084/m9.figshare.12045567) |
| Deep4Chem | ~20,000 | λ_max, λ_em, log(ε), QY, lifetime | Joung 2020, Figshare |
| Jung 2024 | ~26,000 | λ_max, solvent, scaffold | GitHub |
| nablaColors | 24,567 | λ_max (3D-optimized) | Zenodo |
| ChemFluor | 4,232 | Emission λ_em | Extracted from Deep4Chem |
| Mamede log₁₀(MEC) | 81,638 | Molar extinction coefficient | Mamede 2021 |
| Lipophilicity | 4,200 | LogP | MoleculeNet |
| AqSolDB | — | Aqueous solubility (logS) | AqSolDB |

## Citation

Paper under review at J. Chem. Inf. Model. (JCIM).

## License

MIT
