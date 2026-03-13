#!/bin/bash
# Phase 2 GPU jobs — sequential (single GPU)
set -e

echo "=== Phase 2 GPU Track ==="
echo "Started: $(date)"

# Emission Chemprop remaining folds (folds 0-2 done)
for fold in 3 4; do
    echo "[GPU] emission Chemprop fold $fold..."
    python3 run_multi_property.py --property emission --model chemprop --fold $fold
done

# Lipophilicity Chemprop all folds
for fold in 0 1 2 3 4; do
    echo "[GPU] lipophilicity Chemprop fold $fold..."
    python3 run_multi_property.py --property lipophilicity --model chemprop --fold $fold
done

# Solubility Chemprop all folds
for fold in 0 1 2 3 4; do
    echo "[GPU] solubility Chemprop fold $fold..."
    python3 run_multi_property.py --property solubility --model chemprop --fold $fold
done

# log_mec XGBoost on GPU (81K rows — GPU avoids CPU RAM pressure)
for fold in 0 1 2 3 4; do
    echo "[GPU] log_mec XGBoost fold $fold..."
    python3 run_multi_property.py --property log_mec --model xgboost --fold $fold
done

# Emission BiGRU all folds
for fold in 0 1 2 3 4; do
    echo "[GPU] emission BiGRU fold $fold..."
    python3 run_multi_property.py --property emission --model bigru --fold $fold
done

# Lipophilicity BiGRU all folds
for fold in 0 1 2 3 4; do
    echo "[GPU] lipophilicity BiGRU fold $fold..."
    python3 run_multi_property.py --property lipophilicity --model bigru --fold $fold
done

# Solubility BiGRU all folds
for fold in 0 1 2 3 4; do
    echo "[GPU] solubility BiGRU fold $fold..."
    python3 run_multi_property.py --property solubility --model bigru --fold $fold
done

# Emission ChemBERTa all folds
for fold in 0 1 2 3 4; do
    echo "[GPU] emission ChemBERTa fold $fold..."
    python3 run_multi_property.py --property emission --model chemberta --fold $fold
done

# Lipophilicity ChemBERTa all folds
for fold in 0 1 2 3 4; do
    echo "[GPU] lipophilicity ChemBERTa fold $fold..."
    python3 run_multi_property.py --property lipophilicity --model chemberta --fold $fold
done

# Solubility ChemBERTa all folds
for fold in 0 1 2 3 4; do
    echo "[GPU] solubility ChemBERTa fold $fold..."
    python3 run_multi_property.py --property solubility --model chemberta --fold $fold
done

echo "=== Phase 2 GPU Track DONE ==="
echo "Finished: $(date)"
