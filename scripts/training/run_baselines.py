#!/usr/bin/env python3
"""
Run baseline experiments for UV absorption (lambda_max) prediction with 5-fold CV.

Models:
  1. BiGRU (with solvent)        — primary model
  2. BiGRU (without solvent)     — ablation
  3. BiLSTM (with solvent)       — architecture comparison
  4. CNN-BiGRU (with solvent)    — architecture comparison
  5. Random Forest + Morgan FP   — classical ML baseline
  6. XGBoost + Morgan FP         — classical ML baseline

All models use 5-fold cross-validation (KFold, shuffle=True, seed=7).
Metrics: RMSE, MAE, R², Pearson r (reported as mean +/- std across folds)

Usage:
  python3 run_baselines.py                                    # all 6 models, 5-fold CV
  python3 run_baselines.py --model bigru_solvent              # one model, 5-fold CV
  python3 run_baselines.py --model bigru_solvent --fold 0     # one model, one fold
  python3 run_baselines.py --summary                          # regenerate table + plots
"""

import argparse
import gc
import os
import sys
import time

# ─── GPU / CUDA setup ────────────────────────────────────────────────────────
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/home/umesh/.local/cuda_compat"
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from paper1_new_cl.models import (
    compute_metrics,
    build_char_maps,
    vectorize_smiles,
    create_dl_model,
    train_dl_model,
    get_tqdm_callback,
    verify_gpu_available,
    compute_morgan_fps,
)
from paper1_new_cl.splits import create_stratified_folds

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "previous_code", "UV_canonical_full_dataset.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 7
N_FOLDS = 5
EPOCHS = 250
BATCH_SIZE = 32     # publication-quality (was 80 for faster iteration)
PATIENCE = 25

# Model key → (display_name, model_type, with_solvent)
MODEL_REGISTRY = {
    "bigru_solvent":    ("BiGRU + Solvent",     "bigru",    True),
    "bigru_nosolvent":  ("BiGRU (no solvent)",  "bigru",    False),
    "bilstm":           ("BiLSTM + Solvent",    "bilstm",   True),
    "cnn_bigru":        ("CNN-BiGRU + Solvent", "cnn_bigru", True),
    "rf":               ("RF + Morgan FP (MSE)",      "rf",       None),
    "rf_mae":           ("RF + Morgan FP (MAE)",      "rf_mae",   None),
    "xgboost":          ("XGBoost + Morgan FP (MSE)", "xgb",      None),
    "xgboost_mae":      ("XGBoost + Morgan FP (MAE)", "xgb_mae",  None),
    "rf_1024":          ("RF + Morgan FP 1024 (MSE)", "rf_1024",  None),
    "rf_nosolvent":     ("RF + Morgan FP (solute only)", "rf_nosolvent", None),
}
MODEL_ORDER = ["bigru_solvent", "bigru_nosolvent", "bilstm", "cnn_bigru", "rf", "rf_mae", "rf_1024", "rf_nosolvent", "xgboost", "xgboost_mae"]

# v2 models: same architectures, solvent-stratified 3-way split (train/val/test)
MODEL_REGISTRY.update({
    "bigru_solvent_v2":   ("BiGRU + Solvent (v2)",     "bigru",    True),
    "bigru_nosolvent_v2": ("BiGRU no solvent (v2)",    "bigru",    False),
    "bilstm_v2":          ("BiLSTM + Solvent (v2)",    "bilstm",   True),
    "cnn_bigru_v2":       ("CNN-BiGRU + Solvent (v2)", "cnn_bigru", True),
    "rf_v2":              ("RF + Morgan FP MSE (v2)",       "rf",       None),
    "rf_mae_v2":          ("RF + Morgan FP MAE (v2)",       "rf_mae",   None),
    "xgboost_v2":         ("XGBoost + Morgan FP MSE (v2)", "xgb",      None),
    "xgboost_mae_v2":     ("XGBoost + Morgan FP MAE (v2)", "xgb_mae",  None),
    "rf_1024_v2":         ("RF + Morgan FP 1024 MSE (v2)", "rf_1024",  None),
    "rf_nosolvent_v2":    ("RF + Morgan FP solute only (v2)", "rf_nosolvent", None),
})
MODEL_ORDER_V2 = [k + "_v2" for k in MODEL_ORDER]


# ─── Data loading (global — no splitting) ─────────────────────────────────────

def load_and_prepare_global_data():
    """Load dataset and compute charset/embed_len from ALL data. No splitting."""
    print("\n[DATA] Loading full dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df[["canon", "solvents", "lambda_max"]].dropna()
    df["combined"] = df["canon"] + "!" + df["solvents"]

    all_smiles_combined = df["combined"].values
    charset_combined = sorted(set("".join(all_smiles_combined) + "!E"))
    embed_combined = max(len(s) for s in all_smiles_combined) + 5

    all_smiles_solute = df["canon"].values
    charset_solute = sorted(set("".join(all_smiles_solute) + "!E"))
    embed_solute = max(len(s) for s in all_smiles_solute) + 5

    print(f"  Dataset: {len(df)} samples")
    print(f"  Combined charset: {len(charset_combined)} tokens, max seq len: {embed_combined}")
    print(f"  Solute-only charset: {len(charset_solute)} tokens, max seq len: {embed_solute}")

    return {
        "smiles_combined": df["combined"].values,
        "smiles_solute": df["canon"].values,
        "y": df["lambda_max"].values,
        "solvents": df["solvents"].values,
        "charset_combined": charset_combined,
        "charset_solute": charset_solute,
        "embed_combined": embed_combined,
        "embed_solute": embed_solute,
    }


# ─── Fold splitting ──────────────────────────────────────────────────────────

def create_fold_splits(n_samples, n_folds=N_FOLDS, seed=SEED):
    """Create KFold splits. Returns list of (train_idx, test_idx) and saves to disk."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(np.arange(n_samples)))

    # Save fold indices for reproducibility
    save_dict = {}
    for i, (train_idx, test_idx) in enumerate(folds):
        save_dict[f"train_{i}"] = train_idx
        save_dict[f"test_{i}"] = test_idx
    np.savez(os.path.join(RESULTS_DIR, "cv_fold_indices.npz"), **save_dict)
    print(f"  Fold splits saved ({n_folds} folds, {n_samples} samples)")

    return folds


def prepare_fold_data(global_data, train_idx, test_idx):
    """Slice global arrays by fold indices."""
    return {
        "X_train_comb": global_data["smiles_combined"][train_idx],
        "X_test_comb": global_data["smiles_combined"][test_idx],
        "X_train_sol": global_data["smiles_solute"][train_idx],
        "X_test_sol": global_data["smiles_solute"][test_idx],
        "y_train": global_data["y"][train_idx],
        "y_test": global_data["y"][test_idx],
        "solvents_test": global_data["solvents"][test_idx],
        # Pass through global charset info
        "charset_combined": global_data["charset_combined"],
        "charset_solute": global_data["charset_solute"],
        "embed_combined": global_data["embed_combined"],
        "embed_solute": global_data["embed_solute"],
    }


# ─── v2: Solvent-stratified 3-way splits ─────────────────────────────────────

def create_fold_splits_stratified(global_data, n_folds=N_FOLDS, seed=SEED):
    """Create solvent-stratified train/val/test splits.

    Returns list of (train_idx, val_idx, test_idx) and saves to disk.
    """
    solvents = global_data["solvents"]
    n_samples = len(global_data["y"])

    folds = create_stratified_folds(n_samples, solvents, n_folds=n_folds, seed=seed)

    save_dict = {}
    for i, (train_idx, val_idx, test_idx) in enumerate(folds):
        save_dict[f"train_{i}"] = train_idx
        save_dict[f"val_{i}"] = val_idx
        save_dict[f"test_{i}"] = test_idx
    np.savez(os.path.join(RESULTS_DIR, "cv_fold_indices_stratified.npz"), **save_dict)
    print(f"  Stratified fold splits saved ({n_folds} folds, {n_samples} samples)")
    print(f"  Fold 0 sizes: train={len(folds[0][0])}, val={len(folds[0][1])}, test={len(folds[0][2])}")

    return folds


def prepare_fold_data_3way(global_data, train_idx, val_idx, test_idx):
    """Slice global arrays by fold indices (train/val/test 3-way split)."""
    return {
        "X_train_comb": global_data["smiles_combined"][train_idx],
        "X_val_comb": global_data["smiles_combined"][val_idx],
        "X_test_comb": global_data["smiles_combined"][test_idx],
        "X_train_sol": global_data["smiles_solute"][train_idx],
        "X_val_sol": global_data["smiles_solute"][val_idx],
        "X_test_sol": global_data["smiles_solute"][test_idx],
        "y_train": global_data["y"][train_idx],
        "y_val": global_data["y"][val_idx],
        "y_test": global_data["y"][test_idx],
        "solvents_test": global_data["solvents"][test_idx],
        # Pass through global charset info
        "charset_combined": global_data["charset_combined"],
        "charset_solute": global_data["charset_solute"],
        "embed_combined": global_data["embed_combined"],
        "embed_solute": global_data["embed_solute"],
    }


# ─── Fingerprint-based model builder ─────────────────────────────────────────

def compute_all_fps(global_data):
    """Compute Morgan fingerprints for the full dataset once. Returns (fp_array, valid_mask)."""
    print("  Computing fingerprints for full dataset (cached across folds)...")
    solute_list = []
    solvent_list = []
    for s in global_data["smiles_combined"]:
        parts = s.split("!")
        solute_list.append(parts[0])
        solvent_list.append(parts[1] if len(parts) > 1 else "")

    print("  Solute fingerprints...")
    fp_solute, mask1 = compute_morgan_fps(solute_list)
    print("  Solvent fingerprints...")
    fp_solvent, mask2 = compute_morgan_fps(solvent_list)
    fp_all = np.hstack([fp_solute, fp_solvent])
    valid_mask = mask1 & mask2
    n_valid = valid_mask.sum()
    print(f"  Full dataset FP: {n_valid}/{len(valid_mask)} valid molecules")
    return fp_all, valid_mask


def compute_all_fps_nbits(global_data, n_bits=2048):
    """Compute Morgan fingerprints with custom bit size. Returns (fp_array, valid_mask)."""
    print(f"  Computing {n_bits}-bit fingerprints for full dataset...")
    solute_list = []
    solvent_list = []
    for s in global_data["smiles_combined"]:
        parts = s.split("!")
        solute_list.append(parts[0])
        solvent_list.append(parts[1] if len(parts) > 1 else "")

    print(f"  Solute fingerprints ({n_bits}-bit)...")
    fp_solute, mask1 = compute_morgan_fps(solute_list, n_bits=n_bits)
    print(f"  Solvent fingerprints ({n_bits}-bit)...")
    fp_solvent, mask2 = compute_morgan_fps(solvent_list, n_bits=n_bits)
    fp_all = np.hstack([fp_solute, fp_solvent])
    valid_mask = mask1 & mask2
    n_valid = valid_mask.sum()
    print(f"  Full dataset FP ({n_bits}-bit): {n_valid}/{len(valid_mask)} valid, dim={fp_all.shape[1]}")
    return fp_all, valid_mask


def compute_solute_only_fps(global_data, n_bits=2048):
    """Compute Morgan fingerprints for solute ONLY (no solvent). Returns (fp_array, valid_mask)."""
    print(f"  Computing solute-only {n_bits}-bit fingerprints for full dataset...")
    solute_list = list(global_data["smiles_solute"])
    print(f"  Solute fingerprints ({n_bits}-bit)...")
    fp_solute, mask = compute_morgan_fps(solute_list, n_bits=n_bits)
    n_valid = mask.sum()
    print(f"  Solute-only FP: {n_valid}/{len(mask)} valid, dim={fp_solute.shape[1]}")
    return fp_solute, mask


def train_fp_model(model_class, model_name, X_train_fp, X_test_fp,
                   y_train_fp, y_test_fp, **kwargs):
    print(f"\n  >> Training {model_name}...")
    t0 = time.time()
    model = model_class(**kwargs)
    model.fit(X_train_fp, y_train_fp.flatten())
    elapsed = time.time() - t0
    y_pred = model.predict(X_test_fp)
    metrics = compute_metrics(y_test_fp.flatten(), y_pred.flatten())
    print(f"  >> {model_name} done in {elapsed:.0f}s — RMSE={metrics['RMSE']} MAE={metrics['MAE']} R²={metrics['R2']} r={metrics['Pearson_r']}")
    return y_pred.flatten(), metrics, model


# ─── Save / load per-fold results ────────────────────────────────────────────

def save_fold_results(model_key, fold_idx, metrics, y_pred, y_test, test_indices):
    """Save metrics, predictions, y_test, and test indices for a single fold."""
    prefix = f"{model_key}_fold{fold_idx}"
    with open(os.path.join(RESULTS_DIR, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, f"{prefix}_predictions.npy"), y_pred)
    np.save(os.path.join(RESULTS_DIR, f"{prefix}_y_test.npy"), y_test)
    np.save(os.path.join(RESULTS_DIR, f"{prefix}_test_indices.npy"), test_indices)


def load_fold_results(model_key, fold_idx):
    """Load saved fold results. Returns (metrics, y_pred, y_test, test_indices) or None."""
    prefix = f"{model_key}_fold{fold_idx}"
    paths = [
        os.path.join(RESULTS_DIR, f"{prefix}_metrics.json"),
        os.path.join(RESULTS_DIR, f"{prefix}_predictions.npy"),
        os.path.join(RESULTS_DIR, f"{prefix}_y_test.npy"),
        os.path.join(RESULTS_DIR, f"{prefix}_test_indices.npy"),
    ]
    if not all(os.path.exists(p) for p in paths):
        return None
    with open(paths[0]) as f:
        metrics = json.load(f)
    y_pred = np.load(paths[1])
    y_test = np.load(paths[2])
    test_indices = np.load(paths[3])
    return metrics, y_pred, y_test, test_indices


# ─── Aggregate fold results ──────────────────────────────────────────────────

def aggregate_fold_results(model_key, n_folds=N_FOLDS):
    """Load all fold results, compute mean +/- std, pool predictions."""
    all_metrics = []
    all_y_pred = []
    all_y_test = []
    all_test_idx = []

    for i in range(n_folds):
        result = load_fold_results(model_key, i)
        if result is None:
            print(f"  [WARN] Missing fold {i} for {model_key}")
            continue
        metrics, y_pred, y_test, test_idx = result
        all_metrics.append(metrics)
        all_y_pred.append(y_pred)
        all_y_test.append(y_test)
        all_test_idx.append(test_idx)

    n_found = len(all_metrics)
    if n_found == 0:
        return None

    # Mean +/- std across folds
    metric_keys = ["RMSE", "MAE", "R2", "Pearson_r"]
    agg = {"n_folds_found": n_found}
    for mk in metric_keys:
        vals = [m[mk] for m in all_metrics]
        agg[f"{mk}_mean"] = round(np.mean(vals), 4)
        agg[f"{mk}_std"] = round(np.std(vals), 4)
        agg[f"{mk}_values"] = vals

    # Save aggregate
    agg_path = os.path.join(RESULTS_DIR, f"{model_key}_cv_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    # Pool predictions (every sample appears in exactly one test fold)
    pooled_pred = np.concatenate(all_y_pred)
    pooled_test = np.concatenate(all_y_test)
    pooled_idx = np.concatenate(all_test_idx)

    # Sort by original index so pooled arrays are in dataset order
    sort_order = np.argsort(pooled_idx)
    pooled_pred = pooled_pred[sort_order]
    pooled_test = pooled_test[sort_order]
    pooled_idx = pooled_idx[sort_order]

    np.savez(os.path.join(RESULTS_DIR, f"{model_key}_cv_pooled.npz"),
             y_pred=pooled_pred, y_test=pooled_test, indices=pooled_idx)

    # Compute pooled metrics
    pooled_metrics = compute_metrics(pooled_test, pooled_pred)
    agg["pooled_metrics"] = pooled_metrics

    # Re-save with pooled metrics
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"  {model_key}: {n_found}/{n_folds} folds aggregated")
    for mk in metric_keys:
        print(f"    {mk}: {agg[f'{mk}_mean']:.4f} +/- {agg[f'{mk}_std']:.4f}")

    return agg


# ─── Save global config (once per model_key) ─────────────────────────────────

def save_global_config(model_key, global_data):
    """Save charset + config for inference. Only needs to be saved once (not per-fold)."""
    _, model_type, with_solvent = MODEL_REGISTRY[model_key]

    if with_solvent is not None:
        charset = global_data["charset_combined"] if with_solvent else global_data["charset_solute"]
        embed_len = global_data["embed_combined"] if with_solvent else global_data["embed_solute"]
    else:
        # FP models don't need charset, but save for consistency
        charset = global_data["charset_combined"]
        embed_len = global_data["embed_combined"]

    config = {
        "charset": charset,
        "embed_len": embed_len,
        "char_to_int": {c: i for i, c in enumerate(charset)},
        "model_type": model_type,
        "with_solvent": with_solvent,
        "n_folds": N_FOLDS,
        "seed": SEED,
    }
    config_path = os.path.join(RESULTS_DIR, f"{model_key}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config


# ─── Run a single model for one fold ─────────────────────────────────────────

def run_single_fold(model_key, fold_idx, fold_data, test_indices, global_data,
                    fp_all=None, fp_valid_mask=None,
                    fp_all_1024=None, fp_valid_mask_1024=None,
                    fp_solute_only=None, fp_solute_only_mask=None):
    """Run one model on one fold. Returns (metrics, y_pred)."""
    display_name, model_type, with_solvent = MODEL_REGISTRY[model_key]
    safe_name = display_name.replace(" ", "_").replace("+", "w").replace("(", "").replace(")", "")
    fold_name = f"{safe_name}_fold{fold_idx}"

    if model_type in ("bigru", "bilstm", "cnn_bigru"):
        import tensorflow as tf
        tf.random.set_seed(SEED + fold_idx)

        # Vectorize
        print(f"\n  [Fold {fold_idx}] Vectorizing sequences...")
        if with_solvent:
            c2i = build_char_maps(fold_data["charset_combined"])
            X_train_vec = vectorize_smiles(fold_data["X_train_comb"], c2i, fold_data["embed_combined"])
            X_test_vec = vectorize_smiles(fold_data["X_test_comb"], c2i, fold_data["embed_combined"])
            num_words = len(fold_data["charset_combined"])
            input_length = fold_data["embed_combined"] - 1
        else:
            c2i = build_char_maps(fold_data["charset_solute"])
            X_train_vec = vectorize_smiles(fold_data["X_train_sol"], c2i, fold_data["embed_solute"])
            X_test_vec = vectorize_smiles(fold_data["X_test_sol"], c2i, fold_data["embed_solute"])
            num_words = len(fold_data["charset_solute"])
            input_length = fold_data["embed_solute"] - 1
        print("  Done.")

        print(f"\n  [Fold {fold_idx}] Training {display_name}...")
        y_pred, metrics, trained_model = train_dl_model(
            model_type,
            X_train_vec, X_test_vec,
            fold_data["y_train"].reshape(-1, 1), fold_data["y_test"].reshape(-1, 1),
            num_words=num_words, input_length=input_length, name=fold_name,
            results_dir=RESULTS_DIR, seed=SEED, epochs=EPOCHS,
            batch_size=BATCH_SIZE, patience=PATIENCE
        )

        # Save fold results
        save_fold_results(model_key, fold_idx, metrics, y_pred,
                          fold_data["y_test"], test_indices)

        # Save per-fold trained Keras model
        model_path = os.path.join(RESULTS_DIR, f"{model_key}_fold{fold_idx}_model.keras")
        trained_model.save(model_path)
        print(f"  >> Fold model saved to {model_path}")

        # Cleanup
        del trained_model, X_train_vec, X_test_vec
        tf.keras.backend.clear_session()
        gc.collect()

    elif model_type in ("rf", "rf_mae", "rf_1024", "rf_nosolvent", "xgb", "xgb_mae"):
        import joblib

        # Select correct FP arrays (1024-bit, solute-only, or default 2048-bit)
        if model_type == "rf_1024":
            _fp_all = fp_all_1024
            _fp_mask = fp_valid_mask_1024
        elif model_type == "rf_nosolvent":
            _fp_all = fp_solute_only
            _fp_mask = fp_solute_only_mask
        else:
            _fp_all = fp_all
            _fp_mask = fp_valid_mask

        # Use pre-computed fingerprints sliced by fold index
        train_idx = np.setdiff1d(np.arange(len(global_data["y"])), test_indices)

        X_train_fp = _fp_all[train_idx]
        X_test_fp = _fp_all[test_indices]
        mask_train = _fp_mask[train_idx]
        mask_test = _fp_mask[test_indices]

        X_train_fp_valid = X_train_fp[mask_train]
        y_train_valid = global_data["y"][train_idx][mask_train]
        X_test_fp_valid = X_test_fp[mask_test]
        y_test_valid = global_data["y"][test_indices][mask_test]
        print(f"  [Fold {fold_idx}] Valid: {mask_train.sum()}/{len(mask_train)} train, "
              f"{mask_test.sum()}/{len(mask_test)} test")

        print(f"\n  [Fold {fold_idx}] Training {display_name}...")
        if model_type in ("rf", "rf_1024", "rf_nosolvent"):
            from sklearn.ensemble import RandomForestRegressor
            y_pred, metrics, trained_model = train_fp_model(
                RandomForestRegressor, f"{display_name} fold{fold_idx}",
                X_train_fp_valid, X_test_fp_valid,
                y_train_valid.reshape(-1, 1), y_test_valid.reshape(-1, 1),
                n_estimators=500, max_depth=None, n_jobs=-1, random_state=SEED
            )
        elif model_type == "rf_mae":
            from sklearn.ensemble import RandomForestRegressor
            y_pred, metrics, trained_model = train_fp_model(
                RandomForestRegressor, f"{display_name} fold{fold_idx}",
                X_train_fp_valid, X_test_fp_valid,
                y_train_valid.reshape(-1, 1), y_test_valid.reshape(-1, 1),
                criterion="absolute_error",
                n_estimators=500, max_depth=None, n_jobs=-1, random_state=SEED
            )
        elif model_type == "xgb":
            import xgboost as xgb
            y_pred, metrics, trained_model = train_fp_model(
                xgb.XGBRegressor, f"{display_name} fold{fold_idx}",
                X_train_fp_valid, X_test_fp_valid,
                y_train_valid.reshape(-1, 1), y_test_valid.reshape(-1, 1),
                n_estimators=500, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=SEED, tree_method="hist"
            )
        elif model_type == "xgb_mae":
            import xgboost as xgb
            y_pred, metrics, trained_model = train_fp_model(
                xgb.XGBRegressor, f"{display_name} fold{fold_idx}",
                X_train_fp_valid, X_test_fp_valid,
                y_train_valid.reshape(-1, 1), y_test_valid.reshape(-1, 1),
                objective="reg:absoluteerror",
                n_estimators=500, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=SEED, tree_method="hist"
            )

        # Save fold results (use valid-masked y_test)
        # For FP models test_indices are the valid subset
        valid_test_indices = test_indices[mask_test]
        save_fold_results(model_key, fold_idx, metrics, y_pred,
                          y_test_valid, valid_test_indices)

        # Save per-fold trained model via joblib
        model_path = os.path.join(RESULTS_DIR, f"{model_key}_fold{fold_idx}_model.joblib")
        joblib.dump(trained_model, model_path)
        print(f"  >> Fold model saved to {model_path}")

        del trained_model
        gc.collect()

    return metrics, y_pred


# ─── Run all folds for a model ────────────────────────────────────────────────

def run_model_cv(model_key, global_data, folds, fold_idx=None,
                 fp_all=None, fp_valid_mask=None,
                 fp_all_1024=None, fp_valid_mask_1024=None,
                 fp_solute_only=None, fp_solute_only_mask=None):
    """Run 5-fold CV for a model (or a single fold if fold_idx is specified)."""
    display_name = MODEL_REGISTRY[model_key][0]

    # Save global config once
    save_global_config(model_key, global_data)

    if fold_idx is not None:
        # Run a single fold
        fold_range = [fold_idx]
    else:
        fold_range = range(len(folds))

    for i in fold_range:
        # Check if fold already done (skip for resumability)
        existing = load_fold_results(model_key, i)
        if existing is not None:
            print(f"\n  [Fold {i}] Already completed (RMSE={existing[0]['RMSE']}). Skipping.")
            continue

        train_idx, test_idx = folds[i]
        fold_data = prepare_fold_data(global_data, train_idx, test_idx)

        print(f"\n{'─'*70}")
        print(f"  {display_name} — Fold {i}/{N_FOLDS-1}  "
              f"(train={len(train_idx)}, test={len(test_idx)})")
        print(f"{'─'*70}")

        run_single_fold(model_key, i, fold_data, test_idx, global_data,
                        fp_all=fp_all, fp_valid_mask=fp_valid_mask,
                        fp_all_1024=fp_all_1024, fp_valid_mask_1024=fp_valid_mask_1024,
                        fp_solute_only=fp_solute_only, fp_solute_only_mask=fp_solute_only_mask)

    # Aggregate if all folds are done (or after the single fold for partial aggregation)
    print(f"\n  Aggregating results for {display_name}...")
    agg = aggregate_fold_results(model_key)
    return agg


# ─── v2: Run a single fold with 3-way split ──────────────────────────────────

def run_single_fold_v2(model_key, fold_idx, fold_data,
                       train_indices, val_indices, test_indices,
                       global_data, fp_all=None, fp_valid_mask=None,
                       fp_all_1024=None, fp_valid_mask_1024=None,
                       fp_solute_only=None, fp_solute_only_mask=None):
    """Run one model on one fold with 3-way train/val/test split.

    For DL models, passes X_val/y_val to train_dl_model for proper early stopping.
    For FP models, trains on train set only (val unused).
    """
    display_name, model_type, with_solvent = MODEL_REGISTRY[model_key]
    safe_name = display_name.replace(" ", "_").replace("+", "w").replace("(", "").replace(")", "")
    fold_name = f"{safe_name}_fold{fold_idx}"

    if model_type in ("bigru", "bilstm", "cnn_bigru"):
        import tensorflow as tf
        tf.random.set_seed(SEED + fold_idx)

        print(f"\n  [Fold {fold_idx}] Vectorizing sequences...")
        if with_solvent:
            c2i = build_char_maps(fold_data["charset_combined"])
            X_train_vec = vectorize_smiles(fold_data["X_train_comb"], c2i, fold_data["embed_combined"])
            X_val_vec = vectorize_smiles(fold_data["X_val_comb"], c2i, fold_data["embed_combined"])
            X_test_vec = vectorize_smiles(fold_data["X_test_comb"], c2i, fold_data["embed_combined"])
            num_words = len(fold_data["charset_combined"])
            input_length = fold_data["embed_combined"] - 1
        else:
            c2i = build_char_maps(fold_data["charset_solute"])
            X_train_vec = vectorize_smiles(fold_data["X_train_sol"], c2i, fold_data["embed_solute"])
            X_val_vec = vectorize_smiles(fold_data["X_val_sol"], c2i, fold_data["embed_solute"])
            X_test_vec = vectorize_smiles(fold_data["X_test_sol"], c2i, fold_data["embed_solute"])
            num_words = len(fold_data["charset_solute"])
            input_length = fold_data["embed_solute"] - 1
        print("  Done.")

        print(f"\n  [Fold {fold_idx}] Training {display_name}...")
        y_pred, metrics, trained_model = train_dl_model(
            model_type,
            X_train_vec, X_test_vec,
            fold_data["y_train"].reshape(-1, 1), fold_data["y_test"].reshape(-1, 1),
            num_words=num_words, input_length=input_length, name=fold_name,
            results_dir=RESULTS_DIR, seed=SEED, epochs=EPOCHS,
            batch_size=BATCH_SIZE, patience=PATIENCE,
            X_val=X_val_vec, y_val=fold_data["y_val"].reshape(-1, 1),
        )

        save_fold_results(model_key, fold_idx, metrics, y_pred,
                          fold_data["y_test"], test_indices)

        model_path = os.path.join(RESULTS_DIR, f"{model_key}_fold{fold_idx}_model.keras")
        trained_model.save(model_path)
        print(f"  >> Fold model saved to {model_path}")

        del trained_model, X_train_vec, X_val_vec, X_test_vec
        tf.keras.backend.clear_session()
        gc.collect()

    elif model_type in ("rf", "rf_mae", "rf_1024", "rf_nosolvent", "xgb", "xgb_mae"):
        import joblib

        # Select correct FP arrays (1024-bit, solute-only, or default 2048-bit)
        if model_type == "rf_1024":
            _fp_all = fp_all_1024
            _fp_mask = fp_valid_mask_1024
        elif model_type == "rf_nosolvent":
            _fp_all = fp_solute_only
            _fp_mask = fp_solute_only_mask
        else:
            _fp_all = fp_all
            _fp_mask = fp_valid_mask

        # FP models: train on train_indices only (val unused)
        X_train_fp = _fp_all[train_indices]
        X_test_fp = _fp_all[test_indices]
        mask_train = _fp_mask[train_indices]
        mask_test = _fp_mask[test_indices]

        X_train_fp_valid = X_train_fp[mask_train]
        y_train_valid = global_data["y"][train_indices][mask_train]
        X_test_fp_valid = X_test_fp[mask_test]
        y_test_valid = global_data["y"][test_indices][mask_test]
        print(f"  [Fold {fold_idx}] Valid: {mask_train.sum()}/{len(mask_train)} train, "
              f"{mask_test.sum()}/{len(mask_test)} test")

        print(f"\n  [Fold {fold_idx}] Training {display_name}...")
        if model_type in ("rf", "rf_1024", "rf_nosolvent"):
            from sklearn.ensemble import RandomForestRegressor
            y_pred, metrics, trained_model = train_fp_model(
                RandomForestRegressor, f"{display_name} fold{fold_idx}",
                X_train_fp_valid, X_test_fp_valid,
                y_train_valid.reshape(-1, 1), y_test_valid.reshape(-1, 1),
                n_estimators=500, max_depth=None, n_jobs=-1, random_state=SEED
            )
        elif model_type == "rf_mae":
            from sklearn.ensemble import RandomForestRegressor
            y_pred, metrics, trained_model = train_fp_model(
                RandomForestRegressor, f"{display_name} fold{fold_idx}",
                X_train_fp_valid, X_test_fp_valid,
                y_train_valid.reshape(-1, 1), y_test_valid.reshape(-1, 1),
                criterion="absolute_error",
                n_estimators=500, max_depth=None, n_jobs=-1, random_state=SEED
            )
        elif model_type == "xgb":
            import xgboost as xgb
            y_pred, metrics, trained_model = train_fp_model(
                xgb.XGBRegressor, f"{display_name} fold{fold_idx}",
                X_train_fp_valid, X_test_fp_valid,
                y_train_valid.reshape(-1, 1), y_test_valid.reshape(-1, 1),
                n_estimators=500, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=SEED, tree_method="hist"
            )
        elif model_type == "xgb_mae":
            import xgboost as xgb
            y_pred, metrics, trained_model = train_fp_model(
                xgb.XGBRegressor, f"{display_name} fold{fold_idx}",
                X_train_fp_valid, X_test_fp_valid,
                y_train_valid.reshape(-1, 1), y_test_valid.reshape(-1, 1),
                objective="reg:absoluteerror",
                n_estimators=500, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=SEED, tree_method="hist"
            )

        valid_test_indices = test_indices[mask_test]
        save_fold_results(model_key, fold_idx, metrics, y_pred,
                          y_test_valid, valid_test_indices)

        model_path = os.path.join(RESULTS_DIR, f"{model_key}_fold{fold_idx}_model.joblib")
        joblib.dump(trained_model, model_path)
        print(f"  >> Fold model saved to {model_path}")

        del trained_model
        gc.collect()

    return metrics, y_pred


def run_model_cv_v2(model_key, global_data, folds, fold_idx=None,
                    fp_all=None, fp_valid_mask=None,
                    fp_all_1024=None, fp_valid_mask_1024=None,
                    fp_solute_only=None, fp_solute_only_mask=None):
    """Run 5-fold CV with 3-way stratified splits (v2)."""
    display_name = MODEL_REGISTRY[model_key][0]

    save_global_config(model_key, global_data)

    if fold_idx is not None:
        fold_range = [fold_idx]
    else:
        fold_range = range(len(folds))

    for i in fold_range:
        existing = load_fold_results(model_key, i)
        if existing is not None:
            print(f"\n  [Fold {i}] Already completed (RMSE={existing[0]['RMSE']}). Skipping.")
            continue

        train_idx, val_idx, test_idx = folds[i]
        fold_data = prepare_fold_data_3way(global_data, train_idx, val_idx, test_idx)

        print(f"\n{'─'*70}")
        print(f"  {display_name} — Fold {i}/{N_FOLDS-1}  "
              f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
        print(f"{'─'*70}")

        run_single_fold_v2(model_key, i, fold_data,
                           train_idx, val_idx, test_idx, global_data,
                           fp_all=fp_all, fp_valid_mask=fp_valid_mask,
                           fp_all_1024=fp_all_1024, fp_valid_mask_1024=fp_valid_mask_1024,
                           fp_solute_only=fp_solute_only, fp_solute_only_mask=fp_solute_only_mask)

    print(f"\n  Aggregating results for {display_name}...")
    agg = aggregate_fold_results(model_key)
    return agg


# ─── Generate summary + plots ───────────────────────────────────────────────

def generate_summary(v2=False):
    """Load all saved model results and generate comparison table + plots."""
    order = MODEL_ORDER_V2 if v2 else MODEL_ORDER
    label = "v2 stratified 3-way" if v2 else "5-Fold CV"

    print("\n" + "=" * 70)
    print(f"  GENERATING SUMMARY FROM SAVED RESULTS ({label})")
    print("=" * 70)

    all_agg = {}
    for key in order:
        agg_path = os.path.join(RESULTS_DIR, f"{key}_cv_aggregate.json")
        if os.path.exists(agg_path):
            with open(agg_path) as f:
                agg = json.load(f)
            display_name = MODEL_REGISTRY[key][0]
            all_agg[key] = agg
            n = agg.get("n_folds_found", "?")
            print(f"  Loaded {display_name}: {n} folds, "
                  f"RMSE={agg['RMSE_mean']:.2f} +/- {agg['RMSE_std']:.2f}")
        else:
            print(f"  [SKIP] {MODEL_REGISTRY[key][0]} — no aggregate results")

    if not all_agg:
        print("  No results found! Run models first.")
        return

    # Summary table: mean +/- std
    rows = []
    for key in order:
        if key not in all_agg:
            continue
        agg = all_agg[key]
        display_name = MODEL_REGISTRY[key][0]
        rows.append({
            "Model": display_name,
            "RMSE": f"{agg['RMSE_mean']:.2f} +/- {agg['RMSE_std']:.2f}",
            "MAE": f"{agg['MAE_mean']:.2f} +/- {agg['MAE_std']:.2f}",
            "R2": f"{agg['R2_mean']:.4f} +/- {agg['R2_std']:.4f}",
            "Pearson_r": f"{agg['Pearson_r_mean']:.4f} +/- {agg['Pearson_r_std']:.4f}",
        })

    results_df = pd.DataFrame(rows).set_index("Model")
    print(f"\n{results_df.to_string()}")
    csv_name = "baseline_comparison_v2.csv" if v2 else "baseline_comparison.csv"
    results_df.to_csv(os.path.join(RESULTS_DIR, csv_name))
    print(f"\nSaved to {os.path.join(RESULTS_DIR, csv_name)}")

    # ─── Plots (using BiGRU + Solvent pooled predictions) ─────────────────
    bigru_key = "bigru_solvent_v2" if v2 else "bigru_solvent"
    pooled_path = os.path.join(RESULTS_DIR, f"{bigru_key}_cv_pooled.npz")
    if not os.path.exists(pooled_path):
        print("  [SKIP] Plots require bigru_solvent pooled results.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = np.load(pooled_path)
    y_test = pooled["y_test"]
    y_pred_bigru = pooled["y_pred"]

    bigru_agg = all_agg.get(bigru_key, {})
    bigru_rmse = bigru_agg.get("RMSE_mean", 0)

    # Parity plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(y_test, y_pred_bigru, alpha=0.3, s=10, c="steelblue", edgecolors="none")
    lims = [min(y_test.min(), y_pred_bigru.min()) - 10,
            max(y_test.max(), y_pred_bigru.max()) + 10]
    ax.plot(lims, lims, "k--", lw=1, label="Ideal")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Experimental $\\lambda_{\\max}$ (nm)", fontsize=12)
    ax.set_ylabel("Predicted $\\lambda_{\\max}$ (nm)", fontsize=12)
    ax.set_title(f"BiGRU + Solvent — 5-Fold CV (RMSE = {bigru_rmse:.1f} nm)", fontsize=13)
    ax.set_aspect("equal"); ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "parity_plot.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "parity_plot.pdf"))
    print("  Parity plot saved.")

    # Error distribution
    errors_bigru = y_pred_bigru - y_test
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(errors_bigru, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", lw=1)
    ax.set_xlabel("Prediction Error (nm)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Error Distribution — BiGRU + Solvent (5-Fold CV)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.pdf"))
    print("  Error distribution saved.")

    # Error by wavelength range
    abs_errors = np.abs(errors_bigru)
    ranges = [("< 300", y_test < 300),
              ("300-400", (y_test >= 300) & (y_test < 400)),
              ("400-500", (y_test >= 400) & (y_test < 500)),
              ("> 500", y_test >= 500)]

    range_labels, range_means, range_stds, range_counts = [], [], [], []
    for label, mask in ranges:
        if mask.sum() > 0:
            range_labels.append(label)
            range_means.append(abs_errors[mask].mean())
            range_stds.append(abs_errors[mask].std())
            range_counts.append(mask.sum())

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    bars = ax.bar(range_labels, range_means, yerr=range_stds, capsize=5,
                  color="steelblue", edgecolor="white", alpha=0.85)
    for bar, count in zip(bars, range_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={count}", ha="center", fontsize=9)
    ax.set_xlabel("$\\lambda_{\\max}$ Range (nm)", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (nm)", fontsize=12)
    ax.set_title("Error by Wavelength Range — BiGRU + Solvent (5-Fold CV)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_by_wavelength.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "error_by_wavelength.pdf"))
    print("  Error by wavelength saved.")

    # Error by solvent (top 5) — reconstruct from pooled data + global data
    try:
        global_data = load_and_prepare_global_data()
        pooled_indices = pooled["indices"]
        solvent_arr = global_data["solvents"][pooled_indices]

        solvent_names = {
            "CCO": "Ethanol", "CO": "Methanol", "ClCCl": "DCM",
            "CC(C)=O": "Acetone", "CCCCCC": "Hexane", "CS(C)=O": "DMSO",
            "CC#N": "Acetonitrile", "O": "Water", "Cc1ccccc1": "Toluene",
            "C1CCOC1": "THF", "ClC(Cl)Cl": "Chloroform",
            "CCOCC": "Diethyl ether", "c1ccccc1": "Benzene",
            "C(Cl)(Cl)Cl": "Chloroform2",
        }

        unique_solvents, counts = np.unique(solvent_arr, return_counts=True)
        top5_idx = np.argsort(-counts)[:5]
        top5_solvents = unique_solvents[top5_idx]

        sol_labels, sol_errors = [], []
        for sol in top5_solvents:
            mask = solvent_arr == sol
            sol_labels.append(solvent_names.get(sol, sol[:15]))
            sol_errors.append(abs_errors[mask])

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        bp = ax.boxplot(sol_errors, labels=sol_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
        ax.set_xlabel("Solvent", fontsize=12)
        ax.set_ylabel("Absolute Error (nm)", fontsize=12)
        ax.set_title("Error by Solvent — BiGRU + Solvent (5-Fold CV)", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "error_by_solvent.png"), dpi=300)
        plt.savefig(os.path.join(RESULTS_DIR, "error_by_solvent.pdf"))
        print("  Error by solvent saved.")
    except Exception as e:
        print(f"  [SKIP] Error by solvent plot failed: {e}")

    print("\n" + "=" * 70)
    print("  SUMMARY COMPLETE")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="UV λ_max baseline experiments (5-fold CV)")
    # Use base model names (without _v2 suffix) for --model choices
    base_model_keys = list(dict.fromkeys(
        k.removesuffix("_v2") for k in MODEL_REGISTRY
    ))
    parser.add_argument("--model", type=str, choices=base_model_keys,
                        help="Run a single model (e.g., bigru_solvent, rf, xgboost)")
    parser.add_argument("--fold", type=int, choices=list(range(N_FOLDS)),
                        help="Run a single fold (0-4). Requires --model.")
    parser.add_argument("--summary", action="store_true",
                        help="Generate summary table and plots from saved results")
    parser.add_argument("--v2", action="store_true",
                        help="Use v2: solvent-stratified 3-way splits (train/val/test)")
    args = parser.parse_args()

    if args.fold is not None and args.model is None:
        parser.error("--fold requires --model")

    version_label = "v2 stratified 3-way" if args.v2 else "5-Fold CV"
    print("\n" + "=" * 70)
    print(f"  UV λ_max BASELINE EXPERIMENTS — {version_label}")
    print("=" * 70)

    if args.summary:
        generate_summary(v2=args.v2)
        return

    # Verify GPU
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[GPU] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[GPU] Mixed precision (float16) enabled")
        # Run quick benchmark to confirm GPU computation works
        gpu_ok = verify_gpu_available()
        if not gpu_ok:
            print("[GPU] WARNING: GPU benchmark failed. Check CUDA/cuDNN installation.")
    else:
        print("[WARN] No GPU detected — DL training will be very slow on CPU!")
        print("[WARN] Expected ~100-120s/epoch WITH GPU, ~10-20min/epoch on CPU")

    # Load global data (no splitting)
    global_data = load_and_prepare_global_data()

    if args.v2:
        # ─── v2: Solvent-stratified 3-way splits ─────────────────────────
        folds = create_fold_splits_stratified(global_data)

        # Determine models to run (append _v2 suffix)
        if args.model:
            models_to_run = [args.model + "_v2"]
        else:
            models_to_run = MODEL_ORDER_V2

        # Pre-compute fingerprints if needed
        need_fp_2048 = any(MODEL_REGISTRY[k][1] in ("rf", "rf_mae", "xgb", "xgb_mae")
                           for k in models_to_run)
        need_fp_1024 = any(MODEL_REGISTRY[k][1] == "rf_1024" for k in models_to_run)
        need_fp_solute = any(MODEL_REGISTRY[k][1] == "rf_nosolvent" for k in models_to_run)
        fp_all, fp_valid_mask = None, None
        fp_all_1024, fp_valid_mask_1024 = None, None
        fp_solute_only, fp_solute_only_mask = None, None
        if need_fp_2048:
            print("\n[FP] Pre-computing 2048-bit Morgan fingerprints for all data...")
            fp_all, fp_valid_mask = compute_all_fps(global_data)
        if need_fp_1024:
            print("\n[FP] Pre-computing 1024-bit Morgan fingerprints for all data...")
            fp_all_1024, fp_valid_mask_1024 = compute_all_fps_nbits(global_data, n_bits=1024)
        if need_fp_solute:
            print("\n[FP] Pre-computing solute-only 2048-bit Morgan fingerprints...")
            fp_solute_only, fp_solute_only_mask = compute_solute_only_fps(global_data)

        # Run v2 models
        for key in models_to_run:
            display_name = MODEL_REGISTRY[key][0]
            fold_str = f" (fold {args.fold})" if args.fold is not None else " (5-fold CV)"
            print(f"\n{'='*70}")
            print(f"  Running: {display_name}{fold_str}")
            print(f"{'='*70}")

            agg = run_model_cv_v2(key, global_data, folds, fold_idx=args.fold,
                                  fp_all=fp_all, fp_valid_mask=fp_valid_mask,
                                  fp_all_1024=fp_all_1024, fp_valid_mask_1024=fp_valid_mask_1024,
                                  fp_solute_only=fp_solute_only, fp_solute_only_mask=fp_solute_only_mask)

            if agg:
                n = agg.get("n_folds_found", "?")
                print(f"\n  {display_name}: {n} folds — "
                      f"RMSE={agg['RMSE_mean']:.2f}+/-{agg['RMSE_std']:.2f}  "
                      f"MAE={agg['MAE_mean']:.2f}+/-{agg['MAE_std']:.2f}  "
                      f"R²={agg['R2_mean']:.4f}+/-{agg['R2_std']:.4f}  "
                      f"r={agg['Pearson_r_mean']:.4f}+/-{agg['Pearson_r_std']:.4f}")

        if args.model is None:
            generate_summary(v2=True)

    else:
        # ─── v1: Original KFold 2-way splits ─────────────────────────────
        n_samples = len(global_data["y"])
        folds = create_fold_splits(n_samples)

        models_to_run = [args.model] if args.model else MODEL_ORDER
        need_fp_2048 = any(MODEL_REGISTRY[k][1] in ("rf", "rf_mae", "xgb", "xgb_mae")
                           for k in models_to_run)
        need_fp_1024 = any(MODEL_REGISTRY[k][1] == "rf_1024" for k in models_to_run)
        need_fp_solute = any(MODEL_REGISTRY[k][1] == "rf_nosolvent" for k in models_to_run)
        fp_all, fp_valid_mask = None, None
        fp_all_1024, fp_valid_mask_1024 = None, None
        fp_solute_only, fp_solute_only_mask = None, None
        if need_fp_2048:
            print("\n[FP] Pre-computing 2048-bit Morgan fingerprints for all data...")
            fp_all, fp_valid_mask = compute_all_fps(global_data)
        if need_fp_1024:
            print("\n[FP] Pre-computing 1024-bit Morgan fingerprints for all data...")
            fp_all_1024, fp_valid_mask_1024 = compute_all_fps_nbits(global_data, n_bits=1024)
        if need_fp_solute:
            print("\n[FP] Pre-computing solute-only 2048-bit Morgan fingerprints...")
            fp_solute_only, fp_solute_only_mask = compute_solute_only_fps(global_data)

        for key in models_to_run:
            display_name = MODEL_REGISTRY[key][0]
            fold_str = f" (fold {args.fold})" if args.fold is not None else " (5-fold CV)"
            print(f"\n{'='*70}")
            print(f"  Running: {display_name}{fold_str}")
            print(f"{'='*70}")

            agg = run_model_cv(key, global_data, folds, fold_idx=args.fold,
                               fp_all=fp_all, fp_valid_mask=fp_valid_mask,
                               fp_all_1024=fp_all_1024, fp_valid_mask_1024=fp_valid_mask_1024,
                               fp_solute_only=fp_solute_only, fp_solute_only_mask=fp_solute_only_mask)

            if agg:
                n = agg.get("n_folds_found", "?")
                print(f"\n  {display_name}: {n} folds — "
                      f"RMSE={agg['RMSE_mean']:.2f}+/-{agg['RMSE_std']:.2f}  "
                      f"MAE={agg['MAE_mean']:.2f}+/-{agg['MAE_std']:.2f}  "
                      f"R²={agg['R2_mean']:.4f}+/-{agg['R2_std']:.4f}  "
                      f"r={agg['Pearson_r_mean']:.4f}+/-{agg['Pearson_r_std']:.4f}")

        if args.model is None:
            generate_summary()

    print("\n  Done.")


if __name__ == "__main__":
    main()
