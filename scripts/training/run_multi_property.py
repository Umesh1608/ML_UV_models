#!/usr/bin/env python3
"""
Multi-property benchmark: RF, XGBoost, Chemprop, BiGRU, and ChemBERTa across 4 properties.

Tests the locality hypothesis — local-bias models (RF, Chemprop) should dominate on
LOCAL properties while global-attention models (ChemBERTa) compete on GLOBAL properties.

Properties:
  1. Fluorescence emission (λ_em)  — LOCAL, with solvent
  2. log₁₀(MEC)                    — LOCAL, with solvent (RF/XGBoost only — too large for DL)
  3. Lipophilicity (LogP)           — GLOBAL, no solvent
  4. Aqueous solubility (logS)     — GLOBAL, no solvent

Usage:
  python3 run_multi_property.py --property emission --model rf                 # RF on emission, all folds
  python3 run_multi_property.py --property emission --model bigru --fold 0     # BiGRU fold 0
  python3 run_multi_property.py --property solubility --model chemprop         # Chemprop on solubility
  python3 run_multi_property.py --property lipophilicity --model chemberta     # ChemBERTa all folds
  python3 run_multi_property.py --property log_mec --model rf                  # RF on MEC
  python3 run_multi_property.py --summary                                      # aggregate + compare
"""

import argparse
import gc
import json
import os
import sys
import time

# ─── GPU / CUDA setup ────────────────────────────────────────────────────────
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/home/umesh/.local/cuda_compat"
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from paper1_new_cl.models import (
    build_char_maps,
    compute_metrics,
    compute_morgan_fps,
    train_dl_model,
    vectorize_smiles,
    verify_gpu_available,
)
from paper1_new_cl.splits import create_solvent_bins, create_stratified_folds

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_BASE = os.path.join(SCRIPT_DIR, "results")

SEED = 7
N_FOLDS = 5
EPOCHS = 250
BATCH_SIZE = 32
PATIENCE = 25

PROPERTY_INFO = {
    "emission": {
        "file": os.path.join(DATA_DIR, "chemfluor_processed.csv"),
        "target_col": "lambda_max",  # emission wavelength stored as lambda_max
        "display": "Fluorescence Emission (λ_em)",
        "unit": "nm",
        "has_solvent": True,
        "smiles_col": "smiles",
        "solvent_col": "solvent_smiles",
        "models": ["rf", "xgboost", "chemprop", "bigru", "chemberta"],
    },
    "log_mec": {
        "file": os.path.join(DATA_DIR, "mamede_log_mec_processed.csv"),
        "target_col": "log10_mec",
        "display": "Molar Extinction Coefficient (log₁₀ MEC)",
        "unit": "log₁₀",
        "has_solvent": True,
        "smiles_col": "smiles",
        "solvent_col": "solvent_smiles",
        "models": ["rf", "xgboost"],  # DL too expensive on 81K rows
    },
    "lipophilicity": {
        "file": os.path.join(DATA_DIR, "lipophilicity_processed.csv"),
        "target_col": "exp",
        "display": "Lipophilicity (LogP)",
        "unit": "LogP",
        "has_solvent": False,
        "smiles_col": "smiles",
        "solvent_col": None,
        "models": ["rf", "xgboost", "chemprop", "bigru", "chemberta"],
    },
    "solubility": {
        "file": os.path.join(DATA_DIR, "aqsoldb_processed.csv"),
        "target_col": "exp",
        "display": "Aqueous Solubility (logS)",
        "unit": "logS",
        "has_solvent": False,
        "smiles_col": "smiles",
        "solvent_col": None,
        "models": ["rf", "xgboost", "chemprop", "bigru", "chemberta"],
    },
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_property_data(prop_key):
    """Load and prepare data for a property. Returns dict with arrays and metadata."""
    info = PROPERTY_INFO[prop_key]
    path = info["file"]

    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        sys.exit(1)

    print(f"\n[DATA] Loading {info['display']} from {path}...")
    df = pd.read_csv(path)

    smiles = df[info["smiles_col"]].values
    y = df[info["target_col"]].values.astype(np.float64)

    if info["has_solvent"]:
        solvents = df[info["solvent_col"]].values
        combined = np.array([f"{s}!{sol}" for s, sol in zip(smiles, solvents)])
        charset = sorted(set("".join(combined) + "!E"))
        embed_len = max(len(s) for s in combined) + 5
    else:
        solvents = None
        combined = None
        charset = sorted(set("".join(smiles) + "!E"))
        embed_len = max(len(s) for s in smiles) + 5

    print(f"  Samples: {len(df)}")
    print(f"  Target ({info['unit']}): {y.min():.2f} – {y.max():.2f} "
          f"(mean {y.mean():.2f} ± {y.std():.2f})")
    if solvents is not None:
        print(f"  Unique solvents: {len(set(solvents))}")
    print(f"  Charset: {len(charset)} tokens, max seq len: {embed_len}")

    return {
        "smiles": smiles,
        "solvents": solvents,
        "combined": combined,
        "y": y,
        "charset": charset,
        "embed_len": embed_len,
        "has_solvent": info["has_solvent"],
    }


# ─── Fold creation ────────────────────────────────────────────────────────────

def create_folds(data, prop_key):
    """Create 5-fold stratified splits. Uses solvent bins if available, else target bins."""
    n = len(data["y"])

    if data["has_solvent"] and data["solvents"] is not None:
        print("  Creating solvent-stratified folds...")
        folds = create_stratified_folds(n, data["solvents"], n_folds=N_FOLDS, seed=SEED)
    else:
        # No solvent: stratify by target value bins
        print("  Creating target-stratified folds (no solvent)...")
        y = data["y"]
        bins = pd.qcut(y, q=20, labels=False, duplicates="drop").astype(str)
        folds = create_stratified_folds(n, bins, n_folds=N_FOLDS, seed=SEED)

    return folds


# ─── Results I/O ──────────────────────────────────────────────────────────────

def get_results_dir(prop_key):
    d = os.path.join(RESULTS_BASE, prop_key)
    os.makedirs(d, exist_ok=True)
    return d


def save_fold_results(results_dir, prefix, fold_idx, metrics, y_pred, y_test, test_indices):
    tag = f"{prefix}_fold{fold_idx}"
    with open(os.path.join(results_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, f"{tag}_predictions.npy"), y_pred)
    np.save(os.path.join(results_dir, f"{tag}_y_test.npy"), y_test)
    np.save(os.path.join(results_dir, f"{tag}_test_indices.npy"), test_indices)


def load_fold_results(results_dir, prefix, fold_idx):
    tag = f"{prefix}_fold{fold_idx}"
    paths = [
        os.path.join(results_dir, f"{tag}_metrics.json"),
        os.path.join(results_dir, f"{tag}_predictions.npy"),
        os.path.join(results_dir, f"{tag}_y_test.npy"),
        os.path.join(results_dir, f"{tag}_test_indices.npy"),
    ]
    if not all(os.path.exists(p) for p in paths):
        return None
    with open(paths[0]) as f:
        metrics = json.load(f)
    y_pred = np.load(paths[1])
    y_test = np.load(paths[2])
    test_indices = np.load(paths[3])
    return metrics, y_pred, y_test, test_indices


def aggregate_results(results_dir, prefix, n_folds=N_FOLDS):
    """Aggregate fold results. Returns aggregate dict or None."""
    all_metrics = []
    all_y_pred, all_y_test, all_idx = [], [], []

    for i in range(n_folds):
        result = load_fold_results(results_dir, prefix, i)
        if result is None:
            print(f"  [WARN] Missing fold {i} for {prefix}")
            continue
        metrics, y_pred, y_test, test_idx = result
        all_metrics.append(metrics)
        all_y_pred.append(y_pred)
        all_y_test.append(y_test)
        all_idx.append(test_idx)

    n_found = len(all_metrics)
    if n_found == 0:
        return None

    metric_keys = ["RMSE", "MAE", "R2", "Pearson_r"]
    agg = {"n_folds_found": n_found}
    for mk in metric_keys:
        vals = [m[mk] for m in all_metrics]
        agg[f"{mk}_mean"] = round(np.mean(vals), 4)
        agg[f"{mk}_std"] = round(np.std(vals), 4)
        agg[f"{mk}_values"] = vals

    # Pool predictions
    pooled_pred = np.concatenate(all_y_pred)
    pooled_test = np.concatenate(all_y_test)
    pooled_idx = np.concatenate(all_idx)
    sort_order = np.argsort(pooled_idx)

    np.savez(os.path.join(results_dir, f"{prefix}_cv_pooled.npz"),
             y_pred=pooled_pred[sort_order],
             y_test=pooled_test[sort_order],
             indices=pooled_idx[sort_order])

    pooled_metrics = compute_metrics(pooled_test, pooled_pred)
    agg["pooled_metrics"] = pooled_metrics

    agg_path = os.path.join(results_dir, f"{prefix}_cv_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"  {prefix}: {n_found}/{n_folds} folds aggregated")
    for mk in metric_keys:
        print(f"    {mk}: {agg[f'{mk}_mean']:.4f} ± {agg[f'{mk}_std']:.4f}")

    return agg


# ─── Fingerprint computation ─────────────────────────────────────────────────

def compute_fps(data):
    """Compute Morgan FPs. Returns (fp_array, valid_mask)."""
    print("  Computing Morgan fingerprints (solute)...")
    fp_solute, mask_solute = compute_morgan_fps(list(data["smiles"]))

    if data["has_solvent"] and data["solvents"] is not None:
        print("  Computing Morgan fingerprints (solvent)...")
        fp_solvent, mask_solvent = compute_morgan_fps(list(data["solvents"]))
        fp_all = np.hstack([fp_solute, fp_solvent])
        valid_mask = mask_solute & mask_solvent
    else:
        fp_all = fp_solute
        valid_mask = mask_solute

    n_valid = valid_mask.sum()
    print(f"  FP shape: {fp_all.shape}, valid: {n_valid}/{len(valid_mask)}")
    return fp_all, valid_mask


# ─── RF training ──────────────────────────────────────────────────────────────

def run_rf_fold(data, folds, fold_idx, results_dir, fp_all, fp_valid_mask, prefix="rf"):
    """Train RF on one fold. Combines train+val (RF has no early stopping)."""
    existing = load_fold_results(results_dir, prefix, fold_idx)
    if existing is not None:
        print(f"\n  [RF Fold {fold_idx}] Already done (RMSE={existing[0]['RMSE']}). Skipping.")
        return existing[0]

    train_idx, val_idx, test_idx = folds[fold_idx]
    train_val_idx = np.concatenate([train_idx, val_idx])

    # Apply valid mask
    mask_train = fp_valid_mask[train_val_idx]
    mask_test = fp_valid_mask[test_idx]

    X_train = fp_all[train_val_idx][mask_train]
    y_train = data["y"][train_val_idx][mask_train]
    X_test = fp_all[test_idx][mask_test]
    y_test = data["y"][test_idx][mask_test]

    print(f"\n  [RF Fold {fold_idx}] Train: {len(X_train)}, Test: {len(X_test)}")

    # Limit parallelism for large datasets to avoid OOM (each worker copies data)
    n_jobs = 4 if len(X_train) > 20000 else -1

    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=500, max_depth=None, n_jobs=n_jobs, random_state=SEED
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print(f"  [RF Fold {fold_idx}] Done in {elapsed:.0f}s — RMSE={metrics['RMSE']:.2f} "
          f"MAE={metrics['MAE']:.2f} R²={metrics['R2']:.4f}")

    # Save
    import joblib
    save_fold_results(results_dir, prefix, fold_idx, metrics, y_pred, y_test, test_idx[mask_test])
    model_path = os.path.join(results_dir, f"{prefix}_fold{fold_idx}_model.joblib")
    joblib.dump(model, model_path)

    del model
    gc.collect()
    return metrics


# ─── XGBoost training ────────────────────────────────────────────────────

def run_xgboost_fold(data, folds, fold_idx, results_dir, fp_all, fp_valid_mask, prefix="xgboost"):
    """Train XGBoost on one fold. Combines train+val (no early stopping on val)."""
    existing = load_fold_results(results_dir, prefix, fold_idx)
    if existing is not None:
        print(f"\n  [XGBoost Fold {fold_idx}] Already done (RMSE={existing[0]['RMSE']}). Skipping.")
        return existing[0]

    import xgboost as xgb

    train_idx, val_idx, test_idx = folds[fold_idx]
    train_val_idx = np.concatenate([train_idx, val_idx])

    # Apply valid mask
    mask_train = fp_valid_mask[train_val_idx]
    mask_test = fp_valid_mask[test_idx]

    X_train = fp_all[train_val_idx][mask_train]
    y_train = data["y"][train_val_idx][mask_train]
    X_test = fp_all[test_idx][mask_test]
    y_test = data["y"][test_idx][mask_test]

    # Use GPU for large datasets (faster + avoids CPU RAM pressure)
    use_gpu = len(X_train) > 20000
    device_str = "cuda" if use_gpu else "cpu"
    n_jobs = 1 if use_gpu else -1

    print(f"\n  [XGBoost Fold {fold_idx}] Train: {len(X_train)}, Test: {len(X_test)}, device={device_str}")

    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        tree_method="hist", device=device_str, random_state=SEED, n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print(f"  [XGBoost Fold {fold_idx}] Done in {elapsed:.0f}s — RMSE={metrics['RMSE']:.2f} "
          f"MAE={metrics['MAE']:.2f} R²={metrics['R2']:.4f}")

    # Save
    import joblib
    save_fold_results(results_dir, prefix, fold_idx, metrics, y_pred, y_test, test_idx[mask_test])
    model_path = os.path.join(results_dir, f"{prefix}_fold{fold_idx}_model.joblib")
    joblib.dump(model, model_path)

    del model
    gc.collect()
    return metrics


# ─── BiGRU training ───────────────────────────────────────────────────────────

def run_bigru_fold(data, folds, fold_idx, results_dir, prefix="bigru"):
    """Train BiGRU on one fold with val for early stopping."""
    existing = load_fold_results(results_dir, prefix, fold_idx)
    if existing is not None:
        print(f"\n  [BiGRU Fold {fold_idx}] Already done (RMSE={existing[0]['RMSE']}). Skipping.")
        return existing[0]

    import tensorflow as tf
    tf.random.set_seed(SEED + fold_idx)

    train_idx, val_idx, test_idx = folds[fold_idx]
    charset = data["charset"]
    embed_len = data["embed_len"]
    c2i = build_char_maps(charset)

    # Select SMILES source
    if data["has_solvent"] and data["combined"] is not None:
        smiles_arr = data["combined"]
    else:
        smiles_arr = data["smiles"]

    print(f"\n  [BiGRU Fold {fold_idx}] Vectorizing {len(train_idx)} train + "
          f"{len(val_idx)} val + {len(test_idx)} test...")
    X_train = vectorize_smiles(smiles_arr[train_idx], c2i, embed_len)
    X_val = vectorize_smiles(smiles_arr[val_idx], c2i, embed_len)
    X_test = vectorize_smiles(smiles_arr[test_idx], c2i, embed_len)

    y_train = data["y"][train_idx]
    y_val = data["y"][val_idx]
    y_test = data["y"][test_idx]

    num_words = len(charset)
    input_length = embed_len - 1
    fold_name = f"{prefix}_fold{fold_idx}"

    y_pred, metrics, model = train_dl_model(
        "bigru", X_train, X_test,
        y_train.reshape(-1, 1), y_test.reshape(-1, 1),
        num_words=num_words, input_length=input_length, name=fold_name,
        results_dir=results_dir, seed=SEED, epochs=EPOCHS,
        batch_size=BATCH_SIZE, patience=PATIENCE,
        X_val=X_val, y_val=y_val.reshape(-1, 1),
    )

    save_fold_results(results_dir, prefix, fold_idx, metrics, y_pred, y_test, test_idx)
    model_path = os.path.join(results_dir, f"{prefix}_fold{fold_idx}_model.keras")
    model.save(model_path)
    print(f"  >> Model saved to {model_path}")

    del model, X_train, X_val, X_test
    tf.keras.backend.clear_session()
    gc.collect()
    return metrics


# ─── ChemBERTa training ──────────────────────────────────────────────────────

def run_chemberta_fold(data, folds, fold_idx, results_dir, prefix="chemberta"):
    """Train ChemBERTa on one fold. Imports from run_chemberta.py."""
    existing = load_fold_results(results_dir, prefix, fold_idx)
    if existing is not None:
        print(f"\n  [ChemBERTa Fold {fold_idx}] Already done (RMSE={existing[0]['RMSE']}). Skipping.")
        return existing[0]

    # Import the full ChemBERTa training function
    sys.path.insert(0, SCRIPT_DIR)
    from run_chemberta import train_chemberta, gpu_cooldown

    train_idx, val_idx, test_idx = folds[fold_idx]

    # ChemBERTa uses raw SMILES (its own tokenizer handles encoding)
    if data["has_solvent"] and data["combined"] is not None:
        smiles_arr = data["combined"]
    else:
        smiles_arr = data["smiles"]

    train_smiles = list(smiles_arr[train_idx])
    val_smiles = list(smiles_arr[val_idx])
    test_smiles = list(smiles_arr[test_idx])

    y_train = data["y"][train_idx].astype(np.float32)
    y_val = data["y"][val_idx].astype(np.float32)
    y_test = data["y"][test_idx].astype(np.float32)

    fold_name = f"{prefix}_fold{fold_idx}"
    print(f"\n  [ChemBERTa Fold {fold_idx}] Train: {len(train_smiles)}, "
          f"Val: {len(val_smiles)}, Test: {len(test_smiles)}")

    y_pred, metrics = train_chemberta(
        train_smiles, val_smiles, test_smiles,
        y_train, y_val, y_test,
        name=fold_name, results_dir=results_dir,
        test_idx=test_idx,
    )

    save_fold_results(results_dir, prefix, fold_idx, metrics, y_pred, y_test, test_idx)

    gpu_cooldown(seconds=10)
    gc.collect()
    return metrics


# ─── Chemprop training ──────────────────────────────────────────────────────

def run_chemprop_fold(data, folds, fold_idx, results_dir, prefix="chemprop"):
    """Train Chemprop D-MPNN on one fold. Single or multicomponent based on has_solvent."""
    existing = load_fold_results(results_dir, prefix, fold_idx)
    if existing is not None:
        print(f"\n  [Chemprop Fold {fold_idx}] Already done (RMSE={existing[0]['RMSE']}). Skipping.")
        return existing[0]

    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from chemprop import data as cpdata, featurizers, nn as cpnn
    from chemprop.models import multi as cpmulti, MPNN as SingleMPNN
    from chemprop.nn.metrics import RMSE as ChempropRMSE, MAE as ChempropMAE
    from chemprop.nn.transforms import UnscaleTransform

    # Import gpu_cooldown from run_chemprop
    sys.path.insert(0, SCRIPT_DIR)
    from run_chemprop import gpu_cooldown, ProgressCallback

    # Chemprop hyperparameters (same as primary benchmark)
    CP_MAX_EPOCHS = 100
    CP_PATIENCE = 15
    CP_BATCH_SIZE = 64
    CP_D_H = 300
    CP_DEPTH = 3
    CP_WARMUP = 2
    CP_INIT_LR = 1e-4
    CP_MAX_LR = 1e-3
    CP_FINAL_LR = 1e-4

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(SEED + fold_idx, workers=True)

    train_idx, val_idx, test_idx = folds[fold_idx]
    solute_only = not data["has_solvent"]

    print(f"\n  [Chemprop Fold {fold_idx}] Train: {len(train_idx)}, "
          f"Val: {len(val_idx)}, Test: {len(test_idx)}, "
          f"{'single-component' if solute_only else 'multicomponent'}")

    # Create MoleculeDatapoints
    from rdkit import Chem

    def make_datapoints(indices, with_targets=True):
        solute_dps, solvent_dps = [], []
        valid_idx = []
        for i in indices:
            smi = str(data["smiles"][i])
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            yval = np.array([data["y"][i]]) if with_targets else None
            solute_dps.append(cpdata.MoleculeDatapoint.from_smi(smi, y=yval))
            if not solute_only:
                sol_smi = str(data["solvents"][i])
                sol_mol = Chem.MolFromSmiles(sol_smi)
                if sol_mol is None:
                    solute_dps.pop()
                    continue
                solvent_dps.append(cpdata.MoleculeDatapoint.from_smi(sol_smi))
            valid_idx.append(i)
        return solute_dps, solvent_dps, np.array(valid_idx)

    solute_train, solvent_train, train_valid = make_datapoints(train_idx)
    solute_val, solvent_val, val_valid = make_datapoints(val_idx)
    solute_test, solvent_test, test_valid = make_datapoints(test_idx)

    n_skipped = (len(train_idx) + len(val_idx) + len(test_idx)
                 - len(train_valid) - len(val_valid) - len(test_valid))
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} invalid SMILES entries")

    # Build datasets and loaders
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    if solute_only:
        train_dset = cpdata.MoleculeDataset(solute_train, feat)
        val_dset = cpdata.MoleculeDataset(solute_val, feat)
        test_dset = cpdata.MoleculeDataset(solute_test, feat)
        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)
    else:
        train_dsets = [cpdata.MoleculeDataset(solute_train, feat),
                       cpdata.MoleculeDataset(solvent_train, feat)]
        val_dsets = [cpdata.MoleculeDataset(solute_val, feat),
                     cpdata.MoleculeDataset(solvent_val, feat)]
        test_dsets = [cpdata.MoleculeDataset(solute_test, feat),
                      cpdata.MoleculeDataset(solvent_test, feat)]
        train_dset = cpdata.MulticomponentDataset(train_dsets)
        val_dset = cpdata.MulticomponentDataset(val_dsets)
        test_dset = cpdata.MulticomponentDataset(test_dsets)
        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

    train_loader = cpdata.build_dataloader(train_dset, batch_size=CP_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = cpdata.build_dataloader(val_dset, batch_size=CP_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = cpdata.build_dataloader(test_dset, batch_size=CP_BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model
    agg = cpnn.MeanAggregation()
    output_transform = UnscaleTransform.from_standard_scaler(scaler)

    if solute_only:
        mp = cpnn.BondMessagePassing(d_h=CP_D_H, depth=CP_DEPTH)
        ffn = cpnn.RegressionFFN(input_dim=mp.output_dim, output_transform=output_transform)
        model = SingleMPNN(
            mp, agg, ffn,
            metrics=[ChempropRMSE(), ChempropMAE()],
            warmup_epochs=CP_WARMUP, init_lr=CP_INIT_LR,
            max_lr=CP_MAX_LR, final_lr=CP_FINAL_LR,
        )
    else:
        mcmp = cpnn.MulticomponentMessagePassing(
            blocks=[cpnn.BondMessagePassing(d_h=CP_D_H, depth=CP_DEPTH) for _ in range(2)],
            n_components=2,
        )
        ffn = cpnn.RegressionFFN(input_dim=mcmp.output_dim, output_transform=output_transform)
        model = cpmulti.MulticomponentMPNN(
            mcmp, agg, ffn,
            metrics=[ChempropRMSE(), ChempropMAE()],
            warmup_epochs=CP_WARMUP, init_lr=CP_INIT_LR,
            max_lr=CP_MAX_LR, final_lr=CP_FINAL_LR,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  >> Chemprop: {n_params:,} params "
          f"({'single' if solute_only else 'multi'}-component)")

    # Callbacks
    fold_name = f"{prefix}_fold{fold_idx}"
    checkpoint_best = ModelCheckpoint(
        dirpath=results_dir, filename=f"{fold_name}_best",
        monitor="val_loss", save_top_k=1, mode="min",
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=CP_PATIENCE, mode="min", min_delta=1e-5,
    )
    progress = ProgressCallback(name=fold_name, report_every=5)

    # Check for resume checkpoint
    ckpt_resume = None
    for suffix in ["_latest.ckpt", "_best.ckpt"]:
        candidate = os.path.join(results_dir, f"{fold_name}{suffix}")
        if os.path.exists(candidate):
            ckpt_resume = candidate
            print(f"  >> Resuming from {os.path.basename(candidate)}")
            break

    # Train
    trainer = pl.Trainer(
        max_epochs=CP_MAX_EPOCHS, accelerator="auto", devices=1,
        callbacks=[checkpoint_best, early_stop, progress],
        enable_progress_bar=False, logger=False,
        enable_model_summary=(fold_idx == 0),
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_resume)
    elapsed = time.time() - t0
    print(f"  [{fold_name}] Training complete in {elapsed:.0f}s "
          f"({trainer.current_epoch + 1} epochs)")

    # Load best model and predict
    best_ckpt = checkpoint_best.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        if solute_only:
            best_model = SingleMPNN.load_from_checkpoint(best_ckpt)
        else:
            best_model = cpmulti.MulticomponentMPNN.load_from_checkpoint(best_ckpt)
    else:
        best_model = model

    preds = trainer.predict(best_model, test_loader)
    y_pred = torch.cat(preds, dim=0).squeeze(-1).numpy()
    y_test = data["y"][test_valid]

    metrics = compute_metrics(y_test, y_pred)
    print(f"  >> {fold_name} — RMSE={metrics['RMSE']:.2f} MAE={metrics['MAE']:.2f} "
          f"R²={metrics['R2']:.4f} r={metrics['Pearson_r']:.4f}")

    save_fold_results(results_dir, prefix, fold_idx, metrics, y_pred, y_test, test_valid)

    # Cleanup
    del model, best_model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    gpu_cooldown(seconds=10)

    return metrics


# ─── Main runners ─────────────────────────────────────────────────────────────

def run_property_model(prop_key, model_key, fold_idx=None):
    """Run a model on a property for specified fold(s)."""
    info = PROPERTY_INFO[prop_key]
    if model_key not in info["models"]:
        print(f"  ERROR: {model_key} not supported for {prop_key}. "
              f"Supported: {info['models']}")
        sys.exit(1)

    data = load_property_data(prop_key)
    results_dir = get_results_dir(prop_key)
    folds = create_folds(data, prop_key)

    # Save fold indices
    fold_path = os.path.join(results_dir, "fold_indices.npz")
    if not os.path.exists(fold_path):
        save_dict = {}
        for i, (tr, va, te) in enumerate(folds):
            save_dict[f"train_{i}"] = tr
            save_dict[f"val_{i}"] = va
            save_dict[f"test_{i}"] = te
        np.savez(fold_path, **save_dict)

    fold_range = [fold_idx] if fold_idx is not None else list(range(N_FOLDS))
    prefix = model_key

    print(f"\n{'='*70}")
    print(f"  {info['display']} — {model_key.upper()} — "
          f"Folds: {fold_range}")
    print(f"{'='*70}")

    if model_key == "rf":
        fp_all, fp_valid_mask = compute_fps(data)
        for fi in fold_range:
            run_rf_fold(data, folds, fi, results_dir, fp_all, fp_valid_mask, prefix)
    elif model_key == "xgboost":
        fp_all, fp_valid_mask = compute_fps(data)
        for fi in fold_range:
            run_xgboost_fold(data, folds, fi, results_dir, fp_all, fp_valid_mask, prefix)
    elif model_key == "chemprop":
        for fi in fold_range:
            run_chemprop_fold(data, folds, fi, results_dir, prefix)
    elif model_key == "bigru":
        verify_gpu_available()
        for fi in fold_range:
            run_bigru_fold(data, folds, fi, results_dir, prefix)
    elif model_key == "chemberta":
        for fi in fold_range:
            run_chemberta_fold(data, folds, fi, results_dir, prefix)

    # Aggregate
    print(f"\n  Aggregating {model_key} results...")
    agg = aggregate_results(results_dir, prefix)
    return agg


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary():
    """Print comparison table across all properties and models."""
    print(f"\n{'='*90}")
    print(f"  MULTI-PROPERTY BENCHMARK SUMMARY")
    print(f"{'='*90}")

    header = f"{'Property':<35} {'Model':<12} {'RMSE':<18} {'MAE':<18} {'R²':<18}"
    print(header)
    print("─" * 90)

    for prop_key, info in PROPERTY_INFO.items():
        results_dir = get_results_dir(prop_key)
        for model_key in info["models"]:
            agg_path = os.path.join(results_dir, f"{model_key}_cv_aggregate.json")
            if not os.path.exists(agg_path):
                print(f"  {info['display']:<33} {model_key.upper():<12} {'(not run)'}")
                continue
            with open(agg_path) as f:
                agg = json.load(f)
            n = agg["n_folds_found"]
            rmse = f"{agg['RMSE_mean']:.2f} ± {agg['RMSE_std']:.2f}"
            mae = f"{agg['MAE_mean']:.2f} ± {agg['MAE_std']:.2f}"
            r2 = f"{agg['R2_mean']:.4f} ± {agg['R2_std']:.4f}"
            folds_str = f"({n}/{N_FOLDS})" if n < N_FOLDS else ""
            print(f"  {info['display']:<33} {model_key.upper():<12} {rmse:<18} {mae:<18} {r2:<18} {folds_str}")
        print()

    # Also print UV results for reference
    print("  --- UV λ_max (from primary benchmark, for comparison) ---")
    for prefix, label in [
        ("rf_tuned", "RF TUNED"),
        ("xgboost_v2", "XGBoost"),
        ("chemprop_v2", "Chemprop"),
        ("bigru_solvent_v2", "BiGRU+Solvent"),
        ("chemberta", "ChemBERTa"),
    ]:
        agg_path = os.path.join(RESULTS_BASE, f"{prefix}_cv_aggregate.json")
        if os.path.exists(agg_path):
            with open(agg_path) as f:
                agg = json.load(f)
            rmse = f"{agg['RMSE_mean']:.2f} ± {agg['RMSE_std']:.2f}"
            mae = f"{agg['MAE_mean']:.2f} ± {agg['MAE_std']:.2f}"
            r2 = f"{agg['R2_mean']:.4f} ± {agg['R2_std']:.4f}"
            print(f"  {'UV Absorption (λ_max)':<33} {label:<12} {rmse:<18} {mae:<18} {r2:<18}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-property benchmark runner")
    parser.add_argument("--property", choices=list(PROPERTY_INFO.keys()),
                        help="Property to benchmark")
    parser.add_argument("--model", choices=["rf", "xgboost", "chemprop", "bigru", "chemberta"],
                        help="Model to train")
    parser.add_argument("--fold", type=int, choices=range(N_FOLDS), default=None,
                        help="Fold index (0-4). Omit for all folds.")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary table of all completed experiments")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if not args.property or not args.model:
        parser.error("--property and --model are required (or use --summary)")

    run_property_model(args.property, args.model, args.fold)


if __name__ == "__main__":
    main()
