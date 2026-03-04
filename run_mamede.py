#!/usr/bin/env python3
"""
Mamede classification experiments: binary photosafety prediction.

Five experiment modes:
  1. bigru_cls          — BiGRU binary classifier (no solvent), 74,783 molecules
  2. bigru_cls_solvent  — BiGRU binary classifier (solvent-concatenated SMILES), ~69,400 molecules
  3. hybrid_cls         — Hybrid BiGRU + Morgan FP classifier, 74,783 molecules
  4. regression         — BiGRU regression then threshold (original approach)
  5. rf_cls             — RF + Morgan FP 1024-bit (reproducing Mamede's exact method)

Reference: Mamede et al. (2021) RF + Morgan FP → Acc=0.89, Sens=0.90, Spec=0.88

Usage:
    python3 run_mamede.py --model bigru_cls
    python3 run_mamede.py --model bigru_cls_solvent
    python3 run_mamede.py --model hybrid_cls
    python3 run_mamede.py --model regression
    python3 run_mamede.py --model rf_cls
    python3 run_mamede.py --summary              # print comparison table
"""

import argparse
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    f1_score,
    roc_auc_score,
    accuracy_score,
)

from paper1_new_cl.models import (
    build_char_maps,
    vectorize_smiles,
    compute_metrics,
    train_dl_model,
    create_hybrid_model,
    compute_morgan_fps,
    verify_gpu_available,
    get_tqdm_callback,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSIFICATION_FILE = os.path.join(DATA_DIR, "mamede_classification_dataset.csv")
REGRESSION_FILE = os.path.join(DATA_DIR, "mamede_regression_dataset.csv")

SEED = 7
EPOCHS = 250
BATCH_SIZE = 32
PATIENCE = 25

# ICH S10 photosafety thresholds
POS_LO = 290.0
POS_HI = 700.0


# ─── Classification helpers ──────────────────────────────────────────────────

def classify_photosafety(lmax, lo=POS_LO, hi=POS_HI):
    """POS (1) if λ_max ∈ [lo, hi], else NEG (0)."""
    return ((lmax >= lo) & (lmax <= hi)).astype(int)


def photosafety_score(y_pred_lmax, lo=POS_LO, hi=POS_HI):
    """Continuous score for ROC-AUC: negative distance from interval center."""
    center = (lo + hi) / 2.0
    return -np.abs(y_pred_lmax - center)


def compute_classification_metrics(y_true_bin, y_pred_bin, y_pred_score):
    """Compute classification metrics from binary labels + continuous score."""
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

    n_classes = len(np.unique(y_true_bin))
    auc = roc_auc_score(y_true_bin, y_pred_score) if n_classes == 2 else float("nan")

    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "Accuracy": round(acc, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
        "ROC-AUC": round(auc, 4),
        "N": int(tp + fp + tn + fn),
        "N_POS": int(tp + fn),
        "N_NEG": int(tn + fp),
    }


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_classification_data():
    """Load all 74,783 molecules with binary labels (no solvent).

    Returns DataFrame with columns: smiles, class_bin, xrn
    """
    print("\n  Loading classification dataset...")
    cls_df = pd.read_csv(CLASSIFICATION_FILE)
    cls_df["class_bin"] = (cls_df["class_label"] == "POS").astype(int)
    print(f"    {len(cls_df)} molecules — "
          f"POS: {cls_df['class_bin'].sum()}, NEG: {(1 - cls_df['class_bin']).sum()}")
    return cls_df


def load_classification_data_with_solvent():
    """Load classification molecules that have solvent info.

    Merges classification + regression datasets, picks primary solvent per molecule
    (highest MEC with non-null solvent_smiles). Builds combined SMILES.

    Returns DataFrame with columns: smiles, class_bin, xrn, solvent_smiles, combined_smiles
    """
    print("\n  Loading classification + regression datasets for solvent merge...")
    cls_df = pd.read_csv(CLASSIFICATION_FILE)
    reg_df = pd.read_csv(REGRESSION_FILE)

    cls_df["xrn"] = cls_df["xrn"].astype(str)
    reg_df["xrn"] = reg_df["xrn"].astype(str)

    # Pick the primary peak with non-null solvent per molecule (highest MEC)
    reg_with_solvent = reg_df[reg_df["solvent_smiles"].notna()].copy()
    reg_per_mol = (
        reg_with_solvent
        .sort_values(["mec", "lambda_max"], ascending=[False, False])
        .drop_duplicates(subset=["xrn"], keep="first")
        [["xrn", "solvent_smiles"]]
    )

    # Merge
    df = cls_df.merge(reg_per_mol, on="xrn", how="inner")
    df["class_bin"] = (df["class_label"] == "POS").astype(int)
    df["combined_smiles"] = df["smiles"] + "!" + df["solvent_smiles"]

    print(f"    {len(df)} molecules with solvent — "
          f"POS: {df['class_bin'].sum()}, NEG: {(1 - df['class_bin']).sum()}")
    print(f"    Max combined SMILES length: {df['combined_smiles'].str.len().max()}")
    return df


def load_regression_data():
    """Load merged classification + regression data for regression experiment.

    Returns DataFrame with columns: smiles, class_bin, xrn, lambda_max
    Only molecules that have a λ_max value (drops ~4,785 NEG without UV data).
    """
    print("\n  Loading data for regression experiment...")
    cls_df = pd.read_csv(CLASSIFICATION_FILE)
    reg_df = pd.read_csv(REGRESSION_FILE)

    cls_df["xrn"] = cls_df["xrn"].astype(str)
    reg_df["xrn"] = reg_df["xrn"].astype(str)

    # Take the primary peak per molecule (highest MEC, ties broken by longest λ)
    reg_per_mol = (
        reg_df.sort_values(["mec", "lambda_max"], ascending=[False, False])
        .drop_duplicates(subset=["xrn"], keep="first")
        [["xrn", "lambda_max"]]
    )

    # Merge
    df = cls_df.merge(reg_per_mol, on="xrn", how="left")
    df = df.dropna(subset=["lambda_max"]).reset_index(drop=True)
    df["class_bin"] = (df["class_label"] == "POS").astype(int)

    print(f"    {len(df)} molecules with λ_max — "
          f"POS: {df['class_bin'].sum()}, NEG: {(1 - df['class_bin']).sum()}")
    print(f"    λ_max range: {df['lambda_max'].min():.1f} - {df['lambda_max'].max():.1f} nm")
    return df


# ─── Splitting ────────────────────────────────────────────────────────────────

def create_classification_split(n_samples, class_bin, split_name):
    """Create 80/10/10 stratified split. Caches to disk.

    Returns (train_idx, val_idx, test_idx).
    """
    split_path = os.path.join(RESULTS_DIR, f"mamede_cls_split_{split_name}.npz")

    if os.path.exists(split_path):
        data = np.load(split_path)
        train_idx, val_idx, test_idx = data["train"], data["val"], data["test"]
        print(f"  Loaded cached split from {split_path}")
        print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        return train_idx, val_idx, test_idx

    # 80% train vs 20% held-out
    all_idx = np.arange(n_samples)
    train_idx, held_idx = train_test_split(
        all_idx, test_size=0.20, random_state=SEED, stratify=class_bin
    )
    # Split held-out 50/50 into val and test (each 10%)
    held_class = class_bin[held_idx]
    val_idx_rel, test_idx_rel = train_test_split(
        np.arange(len(held_idx)), test_size=0.5, random_state=SEED,
        stratify=held_class
    )
    val_idx = held_idx[val_idx_rel]
    test_idx = held_idx[test_idx_rel]

    np.savez(split_path, train=train_idx, val=val_idx, test=test_idx)
    print(f"  Created and saved split to {split_path}")
    print(f"    Train: {len(train_idx)} (POS: {class_bin[train_idx].sum()}, "
          f"NEG: {(1-class_bin[train_idx]).sum()})")
    print(f"    Val:   {len(val_idx)} (POS: {class_bin[val_idx].sum()}, "
          f"NEG: {(1-class_bin[val_idx]).sum()})")
    print(f"    Test:  {len(test_idx)} (POS: {class_bin[test_idx].sum()}, "
          f"NEG: {(1-class_bin[test_idx]).sum()})")
    return train_idx, val_idx, test_idx


# ─── Output helpers ──────────────────────────────────────────────────────────

def save_experiment_results(prefix, cls_metrics, model, y_pred_proba,
                            y_true_bin, y_pred_bin, test_idx,
                            config, reg_metrics=None):
    """Save metrics JSON, predictions NPZ, and model for an experiment."""
    all_metrics = {
        "classification": cls_metrics,
        "config": config,
    }
    if reg_metrics is not None:
        all_metrics["regression"] = reg_metrics

    metrics_path = os.path.join(RESULTS_DIR, f"{prefix}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    pred_path = os.path.join(RESULTS_DIR, f"{prefix}_predictions.npz")
    np.savez(pred_path, y_pred_proba=y_pred_proba, y_true_bin=y_true_bin,
             y_pred_bin=y_pred_bin, test_idx=test_idx)
    print(f"  Saved: {pred_path}")

    model_path = os.path.join(RESULTS_DIR, f"{prefix}_model.keras")
    model.save(model_path)
    print(f"  Saved: {model_path}")


def print_metrics_table(cls_metrics, model_name):
    """Print classification metrics for a single model."""
    print(f"\n  {'─' * 50}")
    print(f"  Classification metrics — {model_name} (N={cls_metrics['N']}):")
    print(f"    Sensitivity: {cls_metrics['Sensitivity']:.4f}")
    print(f"    Specificity: {cls_metrics['Specificity']:.4f}")
    print(f"    Accuracy:    {cls_metrics['Accuracy']:.4f}")
    print(f"    MCC:         {cls_metrics['MCC']:.4f}")
    print(f"    F1:          {cls_metrics['F1']:.4f}")
    print(f"    ROC-AUC:     {cls_metrics['ROC-AUC']:.4f}")
    print(f"    TP={cls_metrics['TP']} FP={cls_metrics['FP']} "
          f"FN={cls_metrics['FN']} TN={cls_metrics['TN']}")


def print_comparison_table():
    """Print comparison table from all saved experiment results and mamede_comparison.csv."""
    # Mamede reference
    rows = [{
        "Model": "Mamede RF (2021)",
        "N_test": "~998",
        "Sensitivity": 0.90,
        "Specificity": 0.88,
        "Accuracy": 0.89,
        "MCC": None,
        "F1": None,
        "ROC-AUC": None,
    }]

    # Load experiment results
    experiments = [
        ("mamede_rf_cls", "RF + Morgan FP 1024-bit (ours)"),
        ("mamede_bigru_cls", "BiGRU classifier (no solvent)"),
        ("mamede_bigru_cls_solvent", "BiGRU classifier (solvent)"),
        ("mamede_hybrid_cls", "Hybrid BiGRU+FP classifier"),
        ("mamede_nosolvent", "BiGRU regression → threshold"),
    ]

    for prefix, name in experiments:
        path = os.path.join(RESULTS_DIR, f"{prefix}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            cls = data.get("classification", {})
            rows.append({
                "Model": name,
                "N_test": cls.get("N", "—"),
                "Sensitivity": cls.get("Sensitivity"),
                "Specificity": cls.get("Specificity"),
                "Accuracy": cls.get("Accuracy"),
                "MCC": cls.get("MCC"),
                "F1": cls.get("F1"),
                "ROC-AUC": cls.get("ROC-AUC"),
            })

    if len(rows) <= 1:
        print("  No experiment results found yet.")
        return

    # Print table
    print(f"\n  {'═' * 90}")
    print(f"  MAMEDE PHOTOSAFETY CLASSIFICATION — COMPARISON TABLE")
    print(f"  {'═' * 90}")
    header = (f"  {'Model':<35} {'N_test':>7} {'Sens':>7} {'Spec':>7} "
              f"{'Acc':>7} {'MCC':>7} {'F1':>7} {'AUC':>7}")
    print(header)
    print(f"  {'─' * 87}")
    for row in rows:
        n = str(row["N_test"]) if row["N_test"] is not None else "—"
        sens = f"{row['Sensitivity']:.4f}" if row["Sensitivity"] is not None else "—"
        spec = f"{row['Specificity']:.4f}" if row["Specificity"] is not None else "—"
        acc = f"{row['Accuracy']:.4f}" if row["Accuracy"] is not None else "—"
        mcc = f"{row['MCC']:.4f}" if row["MCC"] is not None else "—"
        f1 = f"{row['F1']:.4f}" if row["F1"] is not None else "—"
        auc = f"{row['ROC-AUC']:.4f}" if row["ROC-AUC"] is not None else "—"
        print(f"  {row['Model']:<35} {n:>7} {sens:>7} {spec:>7} "
              f"{acc:>7} {mcc:>7} {f1:>7} {auc:>7}")
    print(f"  {'═' * 90}")

    # Save comparison CSV
    comp_df = pd.DataFrame(rows)
    comp_path = os.path.join(RESULTS_DIR, "mamede_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\n  Saved: {comp_path}")


# ─── Experiment 1: BiGRU binary classifier (no solvent) ─────────────────────

def run_bigru_cls(args):
    """Train BiGRU as binary classifier on bare SMILES (no solvent)."""
    print("=" * 60)
    print("  EXPERIMENT 1: BiGRU BINARY CLASSIFIER (no solvent)")
    print("  74,783 molecules, sigmoid + BCE")
    print("=" * 60)

    # Load data
    df = load_classification_data()
    smiles = df["smiles"].values
    class_bin = df["class_bin"].values

    # Split
    print("\n  Creating 80/10/10 stratified split...")
    train_idx, val_idx, test_idx = create_classification_split(
        len(df), class_bin, "74783"
    )

    # Build charset & vectorize
    print("\n  Building charset and vectorizing SMILES...")
    charset = sorted(set("".join(smiles)))
    charset = ["!", "E"] + charset
    char_to_int = build_char_maps(charset)
    num_words = len(charset)
    max_len = max(len(s) for s in smiles)
    embed_len = max_len + 5
    print(f"    Charset: {num_words}, max len: {max_len}, embed_len: {embed_len}")

    X = vectorize_smiles(smiles, char_to_int, embed_len)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = class_bin[train_idx], class_bin[val_idx], class_bin[test_idx]

    # Train
    print("\n  Training BiGRU classifier...")
    y_pred_proba, _, model = train_dl_model(
        model_type="bigru",
        X_train=X_train, X_test=X_test,
        y_train=y_train.astype(np.float32),
        y_test=y_test.astype(np.float32),
        num_words=num_words,
        input_length=embed_len - 1,
        name="mamede_bigru_cls",
        results_dir=RESULTS_DIR,
        seed=SEED, epochs=EPOCHS,
        batch_size=BATCH_SIZE, patience=PATIENCE,
        X_val=X_val, y_val=y_val.astype(np.float32),
        task="classification",
    )

    # Evaluate
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    cls_metrics = compute_classification_metrics(y_test, y_pred_bin, y_pred_proba)
    print_metrics_table(cls_metrics, "BiGRU classifier (no solvent)")

    # Save
    config = {
        "model": "bigru_cls", "solvent": False,
        "n_total": len(df),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "patience": PATIENCE,
        "seed": SEED, "threshold": 0.5,
    }
    save_experiment_results(
        "mamede_bigru_cls", cls_metrics, model,
        y_pred_proba, y_test, y_pred_bin, test_idx, config
    )
    return cls_metrics


# ─── Experiment 2: BiGRU binary classifier (with solvent) ───────────────────

def run_bigru_cls_solvent(args):
    """Train BiGRU as binary classifier on solvent-concatenated SMILES."""
    print("=" * 60)
    print("  EXPERIMENT 2: BiGRU BINARY CLASSIFIER (with solvent)")
    print("  ~69,400 molecules, solvent-concatenated SMILES")
    print("=" * 60)

    # Load data
    df = load_classification_data_with_solvent()
    combined_smiles = df["combined_smiles"].values
    class_bin = df["class_bin"].values

    # Split (different split from Exp 1 due to different dataset)
    print("\n  Creating 80/10/10 stratified split...")
    train_idx, val_idx, test_idx = create_classification_split(
        len(df), class_bin, "solvent"
    )

    # Build charset & vectorize
    print("\n  Building charset and vectorizing combined SMILES...")
    charset = sorted(set("".join(combined_smiles)))
    charset = ["!", "E"] + charset
    # Ensure "!" is in charset (used as delimiter in combined SMILES)
    char_to_int = build_char_maps(charset)
    num_words = len(charset)
    max_len = max(len(s) for s in combined_smiles)
    embed_len = max_len + 5
    print(f"    Charset: {num_words}, max len: {max_len}, embed_len: {embed_len}")

    X = vectorize_smiles(combined_smiles, char_to_int, embed_len)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = class_bin[train_idx], class_bin[val_idx], class_bin[test_idx]

    # Train
    print("\n  Training BiGRU classifier (with solvent)...")
    y_pred_proba, _, model = train_dl_model(
        model_type="bigru",
        X_train=X_train, X_test=X_test,
        y_train=y_train.astype(np.float32),
        y_test=y_test.astype(np.float32),
        num_words=num_words,
        input_length=embed_len - 1,
        name="mamede_bigru_cls_solvent",
        results_dir=RESULTS_DIR,
        seed=SEED, epochs=EPOCHS,
        batch_size=BATCH_SIZE, patience=PATIENCE,
        X_val=X_val, y_val=y_val.astype(np.float32),
        task="classification",
    )

    # Evaluate
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    cls_metrics = compute_classification_metrics(y_test, y_pred_bin, y_pred_proba)
    print_metrics_table(cls_metrics, "BiGRU classifier (with solvent)")

    # Save
    config = {
        "model": "bigru_cls_solvent", "solvent": True,
        "n_total": len(df),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "patience": PATIENCE,
        "seed": SEED, "threshold": 0.5,
    }
    save_experiment_results(
        "mamede_bigru_cls_solvent", cls_metrics, model,
        y_pred_proba, y_test, y_pred_bin, test_idx, config
    )
    return cls_metrics


# ─── Experiment 3: Hybrid BiGRU + Morgan FP classifier ─────────────────────

def run_hybrid_cls(args):
    """Train hybrid BiGRU + Morgan FP as binary classifier."""
    import tensorflow as tf

    print("=" * 60)
    print("  EXPERIMENT 3: HYBRID BiGRU + MORGAN FP CLASSIFIER")
    print("  74,783 molecules, multi-input model")
    print("=" * 60)

    # Load data
    df = load_classification_data()
    smiles = df["smiles"].values
    class_bin = df["class_bin"].values

    # Split (same as Exp 1 — same 74,783 molecules)
    print("\n  Creating 80/10/10 stratified split...")
    train_idx, val_idx, test_idx = create_classification_split(
        len(df), class_bin, "74783"
    )

    # Build charset & vectorize SMILES
    print("\n  Building charset and vectorizing SMILES...")
    charset = sorted(set("".join(smiles)))
    charset = ["!", "E"] + charset
    char_to_int = build_char_maps(charset)
    num_words = len(charset)
    max_len = max(len(s) for s in smiles)
    embed_len = max_len + 5
    print(f"    Charset: {num_words}, max len: {max_len}, embed_len: {embed_len}")

    X_smiles = vectorize_smiles(smiles, char_to_int, embed_len)

    # Compute Morgan fingerprints
    print("\n  Computing Morgan fingerprints...")
    X_fp, valid_mask = compute_morgan_fps(smiles)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"    WARNING: {n_invalid} invalid SMILES (using zero vectors)")

    # Split
    X_smi_train = X_smiles[train_idx]
    X_smi_val = X_smiles[val_idx]
    X_smi_test = X_smiles[test_idx]
    X_fp_train = X_fp[train_idx]
    X_fp_val = X_fp[val_idx]
    X_fp_test = X_fp[test_idx]
    y_train = class_bin[train_idx].astype(np.float32)
    y_val = class_bin[val_idx].astype(np.float32)
    y_test = class_bin[test_idx].astype(np.float32)

    # Build and compile model
    print("\n  Building hybrid model...")
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    verify_gpu_available()

    model = create_hybrid_model(num_words, embed_len - 1, fp_dim=2048)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-8)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    params = model.count_params()
    print(f"    Hybrid model — {params:,} params")
    model.summary(print_fn=lambda x: print(f"    {x}"))

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=PATIENCE,
        verbose=0, mode="auto", restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(RESULTS_DIR, "mamede_hybrid_cls_best.keras"),
        monitor="val_loss", save_best_only=True, verbose=0
    )
    TqdmProgressCallback = get_tqdm_callback()
    progress = TqdmProgressCallback(EPOCHS, model_name="hybrid_cls", report_interval=10)

    # Train
    print("\n  Training hybrid model...")
    t0 = time.time()
    history = model.fit(
        [X_smi_train, X_fp_train], y_train,
        validation_data=([X_smi_val, X_fp_val], y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, progress], verbose=0,
    )
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s")

    # Save history
    hist_path = os.path.join(RESULTS_DIR, "mamede_hybrid_cls_history.json")
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f)

    # Predict
    y_pred_proba = model.predict([X_smi_test, X_fp_test], verbose=0).flatten()
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    cls_metrics = compute_classification_metrics(
        y_test.astype(int), y_pred_bin, y_pred_proba
    )
    print_metrics_table(cls_metrics, "Hybrid BiGRU+FP classifier")

    # Save
    config = {
        "model": "hybrid_cls", "solvent": False,
        "n_total": len(df),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "patience": PATIENCE,
        "seed": SEED, "threshold": 0.5, "fp_dim": 2048,
    }
    save_experiment_results(
        "mamede_hybrid_cls", cls_metrics, model,
        y_pred_proba, y_test.astype(int), y_pred_bin, test_idx, config
    )
    return cls_metrics


# ─── Experiment 4: Regression → threshold (original approach) ───────────────

def run_regression(args):
    """Train BiGRU regression on Mamede data, then threshold for classification."""
    print("=" * 60)
    print("  EXPERIMENT 4: BiGRU REGRESSION → THRESHOLD")
    print("  Matching original Mamede experiment")
    print("=" * 60)

    # Load data
    df = load_regression_data()
    smiles = df["smiles"].values
    y = df["lambda_max"].values
    class_bin = df["class_bin"].values

    # Split — use own split since dataset is different size (~69,998 vs 74,783)
    print("\n  Creating 80/10/10 stratified split...")
    train_idx, val_idx, test_idx = create_classification_split(
        len(df), class_bin, "regression"
    )

    # Build charset & vectorize
    print("\n  Building charset and vectorizing SMILES...")
    charset = sorted(set("".join(smiles)))
    charset = ["!", "E"] + charset
    char_to_int = build_char_maps(charset)
    num_words = len(charset)
    max_len = max(len(s) for s in smiles)
    embed_len = max_len + 5
    print(f"    Charset: {num_words}, max len: {max_len}, embed_len: {embed_len}")

    X = vectorize_smiles(smiles, char_to_int, embed_len)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Train regression model
    print("\n  Training BiGRU regression model...")
    y_pred, reg_metrics, model = train_dl_model(
        model_type="bigru",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        num_words=num_words,
        input_length=embed_len - 1,
        name="mamede_nosolvent",
        results_dir=RESULTS_DIR,
        seed=SEED, epochs=EPOCHS,
        batch_size=BATCH_SIZE, patience=PATIENCE,
        X_val=X_val, y_val=y_val,
        task="regression",
    )

    # Classify predictions
    print("\n  Classifying predictions...")
    y_true_bin = class_bin[test_idx]
    y_pred_bin = classify_photosafety(y_pred)
    y_pred_score = photosafety_score(y_pred)

    cls_metrics = compute_classification_metrics(y_true_bin, y_pred_bin, y_pred_score)
    print_metrics_table(cls_metrics, "BiGRU regression → threshold")

    print(f"\n  Regression metrics:")
    print(f"    RMSE={reg_metrics['RMSE']}, MAE={reg_metrics['MAE']}, "
          f"R²={reg_metrics['R2']}, r={reg_metrics['Pearson_r']}")

    # Save
    config = {
        "model": "regression", "solvent": False,
        "n_total": len(df),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "patience": PATIENCE,
        "seed": SEED, "threshold_lo": POS_LO, "threshold_hi": POS_HI,
    }
    all_metrics = {
        "classification": cls_metrics,
        "regression": reg_metrics,
        "config": config,
    }
    metrics_path = os.path.join(RESULTS_DIR, "mamede_nosolvent_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    pred_path = os.path.join(RESULTS_DIR, "mamede_nosolvent_predictions.npz")
    np.savez(pred_path, y_test=y_test, y_pred=y_pred,
             y_true_bin=y_true_bin, y_pred_bin=y_pred_bin, test_idx=test_idx)
    print(f"  Saved: {pred_path}")

    model_path = os.path.join(RESULTS_DIR, "mamede_nosolvent_model.keras")
    model.save(model_path)
    print(f"  Saved: {model_path}")

    return cls_metrics


# ─── Experiment 5: RF + Morgan FP (reproducing Mamede's method) ──────────────

def run_rf_cls(args):
    """Reproduce Mamede's RF + Morgan FP 1024-bit classifier.

    Mamede et al. (2021): 500 trees, 1024-bit Morgan FP, 72,788 train / 998 test I / 998 test II.
    """
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    print("=" * 60)
    print("  EXPERIMENT 5: RF + MORGAN FP (Mamede reproduction)")
    print("  74,783 molecules, 1024-bit Morgan FP, 500 trees")
    print("=" * 60)

    # 1. Load data
    df = load_classification_data()
    smiles = df["smiles"].values
    class_bin = df["class_bin"].values

    # 2. Compute Morgan FP (1024 bits, radius 2) — matches Mamede
    print("\n  Computing Morgan fingerprints (1024 bits)...")
    X_fp, valid_mask = compute_morgan_fps(smiles, radius=2, n_bits=1024)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"    WARNING: {n_invalid} invalid SMILES (using zero vectors)")

    # 3. Split: 72,787 train / 998 test I / 998 test II
    #    Mamede: 72,788 train + 998 test I + 998 test II = 74,784
    #    Ours:   72,787 train + 998 test I + 998 test II = 74,783
    print("\n  Creating Mamede-style split (train / test I / test II)...")
    split_path = os.path.join(RESULTS_DIR, "mamede_cls_split_rf.npz")

    if os.path.exists(split_path):
        data = np.load(split_path)
        train_idx = data["train"]
        test1_idx = data["test1"]
        test2_idx = data["test2"]
        print(f"  Loaded cached split from {split_path}")
    else:
        # Hold out 1996 for two test sets
        all_idx = np.arange(len(df))
        train_idx, held_idx = train_test_split(
            all_idx, test_size=1996, random_state=SEED, stratify=class_bin
        )
        # Split held-out 50/50 into test I and test II
        held_class = class_bin[held_idx]
        test1_idx_rel, test2_idx_rel = train_test_split(
            np.arange(len(held_idx)), test_size=0.5, random_state=SEED,
            stratify=held_class
        )
        test1_idx = held_idx[test1_idx_rel]
        test2_idx = held_idx[test2_idx_rel]
        np.savez(split_path, train=train_idx, test1=test1_idx, test2=test2_idx)
        print(f"  Saved split to {split_path}")

    print(f"    Train:  {len(train_idx)} (POS: {class_bin[train_idx].sum()}, "
          f"NEG: {(1 - class_bin[train_idx]).sum()})")
    print(f"    Test I: {len(test1_idx)} (POS: {class_bin[test1_idx].sum()}, "
          f"NEG: {(1 - class_bin[test1_idx]).sum()})")
    print(f"    Test II:{len(test2_idx)} (POS: {class_bin[test2_idx].sum()}, "
          f"NEG: {(1 - class_bin[test2_idx]).sum()})")

    # 4. Train RF
    print("\n  Training Random Forest (500 trees, OOB)...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=500, random_state=SEED, oob_score=True, n_jobs=-1
    )
    rf.fit(X_fp[train_idx], class_bin[train_idx])
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")
    print(f"  OOB score: {rf.oob_score_:.4f}")

    # 5. Evaluate on both test sets
    all_test_metrics = {"oob_score": round(rf.oob_score_, 4)}
    for name, tidx in [("Test I", test1_idx), ("Test II", test2_idx)]:
        y_pred_bin = rf.predict(X_fp[tidx])
        y_pred_proba = rf.predict_proba(X_fp[tidx])[:, 1]
        metrics = compute_classification_metrics(
            class_bin[tidx], y_pred_bin, y_pred_proba
        )
        print_metrics_table(metrics, f"RF classifier ({name})")
        all_test_metrics[name] = metrics

    # Also evaluate on combined test (Test I + Test II)
    combined_idx = np.concatenate([test1_idx, test2_idx])
    y_pred_bin_all = rf.predict(X_fp[combined_idx])
    y_pred_proba_all = rf.predict_proba(X_fp[combined_idx])[:, 1]
    combined_metrics = compute_classification_metrics(
        class_bin[combined_idx], y_pred_bin_all, y_pred_proba_all
    )
    print_metrics_table(combined_metrics, "RF classifier (Test I + II combined)")
    all_test_metrics["Combined"] = combined_metrics

    # 6. Save results
    config = {
        "model": "rf_cls",
        "n_estimators": 500,
        "fp_bits": 1024,
        "fp_radius": 2,
        "n_total": len(df),
        "n_train": int(len(train_idx)),
        "n_test1": int(len(test1_idx)),
        "n_test2": int(len(test2_idx)),
        "seed": SEED,
        "n_invalid_smiles": int(n_invalid),
    }
    all_metrics = {
        "classification": combined_metrics,  # Use combined for comparison table
        "test_I": all_test_metrics["Test I"],
        "test_II": all_test_metrics["Test II"],
        "oob_score": all_test_metrics["oob_score"],
        "config": config,
    }

    metrics_path = os.path.join(RESULTS_DIR, "mamede_rf_cls_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved: {metrics_path}")

    pred_path = os.path.join(RESULTS_DIR, "mamede_rf_cls_predictions.npz")
    np.savez(
        pred_path,
        test1_idx=test1_idx, test2_idx=test2_idx,
        test1_y_true=class_bin[test1_idx],
        test1_y_pred=rf.predict(X_fp[test1_idx]),
        test1_y_proba=rf.predict_proba(X_fp[test1_idx])[:, 1],
        test2_y_true=class_bin[test2_idx],
        test2_y_pred=rf.predict(X_fp[test2_idx]),
        test2_y_proba=rf.predict_proba(X_fp[test2_idx])[:, 1],
    )
    print(f"  Saved: {pred_path}")

    model_path = os.path.join(RESULTS_DIR, "mamede_rf_cls_model.joblib")
    joblib.dump(rf, model_path)
    print(f"  Saved: {model_path}")

    return combined_metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

MODEL_DISPATCH = {
    "bigru_cls": run_bigru_cls,
    "bigru_cls_solvent": run_bigru_cls_solvent,
    "hybrid_cls": run_hybrid_cls,
    "regression": run_regression,
    "rf_cls": run_rf_cls,
}


def main():
    parser = argparse.ArgumentParser(
        description="Mamede photosafety classification experiments"
    )
    parser.add_argument(
        "--model", choices=list(MODEL_DISPATCH.keys()),
        help="Experiment to run"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print comparison table from saved results"
    )
    args = parser.parse_args()

    if args.summary:
        print_comparison_table()
        return

    if args.model is None:
        parser.error("--model is required (unless --summary)")

    # Run the experiment
    cls_metrics = MODEL_DISPATCH[args.model](args)

    # Print comparison table at end
    print_comparison_table()

    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT COMPLETE: {args.model}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
