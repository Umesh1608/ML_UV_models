#!/usr/bin/env python3
"""
Train BiGRU direct classification on the Mamede/Reaxys photosafety dataset.

Uses the tuned BiGRU architecture (3L/256u, embed=50, dropout=0.105, lr=0.00111)
with sigmoid output + BCE loss for direct binary classification.

Matches the RF classification protocol:
- Same 74,783-compound Mamede/Reaxys dataset
- Same train/test split (from mamede_cls_split_rf.npz)
- POS = λ_max ∈ [290,700] AND MEC ≥ 1000

Usage:
    python3 train_bigru_classification.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

sys.path.insert(0, SCRIPT_DIR)
from paper1_new_cl.models import (
    build_char_maps,
    vectorize_smiles,
    train_dl_model,
    verify_gpu_available,
)


def main():
    print(f"\n{'=' * 70}")
    print(f"  BiGRU DIRECT CLASSIFICATION ON MAMEDE/REAXYS PHOTOSAFETY DATASET")
    print(f"{'=' * 70}")

    # ── Load dataset ──────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, "mamede_classification_dataset.csv"))
    print(f"\n  Dataset: {len(df)} compounds")
    print(f"  Labels: {df['class_label'].value_counts().to_dict()}")

    # Binary encode: POS=1, NEG=0
    y_all = (df["class_label"] == "POS").astype(np.float32).values
    smiles_all = df["smiles"].values

    # ── Load same split as RF ────────────────────────────────────────────
    split = np.load(os.path.join(RESULTS_DIR, "mamede_cls_split_rf.npz"))
    train_idx = split["train"]
    test1_idx = split["test1"]
    test2_idx = split["test2"]
    # Combine test1 and test2 for full test set (matches RF evaluation)
    test_idx = np.concatenate([test1_idx, test2_idx])

    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)} "
          f"(test1={len(test1_idx)}, test2={len(test2_idx)})")

    smiles_train = smiles_all[train_idx]
    smiles_test = smiles_all[test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    print(f"  Train POS: {y_train.sum():.0f} ({100*y_train.mean():.1f}%), "
          f"NEG: {len(y_train) - y_train.sum():.0f}")
    print(f"  Test POS: {y_test.sum():.0f} ({100*y_test.mean():.1f}%), "
          f"NEG: {len(y_test) - y_test.sum():.0f}")

    # ── Build character vocabulary ────────────────────────────────────────
    # Use same approach as primary dataset: build charset from all SMILES
    all_chars = set()
    max_len = 0
    for smi in smiles_all:
        all_chars.update(set(smi))
        max_len = max(max_len, len(smi))

    charset = sorted(list(all_chars)) + ["!", "E"]  # start/end tokens
    char_to_int = build_char_maps(charset)
    embed_len = max_len + 5  # padding
    input_length = embed_len - 1

    print(f"\n  Vocabulary: {len(charset)} chars, max SMILES len: {max_len}, "
          f"embed_len: {embed_len}")

    # ── Vectorize SMILES ──────────────────────────────────────────────────
    print("  Vectorizing SMILES...")
    X_train = vectorize_smiles(smiles_train, char_to_int, embed_len)
    X_test = vectorize_smiles(smiles_test, char_to_int, embed_len)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # ── Create validation split from training data ────────────────────────
    # 10% of train for validation (early stopping)
    np.random.seed(7)
    n_val = int(0.1 * len(X_train))
    val_perm = np.random.permutation(len(X_train))
    val_indices = val_perm[:n_val]
    train_indices = val_perm[n_val:]

    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train_final = X_train[train_indices]
    y_train_final = y_train[train_indices]

    print(f"  After val split: train={len(X_train_final)}, val={len(X_val)}")

    # ── Train BiGRU (tuned architecture) ──────────────────────────────────
    # Tuned hyperparameters from HPO Trial 25
    print(f"\n  Architecture: BiGRU 3L/256u, embed=50, dropout=0.105, lr=0.00111")
    print(f"  Task: binary classification (sigmoid + BCE)")

    y_pred_proba, _, model = train_dl_model(
        model_type="bigru",
        X_train=X_train_final,
        X_test=X_test,
        y_train=y_train_final,
        y_test=y_test,
        num_words=len(charset),
        input_length=input_length,
        name="mamede_bigru_cls_tuned",
        results_dir=RESULTS_DIR,
        seed=7,
        epochs=200,
        batch_size=128,
        patience=20,
        X_val=X_val,
        y_val=y_val,
        task="classification",
        n_units=256,
        n_layers=3,
        embed_dim=50,
        dropout=0.105,
        lr=0.00111,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    from sklearn.metrics import (
        confusion_matrix, matthews_corrcoef, accuracy_score,
        roc_auc_score, f1_score, precision_score, recall_score,
    )

    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    y_test_int = y_test.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_int, y_pred_bin, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = accuracy_score(y_test_int, y_pred_bin)
    mcc = matthews_corrcoef(y_test_int, y_pred_bin)
    f1 = f1_score(y_test_int, y_pred_bin)
    precision = precision_score(y_test_int, y_pred_bin)
    auc = roc_auc_score(y_test_int, y_pred_proba)

    metrics = {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
        "ROC-AUC": round(auc, 4),
        "N": int(tp + fp + tn + fn),
        "N_POS": int(tp + fn),
        "N_NEG": int(tn + fp),
    }

    # Also evaluate on test1 and test2 separately
    n_test1 = len(test1_idx)
    y_pred_test1 = y_pred_proba[:n_test1]
    y_pred_test2 = y_pred_proba[n_test1:]
    y_test1 = y_test[:n_test1]
    y_test2 = y_test[n_test1:]

    def eval_subset(y_true, y_pred_p, name):
        yp = (y_pred_p >= 0.5).astype(int)
        yt = y_true.astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = accuracy_score(yt, yp)
        return {
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "Sensitivity": round(sens, 4),
            "Specificity": round(spec, 4),
            "Accuracy": round(acc, 4),
            "MCC": round(matthews_corrcoef(yt, yp), 4),
            "ROC-AUC": round(roc_auc_score(yt, y_pred_p), 4),
        }

    metrics_test1 = eval_subset(y_test1, y_pred_test1, "test_I")
    metrics_test2 = eval_subset(y_test2, y_pred_test2, "test_II")

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: BiGRU Direct Classification (tuned 3L/256u)")
    print(f"{'=' * 70}")
    print(f"  Combined test (N={metrics['N']}):")
    print(f"    Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f"    Specificity: {metrics['Specificity']:.4f}")
    print(f"    Accuracy:    {metrics['Accuracy']:.4f}")
    print(f"    MCC:         {metrics['MCC']:.4f}")
    print(f"    ROC-AUC:     {metrics['ROC-AUC']:.4f}")
    print(f"    F1:          {metrics['F1']:.4f}")
    print(f"\n  Test I (N={len(y_test1)}):")
    print(f"    Sens={metrics_test1['Sensitivity']:.4f}  "
          f"Spec={metrics_test1['Specificity']:.4f}  "
          f"Acc={metrics_test1['Accuracy']:.4f}")
    print(f"\n  Test II (N={len(y_test2)}):")
    print(f"    Sens={metrics_test2['Sensitivity']:.4f}  "
          f"Spec={metrics_test2['Specificity']:.4f}  "
          f"Acc={metrics_test2['Accuracy']:.4f}")

    print(f"\n  Comparison:")
    print(f"    Mamede RF (2021):           Sens=0.90, Spec=0.88, Acc=0.89")
    print(f"    Our RF:                     Sens=0.876, Spec=0.882, Acc=0.879")
    print(f"    BiGRU direct cls (tuned):   Sens={metrics['Sensitivity']:.3f}, "
          f"Spec={metrics['Specificity']:.3f}, Acc={metrics['Accuracy']:.3f}")
    print(f"    BiGRU regress→threshold:    Sens=0.220, Spec=0.964, Acc=0.588")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "classification": metrics,
        "test_I": metrics_test1,
        "test_II": metrics_test2,
        "config": {
            "model": "bigru_cls_tuned",
            "n_units": 256,
            "n_layers": 3,
            "embed_dim": 50,
            "dropout": 0.105,
            "lr": 0.00111,
            "batch_size": 128,
            "epochs": 200,
            "patience": 20,
            "n_total": len(df),
            "n_train": len(X_train_final),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "charset_size": len(charset),
            "embed_len": embed_len,
            "seed": 7,
        },
    }

    out_path = os.path.join(RESULTS_DIR, "mamede_bigru_cls_tuned_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics saved to {out_path}")

    # Save predictions
    np.savez(
        os.path.join(RESULTS_DIR, "mamede_bigru_cls_tuned_predictions.npz"),
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        y_pred_bin=y_pred_bin,
        test1_idx=test1_idx,
        test2_idx=test2_idx,
    )
    print(f"  Predictions saved to results/mamede_bigru_cls_tuned_predictions.npz")

    # Save model
    model.save(os.path.join(RESULTS_DIR, "mamede_bigru_cls_tuned_model.keras"))
    print(f"  Model saved to results/mamede_bigru_cls_tuned_model.keras")

    # Update comparison CSV
    comparison = pd.DataFrame([
        {"Model": "Mamede RF (2021)", "N_test": "~998",
         "Sensitivity": 0.90, "Specificity": 0.88, "Accuracy": 0.89,
         "MCC": "", "F1": "", "ROC-AUC": ""},
        {"Model": "RF + Morgan FP 1024-bit (ours)", "N_test": 1996,
         "Sensitivity": 0.8761, "Specificity": 0.8821, "Accuracy": 0.8793,
         "MCC": 0.7579, "F1": 0.8728, "ROC-AUC": 0.9424},
        {"Model": "BiGRU direct cls (tuned 3L/256u)", "N_test": metrics["N"],
         "Sensitivity": metrics["Sensitivity"], "Specificity": metrics["Specificity"],
         "Accuracy": metrics["Accuracy"], "MCC": metrics["MCC"],
         "F1": metrics["F1"], "ROC-AUC": metrics["ROC-AUC"]},
        {"Model": "BiGRU regression → threshold", "N_test": 1400,
         "Sensitivity": 0.2203, "Specificity": 0.9639, "Accuracy": 0.5879,
         "MCC": 0.2745, "F1": 0.351, "ROC-AUC": 0.6479},
    ])
    comparison.to_csv(os.path.join(RESULTS_DIR, "mamede_comparison.csv"), index=False)
    print(f"  Comparison table saved to results/mamede_comparison.csv")

    print(f"\n{'=' * 70}")
    print(f"  CLASSIFICATION TRAINING COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
