#!/usr/bin/env python3
"""
Photosafety screening classification from UV λ_max predictions.

Repurposes regression model predictions as a binary photosafety classifier:
  POS (photoreactive) if λ_max ∈ [290, 700] nm
  NEG otherwise

Computes sensitivity, specificity, MCC, ROC-AUC, and compares against
Mamede et al. (2021) published results as a benchmark reference.

Uses pooled 5-fold CV predictions from run_baselines.py.

Usage:
  python3 eval_classification.py                           # all models
  python3 eval_classification.py --model bigru_solvent     # one model
  python3 eval_classification.py --threshold 290 700       # custom range
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Model registry (mirrors run_baselines.py)
MODEL_REGISTRY = {
    "bigru_solvent":   "BiGRU + Solvent",
    "bigru_nosolvent": "BiGRU (no solvent)",
    "bilstm":          "BiLSTM + Solvent",
    "cnn_bigru":       "CNN-BiGRU + Solvent",
    "rf":              "RF + Morgan FP",
    "xgboost":         "XGBoost + Morgan FP",
}
MODEL_ORDER = ["bigru_solvent", "bigru_nosolvent", "bilstm", "cnn_bigru", "rf", "xgboost"]

# Default ICH S10 photosafety threshold
DEFAULT_LO = 290.0
DEFAULT_HI = 700.0

# Mamede et al. (2021) published results for reference
MAMEDE_REFERENCE = {
    "Model": "Mamede RF (reference)",
    "Sensitivity": 0.90,
    "Specificity": 0.88,
    "MCC": None,       # not reported
    "ROC-AUC": None,   # not reported
    "F1": None,         # not reported
    "Precision": None,  # not reported
    "N": "~74K",
    "Note": "RF + Morgan FP; POS = λ_max ∈ [290,700] AND MEC ≥ 1000",
}


# ─── Classification helpers ──────────────────────────────────────────────────

def classify_photosafety(lmax, lo, hi):
    """POS (1) if λ_max ∈ [lo, hi], else NEG (0)."""
    return ((lmax >= lo) & (lmax <= hi)).astype(int)


def photosafety_score(y_true_lmax, y_pred_lmax, lo, hi):
    """Compute continuous score for ROC-AUC.

    We need a score that's higher for POS predictions. POS = λ_max ∈ [lo, hi].
    Use negative distance from the interval center as the score — closer to
    center = more likely POS.
    """
    center = (lo + hi) / 2.0
    return -np.abs(y_pred_lmax - center)


def compute_classification_metrics(y_true_lmax, y_pred_lmax, lo, hi):
    """Compute all classification metrics from continuous λ_max values."""
    y_true_bin = classify_photosafety(y_true_lmax, lo, hi)
    y_pred_bin = classify_photosafety(y_pred_lmax, lo, hi)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

    # ROC-AUC using continuous score (negative distance from interval center)
    score = photosafety_score(y_true_lmax, y_pred_lmax, lo, hi)
    n_classes = len(np.unique(y_true_bin))
    if n_classes == 2:
        auc = roc_auc_score(y_true_bin, score)
    else:
        auc = float("nan")

    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "Precision": round(precision, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
        "ROC-AUC": round(auc, 4),
        "N": int(tp + fp + tn + fn),
        "N_POS": int(tp + fn),
        "N_NEG": int(tn + fp),
    }


# ─── Load pooled predictions ─────────────────────────────────────────────────

def load_pooled(model_key):
    """Load pooled 5-fold CV predictions for a model."""
    path = os.path.join(RESULTS_DIR, f"{model_key}_cv_pooled.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["y_test"], data["y_pred"]


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_confusion_matrices(all_metrics, lo, hi):
    """Plot confusion matrices for all models in a grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_metrics)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

    for idx, (key, m) in enumerate(all_metrics.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        im = ax.imshow(cm, cmap="Blues", aspect="equal")

        # Annotate cells
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["NEG", "POS"])
        ax.set_yticklabels(["NEG", "POS"])
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        name = MODEL_REGISTRY.get(key, key)
        ax.set_title(f"{name}\nMCC={m['MCC']:.3f}", fontsize=10)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Photosafety Classification (POS: λ_max ∈ [{lo:.0f}, {hi:.0f}] nm)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(RESULTS_DIR, f"classification_confusion.{ext}")
        plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Confusion matrices saved to results/classification_confusion.png")


def plot_roc_curves(all_data, lo, hi):
    """Plot overlaid ROC curves for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (key, (y_true_lmax, y_pred_lmax)) in enumerate(all_data.items()):
        y_true_bin = classify_photosafety(y_true_lmax, lo, hi)
        score = photosafety_score(y_true_lmax, y_pred_lmax, lo, hi)

        n_classes = len(np.unique(y_true_bin))
        if n_classes < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin, score)
        auc = roc_auc_score(y_true_bin, score)
        name = MODEL_REGISTRY.get(key, key)
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=1.8)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title(f"ROC: Photosafety Classification (POS: λ_max ∈ [{lo:.0f}, {hi:.0f}] nm)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(RESULTS_DIR, f"classification_roc.{ext}")
        plt.savefig(out, dpi=300)
    plt.close()
    print(f"  ROC curves saved to results/classification_roc.png")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_comparison_table(all_metrics, lo, hi):
    """Print formatted comparison table with Mamede reference."""
    rows = []
    for key in MODEL_ORDER:
        if key not in all_metrics:
            continue
        m = all_metrics[key]
        rows.append({
            "Model": MODEL_REGISTRY.get(key, key),
            "N": m["N"],
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "MCC": m["MCC"],
            "ROC-AUC": m["ROC-AUC"],
            "F1": m["F1"],
            "Precision": m["Precision"],
        })

    # Add Mamede reference row
    rows.append({
        "Model": MAMEDE_REFERENCE["Model"],
        "N": MAMEDE_REFERENCE["N"],
        "Sensitivity": MAMEDE_REFERENCE["Sensitivity"],
        "Specificity": MAMEDE_REFERENCE["Specificity"],
        "MCC": "—",
        "ROC-AUC": "—",
        "F1": "—",
        "Precision": "—",
    })

    df = pd.DataFrame(rows)
    print(f"\n{'═' * 90}")
    print(f"  PHOTOSAFETY CLASSIFICATION RESULTS (POS: λ_max ∈ [{lo:.0f}, {hi:.0f}] nm)")
    print(f"{'═' * 90}")
    print(df.to_string(index=False))
    print(f"\n  Note: Mamede et al. (2021) used POS = λ_max ∈ [290, 700] AND MEC ≥ 1000.")
    print(f"  Our classification uses λ_max only (no MEC filter), on Joung+Beard data.")

    return df


def save_metrics_csv(all_metrics, lo, hi):
    """Save classification metrics to CSV."""
    rows = []
    for key in MODEL_ORDER:
        if key not in all_metrics:
            continue
        m = all_metrics[key]
        rows.append({
            "model_key": key,
            "model_name": MODEL_REGISTRY.get(key, key),
            "threshold_lo": lo,
            "threshold_hi": hi,
            **{k: v for k, v in m.items()},
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "classification_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"  Metrics saved to {out_path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Photosafety classification from UV λ_max predictions"
    )
    parser.add_argument(
        "--model", type=str, choices=list(MODEL_REGISTRY.keys()),
        help="Evaluate a single model (default: all available)"
    )
    parser.add_argument(
        "--threshold", type=float, nargs=2, default=[DEFAULT_LO, DEFAULT_HI],
        metavar=("LO", "HI"),
        help=f"POS threshold range in nm (default: {DEFAULT_LO} {DEFAULT_HI})"
    )
    args = parser.parse_args()

    lo, hi = args.threshold
    models_to_eval = [args.model] if args.model else MODEL_ORDER

    print(f"\n{'=' * 60}")
    print(f"  PHOTOSAFETY SCREENING CLASSIFICATION")
    print(f"  POS: λ_max ∈ [{lo:.0f}, {hi:.0f}] nm")
    print(f"{'=' * 60}")

    # Load pooled predictions for each model
    all_metrics = {}
    all_data = {}

    for key in models_to_eval:
        y_test, y_pred = load_pooled(key)
        if y_test is None:
            print(f"\n  [{key}] No pooled predictions found. Skipping.")
            print(f"    Run: python3 run_baselines.py --model {key} --summary")
            continue

        name = MODEL_REGISTRY.get(key, key)
        print(f"\n  [{name}] N={len(y_test)}")

        # Class distribution
        y_true_bin = classify_photosafety(y_test, lo, hi)
        n_pos = y_true_bin.sum()
        n_neg = len(y_true_bin) - n_pos
        print(f"    Ground truth: {n_pos} POS ({100*n_pos/len(y_true_bin):.1f}%), "
              f"{n_neg} NEG ({100*n_neg/len(y_true_bin):.1f}%)")

        metrics = compute_classification_metrics(y_test, y_pred, lo, hi)
        all_metrics[key] = metrics
        all_data[key] = (y_test, y_pred)

        print(f"    Sens={metrics['Sensitivity']:.4f}  "
              f"Spec={metrics['Specificity']:.4f}  "
              f"MCC={metrics['MCC']:.4f}  "
              f"AUC={metrics['ROC-AUC']:.4f}  "
              f"F1={metrics['F1']:.4f}")

    if not all_metrics:
        print("\n  No models with pooled predictions found.")
        print("  Run 'python3 run_baselines.py --summary' first.")
        sys.exit(1)

    # Save metrics CSV
    print(f"\n{'─' * 60}")
    save_metrics_csv(all_metrics, lo, hi)

    # Generate plots
    plot_confusion_matrices(all_metrics, lo, hi)
    plot_roc_curves(all_data, lo, hi)

    # Print comparison table
    print_comparison_table(all_metrics, lo, hi)

    print(f"\n{'=' * 60}")
    print(f"  CLASSIFICATION EVALUATION COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
