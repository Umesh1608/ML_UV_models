#!/usr/bin/env python3
"""
Generate 3×2 multi-model parity plot (RF, XGBoost, Chemprop, BiGRU, ChemBERTa)
from pooled cross-validation predictions.

Output: results/parity_plot_5models.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_pooled(prefix):
    """Load pooled predictions from npz or assemble from per-fold npy files."""
    npz_path = os.path.join(RESULTS_DIR, f"{prefix}_cv_pooled.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        y_key = "y_true" if "y_true" in data else "y_test"
        return data[y_key], data["y_pred"]

    # Assemble from per-fold files
    all_y, all_pred = [], []
    for fold in range(5):
        y_path = os.path.join(RESULTS_DIR, f"{prefix}_fold{fold}_y_test.npy")
        p_path = os.path.join(RESULTS_DIR, f"{prefix}_fold{fold}_predictions.npy")
        if os.path.exists(y_path) and os.path.exists(p_path):
            all_y.append(np.load(y_path))
            all_pred.append(np.load(p_path))
    if all_y:
        return np.concatenate(all_y), np.concatenate(all_pred)
    raise FileNotFoundError(f"No data found for {prefix}")


def main():
    models = [
        ("RF (tuned)", "rf_tuned"),
        ("XGBoost", "xgboost_v2"),
        ("Chemprop (D-MPNN)", "chemprop_v2"),
        ("BiGRU (tuned)", "bigru_solvent_v2"),
        ("ChemBERTa", "chemberta"),
    ]

    # Load all data first to compute shared axis limits
    all_data = []
    for name, prefix in models:
        y_true, y_pred = load_pooled(prefix)
        all_data.append((name, y_true, y_pred))

    global_min = min(min(y.min(), p.min()) for _, y, p in all_data) - 10
    global_max = max(max(y.max(), p.max()) for _, y, p in all_data) + 10
    lims = [global_min, global_max]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (name, y_true, y_pred) in zip(axes, all_data):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.15, s=5, c="steelblue", rasterized=True)

        # Perfect prediction line
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.7)

        # ±20 nm bands
        ax.fill_between(lims, [l - 20 for l in lims], [l + 20 for l in lims],
                        alpha=0.08, color="green")

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Experimental $\\lambda_{\\max}$ (nm)", fontsize=11)
        ax.set_ylabel("Predicted $\\lambda_{\\max}$ (nm)", fontsize=11)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_aspect("equal")

        textstr = f"RMSE = {rmse:.1f} nm\nMAE = {mae:.1f} nm\n$R^2$ = {r2:.3f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        ax.text(0.95, 0.05, f"n = {len(y_true):,}", transform=ax.transAxes,
                fontsize=9, ha="right", va="bottom", color="gray")

    # Hide the 6th (empty) subplot
    axes[5].set_visible(False)

    plt.tight_layout(pad=2.0)
    out_path = os.path.join(RESULTS_DIR, "parity_plot_5models.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved parity plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
