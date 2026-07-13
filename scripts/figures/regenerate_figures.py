#!/usr/bin/env python3
"""Regenerate improved error analysis figures for benchmark paper.

All 5 models: RF, XGBoost, Chemprop, BiGRU, ChemBERTa.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RESULTS_DIR = "results"
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ── Colors for each model ───────────────────────────────────────────────────
MODEL_COLORS = {
    "RF": "#27ae60",
    "XGBoost": "#f39c12",
    "Chemprop": "#c0392b",
    "BiGRU": "#e74c3c",
    "ChemBERTa": "#3498db",
}


def load_all_models():
    """Load pooled v3 cross-validation predictions for all 5 models.

    Returns dict of {name: (y_pred, y_test, indices)}; indices reference rows of
    the v3 primary dataset (previous_code/UV_canonical_v3_dedup.csv).
    """
    prefixes = {
        "RF": "rf_tuned_v3",
        "XGBoost": "xgboost_v3",
        "Chemprop": "chemprop_v3",
        "BiGRU": "bigru_tuned_v3",
        "ChemBERTa": "chemberta_v3",
    }
    models = {}
    for name, prefix in prefixes.items():
        d = np.load(f"{RESULTS_DIR}/{prefix}_cv_pooled.npz")
        models[name] = (d["y_pred"], d["y_test"], d["indices"])

    for name, (yp, yt, idx) in models.items():
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae = mean_absolute_error(yt, yp)
        print(f"  {name}: n={len(yt)}, RMSE={rmse:.2f}, MAE={mae:.2f}")

    return models


def load_solvents():
    """Load the global solvent array from the v3 primary dataset (indices reference it)."""
    data = pd.read_csv("previous_code/UV_canonical_v3_dedup.csv")
    return data["solvents"].values


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6a: Error by wavelength range — ALL 4 models
# ═════════════════════════════════════════════════════════════════════════════

def figure_6a(models):
    """Grouped bar chart: MAE by wavelength range for all 4 models."""
    ranges = [
        ("< 300", lambda y: y < 300),
        ("300–400", lambda y: (y >= 300) & (y < 400)),
        ("400–500", lambda y: (y >= 400) & (y < 500)),
        ("> 500", lambda y: y >= 500),
    ]

    model_names = list(MODEL_COLORS.keys())
    n_models = len(model_names)
    n_ranges = len(ranges)
    bar_width = 0.15
    x = np.arange(n_ranges)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for j, name in enumerate(model_names):
        y_pred, y_test, _ = models[name]
        abs_errors = np.abs(y_pred - y_test)

        means, medians, counts = [], [], []
        for label, mask_fn in ranges:
            mask = mask_fn(y_test)
            if mask.sum() > 0:
                means.append(abs_errors[mask].mean())
                medians.append(np.median(abs_errors[mask]))
                counts.append(mask.sum())
            else:
                means.append(0)
                medians.append(0)
                counts.append(0)

        offset = (j - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width,
                      label=name, color=MODEL_COLORS[name],
                      edgecolor="white", alpha=0.9)

        # Add count labels on the first model only (same across models)
        if j == 0:
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(x[i], ax.get_ylim()[1] * 0.01, f"n={count:,}",
                        ha="center", fontsize=8, color="#666", style="italic")

    ax.set_xlabel("$\\lambda_{\\max}$ Range (nm)")
    ax.set_ylabel("Mean Absolute Error (nm)")
    ax.set_title("(a) Error by Wavelength Range — 5-Fold CV")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in ranges])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(RESULTS_DIR, f"error_by_wavelength.{ext}"), dpi=300)
    plt.close()
    print("  [OK] Figure 6a: error_by_wavelength.pdf/png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6b: Error by solvent — ALL 4 models
# ═════════════════════════════════════════════════════════════════════════════

def figure_6b(models, solvents):
    """Grouped box plots: absolute error by top-5 solvents for all 4 models."""
    solvent_names = {
        "CCO": "Ethanol", "CO": "Methanol", "ClCCl": "DCM",
        "CC(C)=O": "Acetone", "CCCCCC": "Hexane", "CS(C)=O": "DMSO",
        "CC#N": "Acetonitrile", "O": "Water", "Cc1ccccc1": "Toluene",
        "C1CCOC1": "THF", "ClC(Cl)Cl": "Chloroform",
        "CCOCC": "Diethyl ether", "c1ccccc1": "Benzene",
    }

    # Find top 5 solvents by count (use RF indices as reference since most have same coverage)
    _, y_test_rf, idx_rf = models["RF"]
    sol_rf = solvents[idx_rf]
    unique_solvents, counts = np.unique(sol_rf, return_counts=True)
    top5_idx = np.argsort(-counts)[:5]
    top5_solvents = unique_solvents[top5_idx]

    model_names = list(MODEL_COLORS.keys())
    n_models = len(model_names)
    n_solvents = len(top5_solvents)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    positions = []
    colors = []
    data = []
    labels_placed = set()

    group_width = n_models + 1  # spacing between solvent groups
    for i, sol in enumerate(top5_solvents):
        for j, name in enumerate(model_names):
            y_pred, y_test, idx = models[name]
            sol_arr = solvents[idx]
            mask = sol_arr == sol
            abs_err = np.abs(y_pred[mask] - y_test[mask])
            data.append(abs_err)
            positions.append(i * group_width + j)
            colors.append(MODEL_COLORS[name])

    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # X-axis labels at group centers
    group_centers = [i * group_width + (n_models - 1) / 2 for i in range(n_solvents)]
    sol_labels = [solvent_names.get(s, s[:12]) for s in top5_solvents]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(sol_labels)

    # Legend
    handles = [mpatches.Patch(color=MODEL_COLORS[n], label=n) for n in model_names]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    ax.set_xlabel("Solvent")
    ax.set_ylabel("Absolute Error (nm)")
    ax.set_title("(b) Error Distribution by Solvent — 5-Fold CV")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(RESULTS_DIR, f"error_by_solvent.{ext}"), dpi=300)
    plt.close()
    print("  [OK] Figure 6b: error_by_solvent.pdf/png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Training comparison — ADD ChemBERTa
# ═════════════════════════════════════════════════════════════════════════════

def figure_9():
    """Bar chart comparing RMSE and training time across all 5 models."""
    # v3 (Greenman--Song deduplicated) cross-validated metrics, matching Table 2.
    models = {
        "RF": {
            "rmse": 31.50, "mae": 15.37,
            "train_time_min": 15,
            "color": MODEL_COLORS["RF"],
        },
        "XGBoost": {
            "rmse": 33.92, "mae": 20.25,
            "train_time_min": 5,
            "color": MODEL_COLORS["XGBoost"],
        },
        "Chemprop": {
            "rmse": 33.15, "mae": 18.11,
            "train_time_min": 90,  # ~14s/epoch × ~80 epochs × 5 folds
            "color": MODEL_COLORS["Chemprop"],
        },
        "BiGRU": {
            "rmse": 36.20, "mae": 18.81,
            "train_time_min": 1500,  # ~5 hrs/fold × 5 folds
            "color": MODEL_COLORS["BiGRU"],
        },
        "ChemBERTa": {
            "rmse": 54.09, "mae": 26.78,
            "train_time_min": 900,  # ~3 hrs/fold × 5 folds
            "color": MODEL_COLORS["ChemBERTa"],
        },
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    names = list(models.keys())
    rmse_vals = [models[n]["rmse"] for n in names]
    colors = [models[n]["color"] for n in names]
    times = [models[n]["train_time_min"] for n in names]

    # Panel (a): RMSE comparison
    bars1 = ax1.bar(names, rmse_vals, color=colors, edgecolor="white", alpha=0.9, width=0.55)
    for bar, val in zip(bars1, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", fontsize=12, fontweight="bold")
    ax1.set_ylabel("RMSE (nm)")
    ax1.set_title("(a) Prediction Accuracy", fontweight="bold")
    ax1.set_ylim(0, max(rmse_vals) * 1.15)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # Panel (b): Training time (log scale)
    bars2 = ax2.bar(names, times, color=colors, edgecolor="white", alpha=0.9, width=0.55)
    ax2.set_yscale("log")
    for bar, val in zip(bars2, times):
        label = f"{val} min" if val < 60 else f"{val / 60:g} hrs"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.2,
                 label, ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Training Time (min, log scale)")
    ax2.set_title("(b) Computational Cost (5-Fold CV)", fontweight="bold")
    ax2.tick_params(axis="x", labelsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(RESULTS_DIR, f"rf_training_comparison.{ext}"), dpi=300)
    plt.close()
    print("  [OK] Figure 9: rf_training_comparison.pdf/png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7a: Feature importance — fix empty blue legend
# ═════════════════════════════════════════════════════════════════════════════

def figure_7a():
    """Regenerate feature importance with legend matching actual data."""
    from collections import Counter

    # Chemical labels for known Morgan FP bits
    CHEMICAL_LABELS = {
        1057: "Azo group (N=N)",
        389:  "Phenol / ArOH",
        1476: "Sulfoxide (S=O)",
        314:  "Carbonyl (C=O)",
        420:  "Alkynyl silane (C≡C–Si)",
        1052: "Iminium (C=N⁺)",
        694:  "Aryl nitrile (Ar–C≡N)",
        1984: "Aromatic ring (Ar)",
        593:  "Phosphonium ylide",
        1189: "Azo–heterocycle",
        222:  "Biaryl linkage",
        547:  "Alkenyl chain (C=C–C)",
        791:  "Pyridinium (Ar–N⁺)",
        570:  "Ethynyl silane",
        1049: "Aminoaryl (Ar–NH)",
    }

    # Load RF models to get feature importances
    import joblib
    importances = []
    for i in range(5):
        model = joblib.load(f"{RESULTS_DIR}/rf_tuned_fold{i}_model.joblib")
        importances.append(model.feature_importances_)
    avg_imp = np.mean(importances, axis=0)
    std_imp = np.std(importances, axis=0)

    # Load dataset to compute delta lambda_max for each bit
    data = pd.read_csv("previous_code/UV_canonical_full_dataset.csv")
    # Filter out rows with NaN lambda_max (2944 NaN values in raw dataset)
    valid_mask = data["lambda_max"].notna()
    smiles_arr = data.loc[valid_mask, "canon"].values
    y_arr = data.loc[valid_mask, "lambda_max"].values.astype(float)
    print(f"  Using {len(smiles_arr)} molecules with valid lambda_max")

    # Compute Morgan FPs for bit analysis
    from rdkit import Chem
    from rdkit.Chem import AllChem

    print("  Computing Morgan FPs for bit analysis...")
    bit_data = {}
    for bit_idx in range(2048):
        bit_data[bit_idx] = {"lmax_with": [], "lmax_without": []}

    for smi, lmax in zip(smiles_arr, y_arr):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        on_bits = set(fp.GetOnBits())
        for bit_idx in range(2048):
            if bit_idx in on_bits:
                bit_data[bit_idx]["lmax_with"].append(lmax)
            else:
                bit_data[bit_idx]["lmax_without"].append(lmax)

    # Get top 15 solute bits
    top_bits = np.argsort(avg_imp[:2048])[::-1][:15]

    labels = []
    importances_pct = []
    stds_pct = []
    colors = []
    delta_lmax = []

    has_red = False
    has_blue = False
    has_gray = False

    for bit in top_bits:
        bit = int(bit)
        imp = avg_imp[bit] * 100
        sd = std_imp[bit] * 100
        label = CHEMICAL_LABELS.get(bit, f"Bit {bit}")

        with_vals = bit_data[bit]["lmax_with"]
        without_vals = bit_data[bit]["lmax_without"]
        if with_vals and without_vals:
            delta = float(np.nanmean(with_vals) - np.nanmean(without_vals))
        else:
            delta = 0.0

        labels.append(label)
        importances_pct.append(imp)
        stds_pct.append(sd)
        delta_lmax.append(delta)

        if delta > 20:
            colors.append("#c0392b")
            has_red = True
        elif delta < -20:
            colors.append("#2980b9")
            has_blue = True
        else:
            colors.append("#7f8c8d")
            has_gray = True

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, importances_pct, xerr=stds_pct, height=0.7,
                   color=colors, edgecolor="white", capsize=3, alpha=0.9)

    # Add Δλ_max annotations
    for i, (imp, sd, delta) in enumerate(zip(importances_pct, stds_pct, delta_lmax)):
        sign = "+" if delta > 0 else ""
        ax.text(imp + sd + 0.3, i, f"Δλ={sign}{delta:.0f} nm",
                va="center", fontsize=8, color="#555555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (%)")
    ax.set_title("Top 15 Molecular Substructures Driving UV Absorption Prediction")

    # Only include legend entries for colors that actually appear
    legend_handles = []
    if has_red:
        legend_handles.append(mpatches.Patch(color="#c0392b", label="Bathochromic (red shift)"))
    if has_blue:
        legend_handles.append(mpatches.Patch(color="#2980b9", label="Hypsochromic (blue shift)"))
    if has_gray:
        legend_handles.append(mpatches.Patch(color="#7f8c8d", label="Neutral / ubiquitous"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(RESULTS_DIR, f"rf_feature_importance.{ext}"), dpi=300)
    plt.close()
    print(f"  [OK] Figure 7a: rf_feature_importance.pdf/png (red={has_red}, blue={has_blue}, gray={has_gray})")


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: Residual distribution plots for all 4 models
# ═════════════════════════════════════════════════════════════════════════════

def figure_residuals(models):
    """3×2 residual distribution histograms + residual vs actual scatter for 5 models."""
    model_names = list(MODEL_COLORS.keys())

    # Panel 1: Residual histograms (2×3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, name in enumerate(model_names):
        y_pred, y_test, _ = models[name]
        residuals = y_pred - y_test
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        ax = axes[i]
        ax.hist(residuals, bins=80, color=MODEL_COLORS[name], alpha=0.8,
                edgecolor="white", density=True)
        ax.axvline(0, color="black", linestyle="--", lw=1, alpha=0.7)
        ax.axvline(np.mean(residuals), color="red", linestyle="-", lw=1.5, alpha=0.7)

        ax.text(0.97, 0.95,
                f"Mean: {np.mean(residuals):.1f}\n"
                f"Std: {np.std(residuals):.1f}\n"
                f"RMSE: {rmse:.1f}\n"
                f"MAE: {mae:.1f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor="white", alpha=0.8))

        ax.set_xlabel("Residual (predicted − actual, nm)")
        ax.set_ylabel("Density")
        ax.set_title(name, fontweight="bold")
        ax.set_xlim(-200, 200)

    axes[5].set_visible(False)

    plt.suptitle("Residual Distributions — 5-Fold CV", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(RESULTS_DIR, f"residual_distributions.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Supplementary: residual_distributions.pdf/png")

    # Panel 2: Residual vs actual scatter (2×3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, name in enumerate(model_names):
        y_pred, y_test, _ = models[name]
        residuals = y_pred - y_test

        ax = axes[i]
        ax.scatter(y_test, residuals, alpha=0.05, s=3, color=MODEL_COLORS[name], rasterized=True)
        ax.axhline(0, color="black", linestyle="--", lw=1, alpha=0.7)
        ax.set_xlabel("Actual $\\lambda_{\\max}$ (nm)")
        ax.set_ylabel("Residual (nm)")
        ax.set_title(name, fontweight="bold")
        ax.set_ylim(-200, 200)

    axes[5].set_visible(False)

    plt.suptitle("Residuals vs Actual — 5-Fold CV", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(RESULTS_DIR, f"residual_vs_actual.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Supplementary: residual_vs_actual.pdf/png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  REGENERATING IMPROVED ERROR ANALYSIS FIGURES")
    print("=" * 70)

    # Load all model predictions
    print("\nLoading model predictions...")
    models = load_all_models()

    # Load solvent data for Figure 6b
    solvents = load_solvents()

    # Figure 6a: Error by wavelength (all 4 models)
    print("\nFigure 6a: Error by wavelength range...")
    figure_6a(models)

    # Figure 6b: Error by solvent (all 4 models)
    print("\nFigure 6b: Error by solvent...")
    figure_6b(models, solvents)

    # Figure 9: Training comparison (add ChemBERTa)
    print("\nFigure 9: Training comparison...")
    figure_9()

    # Figure 7a: Feature importance (fix legend)
    print("\nFigure 7a: Feature importance...")
    figure_7a()

    # Supplementary: Residual plots
    print("\nSupplementary: Residual plots...")
    figure_residuals(models)

    print("\n" + "=" * 70)
    print("  ALL FIGURES REGENERATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
