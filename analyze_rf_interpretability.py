#!/usr/bin/env python3
"""
RF Model Interpretability Analysis for UV λ_max Prediction.

Generates publication-quality figures and tables demonstrating that
RF + Morgan fingerprints provides interpretable, chemically meaningful
predictions while matching or exceeding deep learning baselines.

Outputs:
  results/rf_feature_importance.pdf       — Top 15 substructure importance bar chart
  results/rf_solute_vs_solvent.pdf        — Solute vs solvent importance breakdown
  results/rf_cumulative_importance.pdf    — Cumulative importance curve
  results/rf_training_comparison.pdf      — Training time & performance comparison
  results/rf_interpretability_table.csv   — Top features with decoded substructures

Usage:
  python3 analyze_rf_interpretability.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_PATH = os.path.join(SCRIPT_DIR, "previous_code", "UV_canonical_full_dataset.csv")
N_FOLDS = 5


def load_rf_importances():
    """Load and average feature importances across all 5 RF v2 folds."""
    all_imp = []
    for fold in range(N_FOLDS):
        path = os.path.join(RESULTS_DIR, f"rf_tuned_fold{fold}_model.joblib")
        model = joblib.load(path)
        all_imp.append(model.feature_importances_)
    avg = np.mean(all_imp, axis=0)
    std = np.std(all_imp, axis=0)
    return avg, std


def decode_morgan_bits(top_bits, n_sample=3000):
    """Decode Morgan FP bits to molecular substructures using RDKit bitInfo."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    df = pd.read_csv(DATA_PATH)
    df = df[["canon", "lambda_max"]].dropna()
    unique_smiles = df["canon"].unique()

    np.random.seed(42)
    sample = np.random.choice(unique_smiles, min(n_sample, len(unique_smiles)), replace=False)

    bit_data = {int(b): {"fragments": [], "lmax_with": [], "lmax_without": []} for b in top_bits}

    for smi in sample:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bi)
        mol_lmax = df[df["canon"] == smi]["lambda_max"].mean()

        for bit in top_bits:
            bit = int(bit)
            if bit in bi:
                bit_data[bit]["lmax_with"].append(mol_lmax)
                atom_idx, radius = bi[bit][0]
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                if submol.GetNumAtoms() > 0:
                    bit_data[bit]["fragments"].append(Chem.MolToSmiles(submol))
            else:
                bit_data[bit]["lmax_without"].append(mol_lmax)

    return bit_data


# Chemical interpretation of top Morgan FP bits (manually curated from decoded SMARTS)
CHEMICAL_LABELS = {
    463:  "Aryl diazo / hydrazine",
    554:  "BODIPY core (B–N⁺)",
    294:  "Alkyl chain (C–C)",
    1955: "Enamine (C=C–N)",
    84:   "Vinyl / conjugation (C=C–C)",
    148:  "Boron heterocycle (B–N)",
    378:  "Thienyl substituent",
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


def plot_feature_importance(avg_imp, std_imp, bit_data):
    """Bar chart of top 15 substructures with chemical labels."""
    top_bits = np.argsort(avg_imp[:2048])[::-1][:15]

    labels = []
    importances = []
    stds = []
    colors = []
    delta_lmax = []

    for bit in top_bits:
        bit = int(bit)
        imp = avg_imp[bit] * 100
        sd = std_imp[bit] * 100
        label = CHEMICAL_LABELS.get(bit, f"Bit {bit}")

        with_vals = bit_data[bit]["lmax_with"]
        without_vals = bit_data[bit]["lmax_without"]
        delta = np.mean(with_vals) - np.mean(without_vals) if with_vals and without_vals else 0

        labels.append(label)
        importances.append(imp)
        stds.append(sd)
        delta_lmax.append(delta)
        # Color by whether feature shifts λ_max red (positive) or blue (negative)
        colors.append("#c0392b" if delta > 20 else "#2980b9" if delta < -20 else "#7f8c8d")

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, importances, xerr=stds, height=0.7,
                   color=colors, edgecolor="white", capsize=3, alpha=0.9)

    # Add Δλ_max annotations
    for i, (imp, delta) in enumerate(zip(importances, delta_lmax)):
        sign = "+" if delta > 0 else ""
        ax.text(imp + stds[i] + 0.3, i, f"Δλ={sign}{delta:.0f} nm",
                va="center", fontsize=8, color="#555555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (%)", fontsize=12)
    ax.set_title("Top 15 Molecular Substructures Driving UV Absorption Prediction", fontsize=13)

    # Legend
    red_patch = mpatches.Patch(color="#c0392b", label="Bathochromic (red shift)")
    blue_patch = mpatches.Patch(color="#2980b9", label="Hypsochromic (blue shift)")
    gray_patch = mpatches.Patch(color="#7f8c8d", label="Neutral / ubiquitous")
    ax.legend(handles=[red_patch, blue_patch, gray_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(RESULTS_DIR, f"rf_feature_importance.{ext}"), dpi=300)
    plt.close()
    print("  Feature importance chart saved.")


def plot_solute_vs_solvent(avg_imp):
    """Pie chart + bar chart showing solute vs solvent importance."""
    solute_sum = avg_imp[:2048].sum()
    solvent_sum = avg_imp[2048:].sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        [solute_sum, solvent_sum],
        labels=["Solute\nstructure", "Solvent\nidentity"],
        autopct="%1.1f%%",
        colors=["#3498db", "#e67e22"],
        startangle=90,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight("bold")
    ax1.set_title("Importance by Component", fontsize=12, fontweight="bold")

    # Bar chart: top solute bits vs top solvent bits
    top_solute = np.argsort(avg_imp[:2048])[::-1][:10]
    top_solvent = np.argsort(avg_imp[2048:])[::-1][:5]

    x_labels = [CHEMICAL_LABELS.get(int(b), f"Bit {b}") for b in top_solute]
    x_labels += [f"Solvent bit {b}" for b in top_solvent]
    x_vals = [avg_imp[int(b)] * 100 for b in top_solute]
    x_vals += [avg_imp[2048 + int(b)] * 100 for b in top_solvent]
    x_colors = ["#3498db"] * 10 + ["#e67e22"] * 5

    y_pos = np.arange(len(x_labels))
    ax2.barh(y_pos, x_vals, color=x_colors, edgecolor="white", alpha=0.9, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(x_labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Feature Importance (%)", fontsize=11)
    ax2.set_title("Top Solute vs Solvent Features", fontsize=12, fontweight="bold")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(RESULTS_DIR, f"rf_solute_vs_solvent.{ext}"), dpi=300)
    plt.close()
    print("  Solute vs solvent chart saved.")


def plot_cumulative_importance(avg_imp):
    """Cumulative importance curve showing how few features explain most variance."""
    sorted_imp = np.sort(avg_imp)[::-1]
    cumsum = np.cumsum(sorted_imp) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumsum) + 1), cumsum, color="#2c3e50", lw=2)

    # Annotate key thresholds
    for pct in [50, 80, 90, 95]:
        idx = np.searchsorted(cumsum, pct)
        ax.axhline(pct, color="#bdc3c7", linestyle="--", lw=0.8)
        ax.axvline(idx + 1, color="#bdc3c7", linestyle="--", lw=0.8)
        ax.annotate(f"{idx + 1} features → {pct}%",
                    xy=(idx + 1, pct), xytext=(idx + 50, pct - 5),
                    fontsize=9, color="#7f8c8d",
                    arrowprops=dict(arrowstyle="->", color="#bdc3c7"))

    ax.set_xlim(0, 500)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Number of Features (ranked by importance)", fontsize=11)
    ax.set_ylabel("Cumulative Importance (%)", fontsize=11)
    ax.set_title("RF Model Sparsity: Few Substructures Explain Most Predictions", fontsize=12)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(RESULTS_DIR, f"rf_cumulative_importance.{ext}"), dpi=300)
    plt.close()
    print("  Cumulative importance curve saved.")


def plot_training_comparison():
    """Bar chart comparing training time and performance across models."""
    models = {
        "RF + Morgan FP": {
            "rmse": 31.34, "mae": 15.16,
            "train_time_min": 15,  # ~3 min/fold × 5 folds (1000 trees)
            "params": "1000 trees\n4096 features",
            "color": "#27ae60",
        },
        "XGBoost + Morgan FP": {
            "rmse": 33.70, "mae": 20.05,
            "train_time_min": 5,   # very fast
            "params": "500 trees\n4096 features",
            "color": "#f39c12",
        },
        "BiGRU + Solvent": {
            "rmse": 36.45, "mae": 20.70,
            "train_time_min": 1500,  # ~5 hrs/fold × 5 folds
            "params": "470K params\nGPU required",
            "color": "#e74c3c",
        },
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    names = list(models.keys())
    rmse_vals = [models[n]["rmse"] for n in names]
    colors = [models[n]["color"] for n in names]
    times = [models[n]["train_time_min"] for n in names]

    # Panel 1: RMSE comparison
    bars1 = ax1.bar(names, rmse_vals, color=colors, edgecolor="white", alpha=0.9, width=0.6)
    for bar, val in zip(bars1, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMSE (nm)", fontsize=12)
    ax1.set_title("Prediction Accuracy", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(rmse_vals) * 1.15)
    ax1.tick_params(axis="x", labelsize=9)

    # Panel 2: Training time (log scale)
    bars2 = ax2.bar(names, times, color=colors, edgecolor="white", alpha=0.9, width=0.6)
    ax2.set_yscale("log")
    for bar, val in zip(bars2, times):
        label = f"{val} min" if val < 60 else f"{val/60:.0f} hrs"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                 label, ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Training Time (min, log scale)", fontsize=12)
    ax2.set_title("Computational Cost (5-fold CV)", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(RESULTS_DIR, f"rf_training_comparison.{ext}"), dpi=300)
    plt.close()
    print("  Training comparison chart saved.")


def save_interpretability_table(avg_imp, std_imp, bit_data):
    """Save CSV table of top features with chemical interpretations."""
    top_bits = np.argsort(avg_imp[:2048])[::-1][:20]

    rows = []
    for rank, bit in enumerate(top_bits, 1):
        bit = int(bit)
        info = bit_data.get(bit, {"fragments": [], "lmax_with": [], "lmax_without": []})
        frags = Counter(info["fragments"]).most_common(1)
        top_frag = frags[0][0] if frags else "—"

        lmax_with = np.mean(info["lmax_with"]) if info["lmax_with"] else 0
        lmax_without = np.mean(info["lmax_without"]) if info["lmax_without"] else 0
        delta = lmax_with - lmax_without if info["lmax_with"] and info["lmax_without"] else 0
        freq = len(info["lmax_with"])

        rows.append({
            "Rank": rank,
            "Bit": bit,
            "Importance (%)": round(avg_imp[bit] * 100, 2),
            "Std (%)": round(std_imp[bit] * 100, 3),
            "Chemical Label": CHEMICAL_LABELS.get(bit, "—"),
            "Top SMILES Fragment": top_frag,
            "Frequency": freq,
            "Mean λ_max WITH (nm)": round(lmax_with, 1),
            "Mean λ_max WITHOUT (nm)": round(lmax_without, 1),
            "Δλ_max (nm)": round(delta, 1),
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "rf_interpretability_table.csv")
    df.to_csv(out_path, index=False)
    print(f"  Interpretability table saved to {out_path}")
    print(df.to_string(index=False))
    return df


def main():
    print(f"\n{'=' * 70}")
    print(f"  RF MODEL INTERPRETABILITY ANALYSIS")
    print(f"{'=' * 70}")

    # Load importances
    print("\n  Loading RF v2 models (5 folds)...")
    avg_imp, std_imp = load_rf_importances()

    solute_total = avg_imp[:2048].sum() * 100
    solvent_total = avg_imp[2048:].sum() * 100
    ratio = avg_imp[:2048].sum() / avg_imp[2048:].sum()
    print(f"  Solute importance: {solute_total:.1f}%")
    print(f"  Solvent importance: {solvent_total:.1f}%")
    print(f"  Ratio: {ratio:.1f}:1")

    n_active_solute = (avg_imp[:2048] > 1e-4).sum()
    n_active_solvent = (avg_imp[2048:] > 1e-4).sum()
    print(f"  Active solute bits: {n_active_solute}/2048")
    print(f"  Active solvent bits: {n_active_solvent}/2048")

    # Cumulative importance
    sorted_imp = np.sort(avg_imp)[::-1]
    cumsum = np.cumsum(sorted_imp)
    for n in [5, 10, 20, 50, 100]:
        print(f"  Top {n:3d} features explain {cumsum[n-1]*100:.1f}%")

    # Decode top bits to substructures
    print(f"\n{'─' * 70}")
    print("  Decoding Morgan FP bits to molecular substructures...")
    top_bits = list(np.argsort(avg_imp[:2048])[::-1][:20])
    bit_data = decode_morgan_bits(top_bits)

    # Generate all outputs
    print(f"\n{'─' * 70}")
    print("  Generating figures...")
    plot_feature_importance(avg_imp, std_imp, bit_data)
    plot_solute_vs_solvent(avg_imp)
    plot_cumulative_importance(avg_imp)
    plot_training_comparison()

    print(f"\n{'─' * 70}")
    print("  Generating interpretability table...")
    save_interpretability_table(avg_imp, std_imp, bit_data)

    print(f"\n{'=' * 70}")
    print(f"  INTERPRETABILITY ANALYSIS COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
